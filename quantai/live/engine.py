"""实盘事件循环 —— 忠实迁移自 `src/trading/engine.py`。

`TradingEngine` 是一个**线程化的事件队列分发器**：一个后台线程从 `queue` 取事件，按类型路由到
注入的 `data_feed` / `strategy` / `broker` / `portfolio`（全部鸭子类型、可注入、可单测）：

- MARKET_DATA -> `broker.on_market_data`（盯市）+ `strategy.on_bar`（决策）-> 把决策当 SIGNAL 回灌；
- SIGNAL / ORDER -> `broker.place_order`；
- FILL -> `portfolio.on_fill`；
- LOG -> 透传给外部监听器（UI/TTS）；ERROR -> 记录。

与旧版的差异：去掉了几处裸 `print()`（库不该往 stdout 喷；真正的输出走 LOG/ERROR 事件），
逻辑零改动。重活（真策略/真券商）由 `live.runner` 注入；这里只管"把事件送对地方"。
"""
from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Any, Optional

from quantai.live.events import Event, EventType


class TradingEngine:
    def __init__(self) -> None:
        self.events: "queue.Queue[Event]" = queue.Queue()
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

        self.paused: bool = False

        self.data_feed: Any = None
        self.strategy: Any = None
        self.broker: Any = None
        self.portfolio: Any = None

    # ----------------------------------------------------------------- #
    # 生命周期
    # ----------------------------------------------------------------- #
    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # ----------------------------------------------------------------- #
    # 入队
    # ----------------------------------------------------------------- #
    def push_event(self, event: Event) -> None:
        self.events.put(event)

    def push(self, event_type: EventType, payload: Any, priority: int = 0) -> None:
        self.push_event(
            Event(type=event_type, timestamp=datetime.now(), payload=payload, priority=priority)
        )

    # ----------------------------------------------------------------- #
    # 主循环 / 分发
    # ----------------------------------------------------------------- #
    def _run_loop(self) -> None:
        while self.is_running:
            try:
                event = self.events.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._handle_event(event)
            except Exception as exc:
                self.push(EventType.ERROR, {"error": str(exc), "event": str(event.type)})

    def _handle_event(self, event: Event) -> None:
        if bool(self.paused) and event.type in {
            EventType.MARKET_DATA,
            EventType.SIGNAL,
            EventType.ORDER,
        }:
            return

        if event.type == EventType.MARKET_DATA:
            if self.broker is not None and hasattr(self.broker, "on_market_data"):
                try:
                    self.broker.on_market_data(event.payload)
                except Exception:
                    pass
            if self.strategy is None:
                return
            if hasattr(self.strategy, "on_bar"):
                self._ingest_strategy_output(self.strategy.on_bar(event.payload))
            elif hasattr(self.strategy, "on_event"):
                self._ingest_strategy_output(self.strategy.on_event(event))
            return

        if event.type in {EventType.SIGNAL, EventType.ORDER}:
            if self.broker is not None and hasattr(self.broker, "place_order"):
                self.broker.place_order(event.payload)
            return

        if event.type == EventType.FILL:
            if self.portfolio is not None and hasattr(self.portfolio, "on_fill"):
                self.portfolio.on_fill(event.payload)
            return

        if event.type == EventType.LOG:
            # LOG 由外部监听器处理（UI/TTS）；引擎只透传。
            return

        if event.type == EventType.ERROR:
            return

    def _ingest_strategy_output(self, out: Any) -> None:
        """策略产出 -> SIGNAL 事件。None 忽略；list 逐个回灌。"""
        if out is None:
            return
        if isinstance(out, list):
            for item in out:
                if item is not None:
                    self.push(EventType.SIGNAL, item)
            return
        self.push(EventType.SIGNAL, out)
