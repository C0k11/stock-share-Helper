"""实盘/模拟运行器 —— 把数据源、事件引擎、券商、策略一行装配起来。

取代旧 103KB `run_live_paper_trading.py` 的**装配核心**（不含其大量 UI/TTS/CLI 杂项）：

    DataFeed --(MARKET_DATA)--> TradingEngine --> AgentStrategy.on_bar
        --> orchestrator.decide --> 信号 --(SIGNAL)--> PaperBroker.place_order
        --> FILL/LOG 事件回灌引擎

券商通过注入回调把 FILL/LOG 灌回引擎（`execution/` 不依赖 `live/`，见 broker 文档）。
默认用纯 `AppConfig()` 装配大脑（零配置即可跑）；调用方可传 `load_config()` 或自建 orchestrator。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from quantai.agents.orchestrator import AgentOrchestrator
from quantai.config.schema import AppConfig
from quantai.execution.broker import PaperBroker
from quantai.live.data_feed import create_data_feed
from quantai.live.engine import TradingEngine
from quantai.live.events import EventType
from quantai.live.strategy_adapter import AgentStrategy


class LiveRunner:
    """组合 feed + engine + broker + strategy 的实盘/模拟运行器。"""

    def __init__(
        self,
        tickers: List[str],
        *,
        cfg: Optional[AppConfig] = None,
        orchestrator: Any = None,
        source: str = "auto",
        interval_sec: float = 5.0,
        cash: float = 100000.0,
        llm: Any = None,
        allow_short: bool = False,
        seed: Optional[int] = None,
        base_prices: Optional[Dict[str, float]] = None,
        recorder: Any = None,
        collect: bool = False,
    ) -> None:
        # 数据飞轮采集（opt-in）：collect=True 时按 cfg 建 EvolutionRecorder 注入券商+策略，
        # 形成"决策 record -> 平仓 log_outcome 回填盈亏"的闭环。默认关（不产生磁盘写）。
        if recorder is None and collect:
            from quantai.evolution import EvolutionRecorder

            recorder = EvolutionRecorder.from_config((cfg or AppConfig()).evolution)
        self.recorder = recorder

        self.engine = TradingEngine()
        self.broker = PaperBroker(
            cash=cash,
            on_fill=lambda fill: self.engine.push(EventType.FILL, fill),
            on_log=lambda msg, prio: self.engine.push(EventType.LOG, msg, prio),
            recorder=recorder,
        )

        if orchestrator is None:
            orchestrator = AgentOrchestrator.from_config(cfg or AppConfig())
        self.orchestrator = orchestrator

        self.strategy = AgentStrategy(
            orchestrator,
            broker=self.broker,
            llm=llm,
            allow_short=allow_short,
            on_log=lambda msg, prio=2: self.engine.push(EventType.LOG, msg, prio),
            recorder=recorder,
        )

        self.feed = create_data_feed(
            tickers,
            source=source,
            interval_sec=interval_sec,
            base_prices=base_prices,
            seed=seed,
        )

        self.engine.broker = self.broker
        self.engine.strategy = self.strategy
        self.feed.subscribe(self._on_market_data)

    # ----------------------------------------------------------------- #
    def _on_market_data(self, md: dict) -> None:
        self.engine.push(EventType.MARKET_DATA, md)

    def start(self) -> None:
        self.engine.start()
        self.feed.start()

    def stop(self) -> None:
        try:
            self.feed.stop()
        finally:
            self.engine.stop()

    def snapshot(self) -> Dict[str, Any]:
        """当前账户/持仓快照（给 UI / API 读）。"""
        positions = {
            tk: {"shares": float(p.shares), "avg_price": float(p.avg_price)}
            for tk, p in (self.broker.positions or {}).items()
            if p is not None
        }
        return {
            "cash": float(self.broker.cash),
            "equity": float(self.broker.equity()),
            "positions": positions,
            "orders": len(self.broker.orders),
            "feed_source": getattr(self.feed, "source", "unknown"),
        }
