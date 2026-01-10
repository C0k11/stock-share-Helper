from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Any, Optional

from .event import Event, EventType


class TradingEngine:
    def __init__(self) -> None:
        self.events: "queue.Queue[Event]" = queue.Queue()
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

        self.data_feed: Any = None
        self.strategy: Any = None
        self.broker: Any = None
        self.portfolio: Any = None

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("Trading Engine Started (Event Loop)")

    def stop(self) -> None:
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        print("Trading Engine Stopped")

    def push_event(self, event: Event) -> None:
        self.events.put(event)

    def push(self, event_type: EventType, payload: Any) -> None:
        self.push_event(Event(type=event_type, timestamp=datetime.now(), payload=payload))

    def _run_loop(self) -> None:
        while self.is_running:
            try:
                event = self.events.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._handle_event(event)
            except Exception as e:
                self.push(EventType.ERROR, {"error": str(e), "event": str(event.type)})

    def _handle_event(self, event: Event) -> None:
        if event.type == EventType.MARKET_DATA:
            try:
                if self.broker is not None and hasattr(self.broker, "on_market_data"):
                    self.broker.on_market_data(event.payload)
            except Exception:
                pass
            if self.strategy is None:
                return
            if hasattr(self.strategy, "on_bar"):
                out = self.strategy.on_bar(event.payload)
                self._ingest_strategy_output(out)
                return
            if hasattr(self.strategy, "on_event"):
                out = self.strategy.on_event(event)
                self._ingest_strategy_output(out)
                return
            return

        if event.type == EventType.SIGNAL:
            if self.broker is None:
                return
            if hasattr(self.broker, "place_order"):
                self.broker.place_order(event.payload)
            return

        if event.type == EventType.ORDER:
            if self.broker is None:
                return
            if hasattr(self.broker, "place_order"):
                self.broker.place_order(event.payload)
            return

        if event.type == EventType.FILL:
            if self.portfolio is None:
                return
            if hasattr(self.portfolio, "on_fill"):
                self.portfolio.on_fill(event.payload)
            return

        if event.type == EventType.LOG:
            # LOG events are handled by external listeners (e.g., Mari TTS)
            # Engine just passes them through
            return

        if event.type == EventType.ERROR:
            try:
                print(f"engine >> ERROR: {event.payload}")
            except Exception:
                pass

    def _ingest_strategy_output(self, out: Any) -> None:
        if out is None:
            return
        if isinstance(out, list):
            for x in out:
                self.push(EventType.SIGNAL, x)
            return
        self.push(EventType.SIGNAL, out)
