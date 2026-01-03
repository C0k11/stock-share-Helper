from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from .event import Event, EventType


@dataclass
class Position:
    ticker: str
    shares: float
    avg_price: float


class PaperBroker:
    def __init__(self, engine: Any, cash: float = 100000.0) -> None:
        self.engine = engine
        self.cash = float(cash)
        self.positions: Dict[str, Position] = {}
        self.orders: list[dict] = []

    def place_order(self, signal: dict) -> None:
        ticker = str(signal.get("ticker") or "").upper().strip()
        action = str(signal.get("action") or "").upper().strip()
        price = float(signal.get("price") or 0.0)
        shares = float(signal.get("shares") or 0.0)
        if not ticker or not action or price <= 0 or shares == 0:
            return

        self.orders.append({"ticker": ticker, "action": action, "price": price, "shares": shares})
        print(f"broker >> Processing Order: {action} {ticker} {shares} @ {price}")

        fill = {
            "ticker": ticker,
            "price": price,
            "shares": shares,
            "action": action,
            "commission": 0.0,
        }
        self.engine.push_event(Event(type=EventType.FILL, timestamp=datetime.now(), payload=fill))
