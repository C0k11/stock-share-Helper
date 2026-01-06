from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from .event import Event, EventType
from src.learning.recorder import recorder as evolution_recorder


@dataclass
class Position:
    ticker: str
    shares: float
    avg_price: float


class PaperBroker:
    def __init__(self, engine: Any, cash: float = 100000.0) -> None:
        self.engine = engine
        self.cash = float(cash)
        self.initial_cash = float(cash)
        self.positions: Dict[str, Position] = {}
        self.orders: list[dict] = []

    def place_order(self, signal: dict) -> None:
        ticker = str(signal.get("ticker") or "").upper().strip()
        action = str(signal.get("action") or "").upper().strip()
        price = float(signal.get("price") or 0.0)
        shares = float(signal.get("shares") or 0.0)
        trace_id = str(signal.get("trace_id") or "").strip() or None
        if not ticker or not action or price <= 0 or shares == 0:
            return

        commission = float(signal.get("commission") or 0.0)
        notional = price * abs(shares)

        if action == "BUY":
            total_cost = notional + commission
            if total_cost > self.cash:
                return
            self.cash -= total_cost
            pos = self.positions.get(ticker)
            if pos is None:
                self.positions[ticker] = Position(ticker=ticker, shares=shares, avg_price=price)
            else:
                new_shares = pos.shares + shares
                if new_shares == 0:
                    self.positions.pop(ticker, None)
                else:
                    pos.avg_price = (pos.avg_price * pos.shares + price * shares) / new_shares
                    pos.shares = new_shares

        elif action == "SELL":
            pos = self.positions.get(ticker)
            if pos is None or pos.shares <= 0:
                return
            sell_shares = min(pos.shares, abs(shares))
            entry_price = float(pos.avg_price)
            proceeds = price * sell_shares - commission
            self.cash += proceeds
            remaining = pos.shares - sell_shares
            if remaining <= 0:
                self.positions.pop(ticker, None)
            else:
                pos.shares = remaining

            try:
                if trace_id:
                    realized = (price - entry_price) * float(sell_shares) - float(commission)
                    evolution_recorder.log_outcome(
                        ref_id=trace_id,
                        outcome=float(realized),
                        comment=f"realized_pnl={realized:.4f} entry={entry_price:.4f} exit={price:.4f} shares={sell_shares:.4f}",
                    )
            except Exception:
                pass
        else:
            return

        self.orders.append({"ticker": ticker, "action": action, "price": price, "shares": shares, "trace_id": trace_id})
        print(f"broker >> Processing Order: {action} {ticker} {shares} @ {price}")

        fill = {
            "ticker": ticker,
            "price": price,
            "shares": shares,
            "action": action,
            "commission": commission,
            "trace_id": trace_id,
        }
        self.engine.push_event(Event(type=EventType.FILL, timestamp=datetime.now(), payload=fill))
