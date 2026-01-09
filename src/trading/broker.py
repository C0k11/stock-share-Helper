from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from .event import Event, EventType
from src.learning.recorder import recorder as evolution_recorder


@dataclass
class Position:
    ticker: str
    shares: float
    avg_price: float
    trace_ids: list[str] = field(default_factory=list)


class PaperBroker:
    def __init__(self, engine: Any, cash: float = 100000.0) -> None:
        self.engine = engine
        self.cash = float(cash)
        self.initial_cash = float(cash)
        self.positions: Dict[str, Position] = {}
        self.orders: list[dict] = []

    def place_order(self, signal: dict) -> None:
        def _rej(msg: str) -> None:
            try:
                self.engine.push_event(Event(type=EventType.LOG, timestamp=datetime.now(), payload=str(msg), priority=2))
            except Exception:
                return

        ticker = str(signal.get("ticker") or "").upper().strip()
        action = str(signal.get("action") or "").upper().strip()
        price = float(signal.get("price") or 0.0)
        shares = float(signal.get("shares") or 0.0)
        trace_id = str(signal.get("trace_id") or "").strip() or None
        trace_ids_raw = signal.get("trace_ids")
        trace_ids: list[str] = []
        try:
            if isinstance(trace_ids_raw, (list, tuple)):
                for x in trace_ids_raw:
                    s = str(x or "").strip()
                    if s:
                        trace_ids.append(s)
        except Exception:
            trace_ids = []
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)

        expert = str(signal.get("expert") or "").strip()
        analysis = str(signal.get("analysis") or "")
        chart_score = signal.get("chart_score")
        news_score = signal.get("news_score")
        news_sentiment = signal.get("news_sentiment")
        news_summary = signal.get("news_summary")
        if not ticker or not action or price <= 0 or shares == 0:
            _rej(f"[Broker] ignore invalid order: action={action} ticker={ticker} price={price} shares={shares}")
            return

        commission = float(signal.get("commission") or 0.0)
        notional = price * abs(shares)

        if action == "BUY":
            total_cost = notional + commission
            if total_cost > self.cash:
                _rej(f"[Broker] reject BUY {ticker} x{shares:g} @ {price:.2f}: cash={self.cash:.2f} need={total_cost:.2f}")
                return
            pos = self.positions.get(ticker)
            if pos is not None and float(pos.shares) < 0:
                # Cover short first
                cover_shares = min(abs(float(pos.shares)), abs(shares))
                total_cost = (price * cover_shares) + commission
                if total_cost > self.cash:
                    _rej(f"[Broker] reject COVER {ticker} x{cover_shares:g} @ {price:.2f}: cash={self.cash:.2f} need={total_cost:.2f}")
                    return
                entry_price = float(pos.avg_price)
                self.cash -= total_cost
                remaining = float(pos.shares) + float(cover_shares)
                if remaining >= 0:
                    self.positions.pop(ticker, None)
                else:
                    pos.shares = float(remaining)

                try:
                    if getattr(pos, "trace_ids", None):
                        realized = (entry_price - price) * float(cover_shares) - float(commission)
                        for rid in list(pos.trace_ids):
                            if not rid:
                                continue
                            evolution_recorder.log_outcome(
                                ref_id=str(rid),
                                outcome=float(realized),
                                comment=f"realized_pnl={realized:.4f} entry={entry_price:.4f} cover={price:.4f} shares={cover_shares:.4f}",
                            )
                except Exception:
                    pass

                extra = float(abs(shares) - cover_shares)
                if extra > 0:
                    # Flip to long with remaining buy size
                    total_cost2 = (price * extra)
                    if (total_cost2) > self.cash:
                        _rej(f"[Broker] reject FLIP->LONG {ticker} x{extra:g} @ {price:.2f}: cash={self.cash:.2f} need={total_cost2:.2f}")
                        return
                    self.cash -= total_cost2
                    self.positions[ticker] = Position(ticker=ticker, shares=extra, avg_price=price, trace_ids=list(trace_ids))
            else:
                self.cash -= total_cost
                if pos is None:
                    self.positions[ticker] = Position(ticker=ticker, shares=shares, avg_price=price, trace_ids=list(trace_ids))
                else:
                    new_shares = pos.shares + shares
                    if new_shares == 0:
                        self.positions.pop(ticker, None)
                    else:
                        pos.avg_price = (pos.avg_price * pos.shares + price * shares) / new_shares
                        pos.shares = new_shares
                        try:
                            for rid in list(trace_ids):
                                if rid and rid not in pos.trace_ids:
                                    pos.trace_ids.append(rid)
                        except Exception:
                            pass

        elif action == "SELL":
            pos = self.positions.get(ticker)
            if pos is None:
                sell_shares = abs(shares)
                proceeds = price * sell_shares - commission
                self.cash += proceeds
                self.positions[ticker] = Position(ticker=ticker, shares=-float(sell_shares), avg_price=price, trace_ids=list(trace_ids))
            elif float(pos.shares) <= 0:
                sell_shares = abs(shares)
                proceeds = price * sell_shares - commission
                self.cash += proceeds
                prev_abs = abs(float(pos.shares))
                new_abs = prev_abs + float(sell_shares)
                if new_abs > 0:
                    pos.avg_price = (float(pos.avg_price) * prev_abs + price * float(sell_shares)) / new_abs
                pos.shares = -float(new_abs)
                try:
                    for rid in list(trace_ids):
                        if rid and rid not in pos.trace_ids:
                            pos.trace_ids.append(rid)
                except Exception:
                    pass
            else:
                sell_shares = min(float(pos.shares), abs(shares))
                entry_price = float(pos.avg_price)
                proceeds = price * sell_shares - commission
                self.cash += proceeds
                remaining = float(pos.shares) - float(sell_shares)
                if remaining <= 0:
                    self.positions.pop(ticker, None)
                else:
                    pos.shares = float(remaining)

                try:
                    if getattr(pos, "trace_ids", None):
                        realized = (price - entry_price) * float(sell_shares) - float(commission)
                        for rid in list(pos.trace_ids):
                            if not rid:
                                continue
                            evolution_recorder.log_outcome(
                                ref_id=str(rid),
                                outcome=float(realized),
                                comment=f"realized_pnl={realized:.4f} entry={entry_price:.4f} exit={price:.4f} shares={sell_shares:.4f}",
                            )
                except Exception:
                    pass
        else:
            _rej(f"[Broker] ignore unsupported action: {action}")
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
            "trace_ids": trace_ids,
            "expert": expert,
            "analysis": analysis,
            "chart_score": chart_score,
            "news_score": news_score,
            "news_sentiment": news_sentiment,
            "news_summary": news_summary,
        }
        self.engine.push_event(Event(type=EventType.FILL, timestamp=datetime.now(), payload=fill))
