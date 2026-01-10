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
        self.last_price: Dict[str, float] = {}
        self.max_leverage = 3.0
        self.initial_margin = 0.35
        self.maintenance_margin = 0.25

    def _mark_price(self, ticker: str, *, fallback: float = 0.0) -> float:
        try:
            p = float(self.last_price.get(str(ticker).upper(), 0.0) or 0.0)
            if p > 0:
                return p
        except Exception:
            pass
        try:
            if fallback and float(fallback) > 0:
                return float(fallback)
        except Exception:
            pass
        try:
            pos = self.positions.get(str(ticker).upper())
            if pos is not None:
                ap = float(getattr(pos, "avg_price", 0.0) or 0.0)
                if ap > 0:
                    return ap
        except Exception:
            pass
        return 0.0

    def _compute_equity_gross(self, *, cash: float, pos_shares: Dict[str, float], marks: Dict[str, float]) -> tuple[float, float]:
        eq = float(cash)
        gross = 0.0
        for tk, sh in (pos_shares or {}).items():
            try:
                sh_f = float(sh or 0.0)
            except Exception:
                continue
            if sh_f == 0.0:
                continue
            try:
                px = float((marks or {}).get(str(tk).upper(), 0.0) or 0.0)
            except Exception:
                px = 0.0
            if px <= 0.0:
                continue
            eq += sh_f * px
            gross += abs(sh_f) * px
        return float(eq), float(gross)

    def _pretrade_risk_check(self, *, ticker: str, action: str, price: float, shares: float, commission: float) -> tuple[bool, str]:
        tk = str(ticker).upper().strip()
        try:
            px = float(price)
        except Exception:
            px = 0.0
        if not (tk and px > 0.0):
            return False, "invalid_price"

        try:
            sh0 = float(abs(shares) or 0.0)
        except Exception:
            sh0 = 0.0
        if sh0 <= 0.0:
            return False, "invalid_shares"

        cash_new = float(self.cash)
        pos_shares: Dict[str, float] = {}
        marks: Dict[str, float] = {}
        try:
            for k, p in (self.positions or {}).items():
                if p is None:
                    continue
                pos_shares[str(k).upper()] = float(getattr(p, "shares", 0.0) or 0.0)
                marks[str(k).upper()] = self._mark_price(str(k).upper(), fallback=float(getattr(p, "avg_price", 0.0) or 0.0))
        except Exception:
            pos_shares = {}
            marks = {}
        marks[tk] = float(px)

        cur = float(pos_shares.get(tk, 0.0) or 0.0)
        act = str(action or "").upper().strip()

        if act == "BUY":
            buy_sh = float(sh0)
            if cur < 0.0:
                cover = min(abs(cur), buy_sh)
                cash_new -= (px * cover) + float(commission)
                cur = cur + cover
                buy_sh = float(buy_sh - cover)
                if buy_sh > 0.0:
                    cash_new -= px * buy_sh
                    cur = cur + buy_sh
            else:
                cash_new -= (px * buy_sh) + float(commission)
                cur = cur + buy_sh
        elif act == "SELL":
            sell_sh = float(sh0)
            if cur > 0.0:
                sell2 = min(cur, sell_sh)
                cash_new += (px * sell2) - float(commission)
                cur = cur - sell2
            else:
                cash_new += (px * sell_sh) - float(commission)
                cur = cur - sell_sh
        else:
            return False, "unsupported_action"

        if abs(cur) < 1e-12:
            pos_shares.pop(tk, None)
        else:
            pos_shares[tk] = float(cur)

        eq, gross = self._compute_equity_gross(cash=cash_new, pos_shares=pos_shares, marks=marks)
        if gross <= 0.0:
            return True, "ok"
        if not (eq > 0.0):
            return False, f"equity_nonpositive eq={eq:.2f}"

        try:
            ml = float(getattr(self, "max_leverage", 3.0) or 3.0)
        except Exception:
            ml = 3.0
        ml = max(1.0, min(ml, 50.0))

        lev = gross / eq
        if lev > float(ml) + 1e-9:
            return False, f"leverage_exceeded lev={lev:.2f} max={ml:.2f}"

        try:
            im = float(getattr(self, "initial_margin", 0.35) or 0.35)
        except Exception:
            im = 0.35
        im = max(0.01, min(im, 1.0))
        if eq < (im * gross) - 1e-6:
            return False, f"margin_exceeded eq={eq:.2f} req={im * gross:.2f}"

        return True, "ok"

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

        ok, why = self._pretrade_risk_check(
            ticker=ticker,
            action=action,
            price=price,
            shares=shares,
            commission=commission,
        )
        if not bool(ok):
            _rej(f"[Broker] reject {action} {ticker} x{shares:g} @ {price:.2f}: {why}")
            return

        if action == "BUY":
            total_cost = notional + commission
            pos = self.positions.get(ticker)
            if pos is not None and float(pos.shares) < 0:
                # Cover short first
                cover_shares = min(abs(float(pos.shares)), abs(shares))
                total_cost = (price * cover_shares) + commission
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
        try:
            self.last_price[str(ticker).upper()] = float(price)
        except Exception:
            pass
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
