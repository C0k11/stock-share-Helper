"""纸面券商（PaperBroker）核心撮合 —— 迁自 `src/trading/broker.py` 的核心层。

只迁**核心撮合**：持仓/现金记账、BUY/SELL（含空头回补/反手/加仓均价）、
盘前风控（杠杆/保证金/单标的）、盯市、保证金利息 + 借券费计提、维持保证金自动强平、FILL 事件。

**未迁**：US 市场微结构（熔断 MWCB / LULD / Rule 201 报升 / PDT / T+1 交割 / locate，~700 行，
全部默认关）。判定 niche + 纸面 demo 几乎不触发，主动延后；真实代码仍在 legacy `src/trading/broker.py`，
不删、日后需要可迁。

**解耦**：旧版直接 `engine.push_event(...)` 把 FILL/LOG 灌回引擎；新版改成**注入回调**
`on_fill(fill: dict)` / `on_log(msg, priority)`，让 `execution/` 不依赖 `live/`（保持分层）。
已实现的 PnL 回填（喂数据飞轮）改成注入 `recorder`（需有 `log_outcome`），默认 None=不记。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

FillSink = Callable[[Dict[str, Any]], None]
LogSink = Callable[[str, int], None]


@dataclass
class Position:
    ticker: str
    shares: float
    avg_price: float
    trace_ids: List[str] = field(default_factory=list)


class PaperBroker:
    """纸面券商核心：记账 + 撮合 + 杠杆/保证金风控 + 盯市/计费/强平。"""

    def __init__(
        self,
        cash: float = 100000.0,
        *,
        on_fill: Optional[FillSink] = None,
        on_log: Optional[LogSink] = None,
        recorder: Any = None,
    ) -> None:
        self.cash = float(cash)
        self.initial_cash = float(cash)
        self.positions: Dict[str, Position] = {}
        self.orders: List[dict] = []
        self.last_price: Dict[str, float] = {}

        # 风控/计费参数（可在构造后覆盖）
        self.max_leverage = 3.0
        self.initial_margin = 0.35
        self.maintenance_margin = 0.25
        self.margin_interest_apr = 0.12
        self.short_borrow_fee_apr = 0.03
        self.settlement_interval_sec = 60.0
        self.liquidation_enabled = True
        self.liquidation_commission = 0.0
        self.allow_short = True

        self._last_settle_ts = time.time()
        self._last_liq_log_ts = 0.0
        self._last_fee_log_ts = 0.0

        # 注入的输出接缝（默认 no-op，保持纯净/可测）
        self._on_fill: FillSink = on_fill or (lambda _fill: None)
        self._on_log: LogSink = on_log or (lambda _msg, _prio: None)
        self._recorder = recorder

    # ----------------------------------------------------------------- #
    # 输出接缝
    # ----------------------------------------------------------------- #
    def _log(self, msg: str, *, priority: int = 2) -> None:
        try:
            self._on_log(str(msg), int(priority))
        except Exception:
            pass

    def _emit_fill(self, fill: Dict[str, Any]) -> None:
        try:
            self._on_fill(fill)
        except Exception:
            pass

    def _record_outcome(self, *, trace_ids: List[str], realized: float, comment: str) -> None:
        rec = self._recorder
        if rec is None:
            return
        for rid in list(trace_ids or []):
            if not rid:
                continue
            try:
                rec.log_outcome(ref_id=str(rid), outcome=float(realized), comment=comment)
            except Exception:
                continue

    # ----------------------------------------------------------------- #
    # 行情进来 -> 更新最新价 + 盯市
    # ----------------------------------------------------------------- #
    def on_market_data(self, market_data: dict) -> None:
        md = market_data if isinstance(market_data, dict) else {}
        ticker = str(md.get("ticker") or md.get("symbol") or md.get("code") or "").upper().strip()
        price_raw = md.get("close")
        for key in ("price", "last", "c"):
            if price_raw is None:
                price_raw = md.get(key)
        try:
            price = float(price_raw or 0.0)
        except Exception:
            price = 0.0

        ts0: Optional[float] = None
        t = md.get("time")
        try:
            if hasattr(t, "timestamp"):
                ts0 = float(t.timestamp())
            elif isinstance(t, (int, float)):
                ts0 = float(t)
        except Exception:
            ts0 = None

        if ticker and price > 0.0:
            self.last_price[ticker] = price

        self.mark_to_market(asof_ts=ts0)

    # ----------------------------------------------------------------- #
    # 盯市：按真实时间步进计费 + 维持保证金检查
    # ----------------------------------------------------------------- #
    def mark_to_market(self, *, asof_ts: Optional[float] = None) -> None:
        try:
            now = float(asof_ts) if asof_ts is not None else float(time.time())
        except Exception:
            now = float(time.time())

        last = float(self._last_settle_ts if self._last_settle_ts is not None else now)
        if now <= last:
            return

        interval = max(1.0, min(float(self.settlement_interval_sec or 60.0), 3600.0))
        dt = float(now - last)
        if dt < 0.5:
            return
        n_steps = int(dt // interval)
        if n_steps <= 0:
            return

        step_dt = dt / float(n_steps)
        for _ in range(n_steps):
            self._accrue_fees(step_dt)
            self._maintenance_check_and_liquidate(now)
        self._last_settle_ts = float(now)

    def _accrue_fees(self, dt_sec: float) -> None:
        dt = float(dt_sec or 0.0)
        if dt <= 0.0:
            return
        mi = max(0.0, min(float(self.margin_interest_apr or 0.0), 5.0))
        sb = max(0.0, min(float(self.short_borrow_fee_apr or 0.0), 5.0))
        year_sec = 365.0 * 24.0 * 3600.0

        interest = 0.0
        if float(self.cash) < 0.0 and mi > 0.0:
            interest = (-float(self.cash)) * (mi / year_sec) * dt

        short_value = 0.0
        if sb > 0.0:
            for tk, pos in (self.positions or {}).items():
                if pos is None:
                    continue
                sh = float(pos.shares or 0.0)
                if sh >= 0.0:
                    continue
                px = self._mark_price(str(tk).upper(), fallback=float(pos.avg_price or 0.0))
                if px > 0.0:
                    short_value += abs(sh) * px
        borrow_fee = short_value * (sb / year_sec) * dt if (short_value > 0.0 and sb > 0.0) else 0.0

        total_fee = float(interest + borrow_fee)
        if total_fee > 0.0:
            self.cash -= total_fee
            now = float(time.time())
            if (now - float(self._last_fee_log_ts or 0.0)) >= 30.0:
                self._last_fee_log_ts = now
                self._log(
                    f"[Broker] fees accrued: interest={interest:.4f} "
                    f"borrow_fee={borrow_fee:.4f} cash={float(self.cash):.2f}",
                    priority=1,
                )

    def _maintenance_check_and_liquidate(self, asof_ts: float) -> None:
        if not bool(self.liquidation_enabled):
            return
        marks: Dict[str, float] = {}
        pos_shares: Dict[str, float] = {}
        for tk, p in (self.positions or {}).items():
            if p is None:
                continue
            sh = float(p.shares or 0.0)
            if sh == 0.0:
                continue
            pos_shares[str(tk).upper()] = sh
            marks[str(tk).upper()] = self._mark_price(str(tk).upper(), fallback=float(p.avg_price or 0.0))

        eq, gross = self._compute_equity_gross(cash=float(self.cash), pos_shares=pos_shares, marks=marks)
        if gross <= 0.0:
            return
        mm = max(0.01, min(float(self.maintenance_margin or 0.25), 1.0))
        req = mm * gross
        if eq >= req:
            return
        self._auto_liquidate(marks=marks)

    def _auto_liquidate(self, *, marks: Dict[str, float]) -> None:
        now = float(time.time())
        if (now - float(self._last_liq_log_ts or 0.0)) >= 5.0:
            self._last_liq_log_ts = now
            self._log("[Broker] [Liquidation] start", priority=2)

        mm = max(0.01, min(float(self.maintenance_margin or 0.25), 1.0))
        comm = max(0.0, min(float(self.liquidation_commission or 0.0), 100000.0))

        for _ in range(30):  # 至多 30 步，避免极端死循环
            # 选当前敞口最大的持仓平
            target: Optional[Position] = None
            px = 0.0
            biggest = 0.0
            for tk, p in (self.positions or {}).items():
                if p is None or float(p.shares or 0.0) == 0.0:
                    continue
                mk = float((marks or {}).get(str(tk).upper(), 0.0) or 0.0)
                if mk <= 0.0:
                    continue
                exposure = abs(float(p.shares)) * mk
                if exposure > biggest:
                    biggest, target, px = exposure, p, mk
            if target is None or px <= 0.0:
                return

            pos_shares = {
                str(k).upper(): float(p.shares)
                for k, p in (self.positions or {}).items()
                if p is not None and float(p.shares or 0.0) != 0.0
            }
            eq2, gross2 = self._compute_equity_gross(cash=float(self.cash), pos_shares=pos_shares, marks=marks)
            req2 = mm * gross2
            if gross2 <= 0.0 or eq2 >= req2:
                return

            tk = str(target.ticker or "").upper().strip()
            sh0 = float(target.shares or 0.0)
            deficit = float(req2 - eq2)
            qty = max(1.0, abs(sh0) * 0.25)
            qty = max(qty, deficit / px)
            qty = min(abs(sh0), qty)

            if sh0 > 0.0:  # 平多
                self.cash += px * qty - comm
                new_sh = sh0 - qty
                if new_sh <= 0.0:
                    self.positions.pop(tk, None)
                else:
                    target.shares = new_sh
                action, analysis = "SELL", "maintenance_margin_auto_sell"
            else:  # 平空
                self.cash -= px * qty + comm
                new_sh = sh0 + qty
                if new_sh >= 0.0:
                    self.positions.pop(tk, None)
                else:
                    target.shares = new_sh
                action, analysis = "BUY", "maintenance_margin_auto_cover"

            self._emit_fill(
                {
                    "ticker": tk,
                    "price": px,
                    "shares": qty,
                    "action": action,
                    "commission": comm,
                    "trace_id": None,
                    "trace_ids": [],
                    "expert": "liquidation",
                    "analysis": analysis,
                }
            )
            marks[tk] = px

    # ----------------------------------------------------------------- #
    # 估值辅助
    # ----------------------------------------------------------------- #
    def _mark_price(self, ticker: str, *, fallback: float = 0.0) -> float:
        p = float(self.last_price.get(str(ticker).upper(), 0.0) or 0.0)
        if p > 0:
            return p
        if fallback and float(fallback) > 0:
            return float(fallback)
        pos = self.positions.get(str(ticker).upper())
        if pos is not None and float(pos.avg_price or 0.0) > 0:
            return float(pos.avg_price)
        return 0.0

    def _compute_equity_gross(
        self, *, cash: float, pos_shares: Dict[str, float], marks: Dict[str, float]
    ) -> tuple[float, float]:
        eq = float(cash)
        gross = 0.0
        for tk, sh in (pos_shares or {}).items():
            sh_f = float(sh or 0.0)
            if sh_f == 0.0:
                continue
            px = float((marks or {}).get(str(tk).upper(), 0.0) or 0.0)
            if px <= 0.0:
                continue
            eq += sh_f * px
            gross += abs(sh_f) * px
        return float(eq), float(gross)

    def equity(self) -> float:
        """当前总权益 = 现金 + Σ 持仓市值（多头加、空头减）。"""
        marks = {
            str(tk).upper(): self._mark_price(str(tk).upper(), fallback=float(p.avg_price or 0.0))
            for tk, p in (self.positions or {}).items()
            if p is not None
        }
        pos_shares = {str(tk).upper(): float(p.shares) for tk, p in (self.positions or {}).items() if p is not None}
        eq, _ = self._compute_equity_gross(cash=float(self.cash), pos_shares=pos_shares, marks=marks)
        return eq

    # ----------------------------------------------------------------- #
    # 盘前风控：模拟成交后的杠杆/保证金是否越界
    # ----------------------------------------------------------------- #
    def _pretrade_risk_check(
        self, *, ticker: str, action: str, price: float, shares: float, commission: float
    ) -> tuple[bool, str]:
        tk = str(ticker).upper().strip()
        px = float(price or 0.0)
        if not (tk and px > 0.0):
            return False, "invalid_price"
        sh0 = float(abs(shares) or 0.0)
        if sh0 <= 0.0:
            return False, "invalid_shares"

        cash_new = float(self.cash)
        pos_shares: Dict[str, float] = {}
        marks: Dict[str, float] = {}
        for k, p in (self.positions or {}).items():
            if p is None:
                continue
            pos_shares[str(k).upper()] = float(p.shares or 0.0)
            marks[str(k).upper()] = self._mark_price(str(k).upper(), fallback=float(p.avg_price or 0.0))
        marks[tk] = px

        cur = float(pos_shares.get(tk, 0.0) or 0.0)
        act = str(action or "").upper().strip()
        if act == "BUY":
            buy_sh = sh0
            if cur < 0.0:
                cover = min(abs(cur), buy_sh)
                cash_new -= px * cover + float(commission)
                cur += cover
                buy_sh -= cover
                if buy_sh > 0.0:
                    cash_new -= px * buy_sh
                    cur += buy_sh
            else:
                cash_new -= px * buy_sh + float(commission)
                cur += buy_sh
        elif act == "SELL":
            sell_sh = sh0
            if cur > 0.0:
                sell2 = min(cur, sell_sh)
                cash_new += px * sell2 - float(commission)
                cur -= sell2
            else:
                cash_new += px * sell_sh - float(commission)
                cur -= sell_sh
        else:
            return False, "unsupported_action"

        if abs(cur) < 1e-12:
            pos_shares.pop(tk, None)
        else:
            pos_shares[tk] = cur

        eq, gross = self._compute_equity_gross(cash=cash_new, pos_shares=pos_shares, marks=marks)
        if gross <= 0.0:
            return True, "ok"
        if not (eq > 0.0):
            return False, f"equity_nonpositive eq={eq:.2f}"

        ml = max(1.0, min(float(self.max_leverage or 3.0), 50.0))
        lev = gross / eq
        if lev > ml + 1e-9:
            return False, f"leverage_exceeded lev={lev:.2f} max={ml:.2f}"

        im = max(0.01, min(float(self.initial_margin or 0.35), 1.0))
        if eq < im * gross - 1e-6:
            return False, f"margin_exceeded eq={eq:.2f} req={im * gross:.2f}"
        return True, "ok"

    # ----------------------------------------------------------------- #
    # 下单（核心撮合）
    # ----------------------------------------------------------------- #
    def place_order(self, signal: dict) -> None:
        signal = signal if isinstance(signal, dict) else {}
        ticker = str(signal.get("ticker") or "").upper().strip()
        action = str(signal.get("action") or "").upper().strip()
        price = float(signal.get("price") or 0.0)
        shares = float(signal.get("shares") or 0.0)

        trace_id = str(signal.get("trace_id") or "").strip() or None
        trace_ids: List[str] = []
        raw_ids = signal.get("trace_ids")
        if isinstance(raw_ids, (list, tuple)):
            trace_ids = [str(x).strip() for x in raw_ids if str(x or "").strip()]
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)

        if not ticker or not action or price <= 0 or shares == 0:
            self._log(
                f"[Broker] ignore invalid order: action={action} ticker={ticker} "
                f"price={price} shares={shares}",
                priority=2,
            )
            return

        commission = float(signal.get("commission") or 0.0)
        notional = price * abs(shares)

        ok, why = self._pretrade_risk_check(
            ticker=ticker, action=action, price=price, shares=shares, commission=commission
        )
        if not ok:
            self._log(f"[Broker] reject {action} {ticker} x{shares:g} @ {price:.2f}: {why}", priority=2)
            return

        sell_shares: Optional[float] = None

        if action == "BUY":
            total_cost = notional + commission
            pos = self.positions.get(ticker)
            if pos is not None and float(pos.shares) < 0:
                # 先回补空头
                cover_shares = min(abs(float(pos.shares)), abs(shares))
                total_cost = price * cover_shares + commission
                entry_price = float(pos.avg_price)
                self.cash -= total_cost
                remaining = float(pos.shares) + cover_shares
                if remaining >= 0:
                    self.positions.pop(ticker, None)
                else:
                    pos.shares = remaining
                self._record_outcome(
                    trace_ids=getattr(pos, "trace_ids", []),
                    realized=(entry_price - price) * cover_shares - commission,
                    comment=f"realized_pnl entry={entry_price:.4f} cover={price:.4f} shares={cover_shares:.4f}",
                )
                extra = abs(shares) - cover_shares
                if extra > 0:  # 反手做多
                    self.cash -= price * extra
                    self.positions[ticker] = Position(ticker, extra, price, list(trace_ids))
            else:
                self.cash -= total_cost
                if pos is None:
                    self.positions[ticker] = Position(ticker, shares, price, list(trace_ids))
                else:
                    new_shares = pos.shares + shares
                    if new_shares == 0:
                        self.positions.pop(ticker, None)
                    else:
                        pos.avg_price = (pos.avg_price * pos.shares + price * shares) / new_shares
                        pos.shares = new_shares
                        for rid in trace_ids:
                            if rid and rid not in pos.trace_ids:
                                pos.trace_ids.append(rid)

        elif action == "SELL":
            pos = self.positions.get(ticker)
            if pos is None:
                sell_shares = abs(shares)
                self.cash += price * sell_shares - commission
                self.positions[ticker] = Position(ticker, -sell_shares, price, list(trace_ids))
            elif float(pos.shares) <= 0:
                # 加空
                sell_shares = abs(shares)
                self.cash += price * sell_shares - commission
                prev_abs = abs(float(pos.shares))
                new_abs = prev_abs + sell_shares
                if new_abs > 0:
                    pos.avg_price = (float(pos.avg_price) * prev_abs + price * sell_shares) / new_abs
                pos.shares = -new_abs
                for rid in trace_ids:
                    if rid and rid not in pos.trace_ids:
                        pos.trace_ids.append(rid)
            else:
                # 减多
                sell_shares = min(float(pos.shares), abs(shares))
                entry_price = float(pos.avg_price)
                self.cash += price * sell_shares - commission
                remaining = float(pos.shares) - sell_shares
                if remaining <= 0:
                    self.positions.pop(ticker, None)
                else:
                    pos.shares = remaining
                self._record_outcome(
                    trace_ids=getattr(pos, "trace_ids", []),
                    realized=(price - entry_price) * sell_shares - commission,
                    comment=f"realized_pnl entry={entry_price:.4f} exit={price:.4f} shares={sell_shares:.4f}",
                )
        else:
            self._log(f"[Broker] ignore unsupported action: {action}", priority=2)
            return

        self.orders.append(
            {"ticker": ticker, "action": action, "price": price, "shares": shares, "trace_id": trace_id}
        )
        self.last_price[ticker] = price

        fill_shares = float(sell_shares) if (action == "SELL" and sell_shares is not None) else shares
        self._emit_fill(
            {
                "ticker": ticker,
                "price": price,
                "shares": fill_shares,
                "action": action,
                "commission": commission,
                "trace_id": trace_id,
                "trace_ids": trace_ids,
                "expert": str(signal.get("expert") or "").strip(),
                "analysis": str(signal.get("analysis") or ""),
                "chart_score": signal.get("chart_score"),
                "news_score": signal.get("news_score"),
                "news_sentiment": signal.get("news_sentiment"),
                "news_summary": signal.get("news_summary"),
            }
        )
