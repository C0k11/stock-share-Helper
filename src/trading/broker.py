from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
import time
from collections import deque
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

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
        self.margin_interest_apr = 0.12
        self.short_borrow_fee_apr = 0.03
        self.settlement_interval_sec = 60.0
        self.liquidation_enabled = True
        self.liquidation_commission = 0.0
        self._last_settle_ts = time.time()
        self._last_liq_log_ts = 0.0
        self._last_fee_log_ts = 0.0

        self.us_rules_enabled = False
        self.us_tz = "America/New_York"
        self.us_allow_pre_market = True
        self.us_allow_regular = True
        self.us_allow_after_hours = True
        self.us_pre_market_start = "04:00"
        self.us_regular_start = "09:30"
        self.us_regular_end = "16:00"
        self.us_after_hours_end = "20:00"

        self.us_circuit_breaker_enabled = False
        self.us_circuit_breaker_proxy = "SPY"
        self.us_circuit_breaker_lvl1 = 0.07
        self.us_circuit_breaker_lvl2 = 0.13
        self.us_circuit_breaker_lvl3 = 0.20
        self.us_circuit_breaker_halt_sec = 900.0
        self._us_market_halt_until_ts: float = 0.0
        self._us_market_halt_level: int = 0
        self._us_market_ref_close: Dict[str, Dict[str, Any]] = {}

        self.us_luld_enabled = False
        self.us_luld_window_sec = 300.0
        self.us_luld_halt_sec = 300.0
        self.us_luld_band_tier1 = 0.05
        self.us_luld_band_tier2 = 0.10
        self.us_luld_tier1_tickers: list[str] = []
        self._us_symbol_halt_until: Dict[str, float] = {}
        self._us_price_hist: Dict[str, deque] = {}

        self.allow_short = True
        self.us_locate_required = False
        self.us_locate_max_shares = 0.0
        self._us_locate_remaining: Dict[str, float] = {}

        self.us_rule201_enabled = False
        self.us_rule201_tick_size = 0.01
        self.us_rule201_auto_price_improve = True
        self._us_rule201_active: Dict[str, Dict[str, Any]] = {}

        self.us_settlement_enabled = False
        self.us_settlement_days = 1
        self._us_settled_cash = float(self.cash)
        self._us_unsettled: deque = deque()
        self._us_last_settle_date: Optional[date] = None

        self.us_pdt_enabled = False
        self.us_pdt_min_equity = 25000.0
        self.us_pdt_max_day_trades = 4
        self.us_pdt_window_days = 5
        self._us_pdt_day_trades: deque = deque()
        self._us_pos_open_date: Dict[str, date] = {}

    def on_market_data(self, market_data: dict) -> None:
        try:
            md = market_data if isinstance(market_data, dict) else {}
        except Exception:
            md = {}

        ticker_raw = md.get("ticker")
        if not ticker_raw:
            ticker_raw = md.get("symbol")
        if not ticker_raw:
            ticker_raw = md.get("code")
        ticker = str(ticker_raw or "").upper().strip()

        price_raw = md.get("close")
        if price_raw is None:
            price_raw = md.get("price")
        if price_raw is None:
            price_raw = md.get("last")
        if price_raw is None:
            price_raw = md.get("c")
        try:
            price = float(price_raw or 0.0)
        except Exception:
            price = 0.0

        ts0 = None
        try:
            t = md.get("time")
            if isinstance(t, datetime):
                ts0 = t.timestamp()
            elif isinstance(t, (int, float)):
                ts0 = float(t)
        except Exception:
            ts0 = None

        if ticker and price > 0.0:
            try:
                self.last_price[str(ticker).upper()] = float(price)
            except Exception:
                pass

        try:
            if ts0 is not None:
                setattr(self, "_last_md_ts", float(ts0))
        except Exception:
            pass

        try:
            self._us_on_market_data(market_data=md, ts0=ts0)
        except Exception:
            pass

        self.mark_to_market(asof_ts=ts0)

    def _rules_log(self, msg: str, *, priority: int = 2) -> None:
        try:
            self.engine.push_event(Event(type=EventType.LOG, timestamp=datetime.now(), payload=str(msg), priority=int(priority)))
        except Exception:
            return

    def _us_tzinfo(self):
        try:
            if ZoneInfo is None:
                return None
            return ZoneInfo(str(getattr(self, "us_tz", "America/New_York") or "America/New_York"))
        except Exception:
            return None

    def _us_now_et(self, *, ts: Optional[float] = None) -> datetime:
        tz = self._us_tzinfo()
        try:
            if ts is not None:
                if tz is not None:
                    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(tz)
                return datetime.fromtimestamp(float(ts))
        except Exception:
            pass
        try:
            if tz is not None:
                return datetime.now(timezone.utc).astimezone(tz)
        except Exception:
            pass
        return datetime.now()

    def _us_parse_hhmm(self, s: str, fallback: dt_time) -> dt_time:
        try:
            t = str(s or "").strip()
            if not t:
                return fallback
            hh, mm = t.split(":", 1)
            return dt_time(hour=int(hh), minute=int(mm))
        except Exception:
            return fallback

    def _us_session(self, dt_et: datetime) -> str:
        try:
            pre0 = self._us_parse_hhmm(getattr(self, "us_pre_market_start", "04:00"), dt_time(4, 0))
            r0 = self._us_parse_hhmm(getattr(self, "us_regular_start", "09:30"), dt_time(9, 30))
            r1 = self._us_parse_hhmm(getattr(self, "us_regular_end", "16:00"), dt_time(16, 0))
            a1 = self._us_parse_hhmm(getattr(self, "us_after_hours_end", "20:00"), dt_time(20, 0))
        except Exception:
            pre0 = dt_time(4, 0)
            r0 = dt_time(9, 30)
            r1 = dt_time(16, 0)
            a1 = dt_time(20, 0)

        try:
            tt = dt_et.timetz() if hasattr(dt_et, "timetz") else dt_et.time()
        except Exception:
            tt = dt_et.time()

        try:
            if tt < pre0:
                return "closed"
            if tt < r0:
                return "pre"
            if tt < r1:
                return "regular"
            if tt < a1:
                return "after"
            return "closed"
        except Exception:
            return "regular"

    def _us_is_open_time(self, dt_et: datetime) -> bool:
        try:
            # Basic US market calendar approximation: Mon-Fri only (no holiday calendar).
            if int(dt_et.weekday()) >= 5:
                return False
        except Exception:
            pass
        sess = self._us_session(dt_et)
        if sess == "pre":
            return bool(getattr(self, "us_allow_pre_market", True))
        if sess == "regular":
            return bool(getattr(self, "us_allow_regular", True))
        if sess == "after":
            return bool(getattr(self, "us_allow_after_hours", True))
        return False

    def _us_settlement_process(self, dt_et: datetime) -> None:
        if not bool(getattr(self, "us_settlement_enabled", False)):
            self._us_settled_cash = float(self.cash)
            return

        d0 = dt_et.date()
        if self._us_last_settle_date is None:
            self._us_last_settle_date = d0
            try:
                self._us_settled_cash = float(self._us_settled_cash)
            except Exception:
                self._us_settled_cash = float(self.cash)
            return

        if d0 == self._us_last_settle_date:
            return

        try:
            self._us_last_settle_date = d0
        except Exception:
            pass

        try:
            moved = 0.0
            left = deque()
            while self._us_unsettled:
                item = self._us_unsettled.popleft()
                if not isinstance(item, dict):
                    continue
                sd = item.get("settle_date")
                amt = float(item.get("amount") or 0.0)
                if sd is None:
                    continue
                if isinstance(sd, date) and sd <= d0:
                    moved += amt
                else:
                    left.append(item)
            self._us_unsettled = left
            self._us_settled_cash += float(moved)
        except Exception:
            return

    def _us_on_market_data(self, *, market_data: dict, ts0: Optional[float]) -> None:
        if not bool(getattr(self, "us_rules_enabled", False)):
            return

        dt_et = self._us_now_et(ts=ts0)
        try:
            self._us_settlement_process(dt_et)
        except Exception:
            pass

        ticker_raw = market_data.get("ticker") or market_data.get("symbol") or market_data.get("code")
        ticker = str(ticker_raw or "").upper().strip()
        if not ticker:
            return

        price_raw = market_data.get("close")
        if price_raw is None:
            price_raw = market_data.get("price")
        if price_raw is None:
            price_raw = market_data.get("last")
        if price_raw is None:
            price_raw = market_data.get("c")
        try:
            price = float(price_raw or 0.0)
        except Exception:
            price = 0.0
        if price <= 0.0:
            return

        if bool(getattr(self, "us_circuit_breaker_enabled", False)):
            proxy = str(getattr(self, "us_circuit_breaker_proxy", "SPY") or "SPY").upper().strip()
            if ticker == proxy:
                self._us_update_circuit_breaker(dt_et=dt_et, price=price)

        if bool(getattr(self, "us_luld_enabled", False)):
            self._us_update_luld(dt_et=dt_et, ticker=ticker, price=price, ts0=ts0)

        if bool(getattr(self, "us_rule201_enabled", False)):
            self._us_update_rule201(dt_et=dt_et, ticker=ticker, price=price)

    def _us_update_circuit_breaker(self, *, dt_et: datetime, price: float) -> None:
        d0 = dt_et.date()
        proxy = str(getattr(self, "us_circuit_breaker_proxy", "SPY") or "SPY").upper().strip()
        state = self._us_market_ref_close.get(proxy)
        if not isinstance(state, dict):
            state = {}
            self._us_market_ref_close[proxy] = state

        prev_date = state.get("date")
        prev_close = state.get("close")
        if prev_date != d0 or not (isinstance(prev_close, (int, float)) and float(prev_close) > 0.0):
            state["date"] = d0
            state["close"] = float(price)
            self._us_market_halt_level = 0
            self._us_market_halt_until_ts = 0.0
            return

        ref = float(prev_close)
        drop = (float(price) - ref) / ref
        lvl1 = -abs(float(getattr(self, "us_circuit_breaker_lvl1", 0.07) or 0.07))
        lvl2 = -abs(float(getattr(self, "us_circuit_breaker_lvl2", 0.13) or 0.13))
        lvl3 = -abs(float(getattr(self, "us_circuit_breaker_lvl3", 0.20) or 0.20))

        lvl = 0
        if drop <= lvl1:
            lvl = 1
        if drop <= lvl2:
            lvl = 2
        if drop <= lvl3:
            lvl = 3

        if lvl <= int(getattr(self, "_us_market_halt_level", 0) or 0):
            return

        try:
            t_r_end = self._us_parse_hhmm(getattr(self, "us_regular_end", "16:00"), dt_time(16, 0))
            tt = dt_et.timetz() if hasattr(dt_et, "timetz") else dt_et.time()
            cutoff = (datetime.combine(dt_et.date(), t_r_end, tzinfo=dt_et.tzinfo) - timedelta(minutes=35)).timetz()
            before_cutoff = bool(tt < cutoff)
        except Exception:
            before_cutoff = True

        now_ts = dt_et.timestamp() if dt_et.tzinfo is not None else time.time()
        if lvl >= 3:
            try:
                end_dt = datetime.combine(dt_et.date(), dt_time(23, 59, 59), tzinfo=dt_et.tzinfo)
                self._us_market_halt_until_ts = float(end_dt.timestamp())
            except Exception:
                self._us_market_halt_until_ts = float(now_ts + 24 * 3600)
        else:
            if before_cutoff:
                halt_sec = float(getattr(self, "us_circuit_breaker_halt_sec", 900.0) or 900.0)
                self._us_market_halt_until_ts = float(now_ts + max(1.0, halt_sec))

        self._us_market_halt_level = int(lvl)
        self._rules_log(f"[RULES] MWCB level={lvl} proxy={getattr(self, 'us_circuit_breaker_proxy', 'SPY')} drop={drop:.2%} halt_until={self._us_market_halt_until_ts:.0f}", priority=2)

    def _us_update_luld(self, *, dt_et: datetime, ticker: str, price: float, ts0: Optional[float]) -> None:
        if self._us_session(dt_et) != "regular":
            return

        try:
            now_ts = float(ts0) if ts0 is not None else float(dt_et.timestamp())
        except Exception:
            now_ts = float(time.time())

        w = float(getattr(self, "us_luld_window_sec", 300.0) or 300.0)
        w = max(30.0, min(w, 3600.0))
        hist = self._us_price_hist.get(ticker)
        if hist is None or not isinstance(hist, deque):
            hist = deque()
            self._us_price_hist[ticker] = hist

        try:
            hist.append((now_ts, float(price)))
            while hist and (now_ts - float(hist[0][0])) > w:
                hist.popleft()
        except Exception:
            return

        if len(hist) < 3:
            return

        try:
            ref = sum(float(p) for _t, p in list(hist)) / float(len(hist))
        except Exception:
            return

        if not (ref > 0.0):
            return

        tku = str(ticker).upper()
        is_t1 = False
        try:
            is_t1 = tku in {str(x or "").upper().strip() for x in (getattr(self, "us_luld_tier1_tickers", []) or [])}
        except Exception:
            is_t1 = False

        band = float(getattr(self, "us_luld_band_tier1", 0.05) or 0.05) if is_t1 else float(getattr(self, "us_luld_band_tier2", 0.10) or 0.10)
        band = max(0.01, min(band, 0.50))

        try:
            t_r_end = self._us_parse_hhmm(getattr(self, "us_regular_end", "16:00"), dt_time(16, 0))
            cutoff_dt = datetime.combine(dt_et.date(), t_r_end, tzinfo=dt_et.tzinfo) - timedelta(minutes=25)
            if dt_et >= cutoff_dt:
                band = float(band) * 2.0
        except Exception:
            pass

        dev = abs(float(price) - float(ref)) / float(ref)
        if dev <= band:
            return

        halt_sec = float(getattr(self, "us_luld_halt_sec", 300.0) or 300.0)
        halt_sec = max(10.0, min(halt_sec, 3600.0))
        until = float(now_ts + halt_sec)
        prev = float(self._us_symbol_halt_until.get(ticker, 0.0) or 0.0)
        if until <= prev:
            return
        self._us_symbol_halt_until[ticker] = until
        self._rules_log(f"[RULES] LULD halt ticker={ticker} dev={dev:.2%} band={band:.2%} ref={ref:.4f} px={price:.4f} until={until:.0f}", priority=2)

    def _us_update_rule201(self, *, dt_et: datetime, ticker: str, price: float) -> None:
        d0 = dt_et.date()
        st = self._us_rule201_active.get(ticker)
        if not isinstance(st, dict):
            st = {"date": d0, "ref": float(price), "active": False}
            self._us_rule201_active[ticker] = st
            return

        if st.get("date") != d0:
            st["date"] = d0
            st["ref"] = float(price)
            st["active"] = False
            return

        try:
            ref = float(st.get("ref") or 0.0)
        except Exception:
            ref = 0.0
        if not (ref > 0.0):
            st["ref"] = float(price)
            st["active"] = False
            return

        try:
            drop = (float(price) - ref) / ref
        except Exception:
            drop = 0.0
        if drop <= -0.10:
            if not bool(st.get("active", False)):
                st["active"] = True
                self._rules_log(f"[RULES] Rule201 active ticker={ticker} drop={drop:.2%} ref={ref:.4f} px={price:.4f}", priority=2)

    def _us_is_market_halted(self, dt_et: datetime) -> tuple[bool, str]:
        try:
            until = float(getattr(self, "_us_market_halt_until_ts", 0.0) or 0.0)
        except Exception:
            until = 0.0
        if until <= 0.0:
            return False, ""
        try:
            now_ts = dt_et.timestamp() if dt_et.tzinfo is not None else time.time()
        except Exception:
            now_ts = time.time()
        if float(now_ts) < float(until):
            lvl = int(getattr(self, "_us_market_halt_level", 0) or 0)
            return True, f"MWCB(level={lvl})"
        return False, ""

    def _us_is_symbol_halted(self, ticker: str, dt_et: datetime) -> tuple[bool, str]:
        try:
            until = float(self._us_symbol_halt_until.get(str(ticker).upper(), 0.0) or 0.0)
        except Exception:
            until = 0.0
        if until <= 0.0:
            return False, ""
        try:
            now_ts = dt_et.timestamp() if dt_et.tzinfo is not None else time.time()
        except Exception:
            now_ts = time.time()
        if float(now_ts) < float(until):
            return True, "LULD"
        return False, ""

    def _us_equity(self) -> float:
        eq = float(getattr(self, "cash", 0.0) or 0.0)
        try:
            for tk, pos in (self.positions or {}).items():
                if pos is None:
                    continue
                sh = float(getattr(pos, "shares", 0.0) or 0.0)
                if sh == 0.0:
                    continue
                px = self._mark_price(str(tk).upper(), fallback=float(getattr(pos, "avg_price", 0.0) or 0.0))
                if px <= 0.0:
                    continue
                eq += sh * float(px)
        except Exception:
            pass
        return float(eq)

    def _us_pdt_restricted(self, dt_et: datetime) -> bool:
        if not bool(getattr(self, "us_pdt_enabled", False)):
            return False
        eq = float(self._us_equity())
        if eq >= float(getattr(self, "us_pdt_min_equity", 25000.0) or 25000.0):
            return False

        try:
            window = int(getattr(self, "us_pdt_window_days", 5) or 5)
        except Exception:
            window = 5
        window = max(1, min(window, 30))

        cutoff = dt_et.date() - timedelta(days=window * 2)
        try:
            while self._us_pdt_day_trades and isinstance(self._us_pdt_day_trades[0], date) and self._us_pdt_day_trades[0] < cutoff:
                self._us_pdt_day_trades.popleft()
        except Exception:
            pass

        try:
            max_dt = int(getattr(self, "us_pdt_max_day_trades", 4) or 4)
        except Exception:
            max_dt = 4
        max_dt = max(1, min(max_dt, 20))
        return len(self._us_pdt_day_trades) >= max_dt

    def mark_to_market(self, *, asof_ts: Optional[float] = None) -> None:
        try:
            now = float(asof_ts) if asof_ts is not None else float(time.time())
        except Exception:
            now = float(time.time())

        try:
            last = float(getattr(self, "_last_settle_ts", now) or now)
        except Exception:
            last = now
        if now <= last:
            return

        try:
            interval = float(getattr(self, "settlement_interval_sec", 60.0) or 60.0)
        except Exception:
            interval = 60.0
        interval = max(1.0, min(interval, 3600.0))

        dt = float(now - last)
        if dt < 0.5:
            return

        n_steps = int(dt // interval)
        if n_steps <= 0:
            return

        step_dt = float(dt) / float(n_steps)
        for _ in range(n_steps):
            self._accrue_fees(step_dt)
            self._maintenance_check_and_liquidate(now)

        try:
            self._last_settle_ts = float(now)
        except Exception:
            pass

    def _accrue_fees(self, dt_sec: float) -> None:
        try:
            dt = float(dt_sec or 0.0)
        except Exception:
            dt = 0.0
        if dt <= 0.0:
            return

        try:
            mi = float(getattr(self, "margin_interest_apr", 0.12) or 0.0)
        except Exception:
            mi = 0.0
        try:
            sb = float(getattr(self, "short_borrow_fee_apr", 0.03) or 0.0)
        except Exception:
            sb = 0.0

        mi = max(0.0, min(mi, 5.0))
        sb = max(0.0, min(sb, 5.0))
        year_sec = 365.0 * 24.0 * 3600.0

        interest = 0.0
        if float(self.cash) < 0.0 and mi > 0.0:
            interest = (-float(self.cash)) * (mi / year_sec) * float(dt)

        short_value = 0.0
        if sb > 0.0:
            try:
                for tk, pos in (self.positions or {}).items():
                    if pos is None:
                        continue
                    sh = float(getattr(pos, "shares", 0.0) or 0.0)
                    if sh >= 0.0:
                        continue
                    px = self._mark_price(str(tk).upper(), fallback=float(getattr(pos, "avg_price", 0.0) or 0.0))
                    if px <= 0.0:
                        continue
                    short_value += abs(sh) * float(px)
            except Exception:
                short_value = 0.0

        borrow_fee = 0.0
        if short_value > 0.0 and sb > 0.0:
            borrow_fee = float(short_value) * (sb / year_sec) * float(dt)

        total_fee = float(interest + borrow_fee)
        if total_fee > 0.0:
            self.cash -= float(total_fee)
            try:
                now = float(time.time())
                last = float(getattr(self, "_last_fee_log_ts", 0.0) or 0.0)
                if (now - last) >= 30.0:
                    self._last_fee_log_ts = now
                    self.engine.push_event(
                        Event(
                            type=EventType.LOG,
                            timestamp=datetime.now(),
                            payload=f"[Broker] fees accrued: interest={interest:.4f} borrow_fee={borrow_fee:.4f} cash={float(self.cash):.2f}",
                            priority=1,
                        )
                    )
            except Exception:
                pass

    def _maintenance_check_and_liquidate(self, asof_ts: float) -> None:
        try:
            if not bool(getattr(self, "liquidation_enabled", True)):
                return
        except Exception:
            return

        marks: Dict[str, float] = {}
        pos_shares: Dict[str, float] = {}
        try:
            for tk, p in (self.positions or {}).items():
                if p is None:
                    continue
                sh = float(getattr(p, "shares", 0.0) or 0.0)
                if sh == 0.0:
                    continue
                pos_shares[str(tk).upper()] = float(sh)
                marks[str(tk).upper()] = self._mark_price(str(tk).upper(), fallback=float(getattr(p, "avg_price", 0.0) or 0.0))
        except Exception:
            return

        eq, gross = self._compute_equity_gross(cash=float(self.cash), pos_shares=pos_shares, marks=marks)
        if gross <= 0.0:
            return

        try:
            mm = float(getattr(self, "maintenance_margin", 0.25) or 0.25)
        except Exception:
            mm = 0.25
        mm = max(0.01, min(mm, 1.0))
        req = float(mm) * float(gross)
        if eq >= req:
            return

        self._auto_liquidate(marks=marks, eq=eq, gross=gross, req=req)

    def _auto_liquidate(self, *, marks: Dict[str, float], eq: float, gross: float, req: float) -> None:
        try:
            now = float(time.time())
            last = float(getattr(self, "_last_liq_log_ts", 0.0) or 0.0)
            if (now - last) >= 5.0:
                self._last_liq_log_ts = now
                self.engine.push_event(
                    Event(
                        type=EventType.LOG,
                        timestamp=datetime.now(),
                        payload=f"[Broker] [Liquidation] start eq={eq:.2f} gross={gross:.2f} req={req:.2f} cash={float(self.cash):.2f}",
                        priority=2,
                    )
                )
        except Exception:
            pass

        for _ in range(30):
            pos = None
            px = 0.0
            exp = 0.0
            try:
                for tk, p in (self.positions or {}).items():
                    if p is None:
                        continue
                    sh = float(getattr(p, "shares", 0.0) or 0.0)
                    if sh == 0.0:
                        continue
                    mk = float((marks or {}).get(str(tk).upper(), 0.0) or 0.0)
                    if mk <= 0.0:
                        continue
                    e = abs(sh) * mk
                    if e > exp:
                        exp = e
                        pos = p
                        px = mk
            except Exception:
                pos = None

            if pos is None or not (px > 0.0):
                return

            tk = str(getattr(pos, "ticker", "") or "").upper().strip()
            sh0 = float(getattr(pos, "shares", 0.0) or 0.0)

            try:
                mm = float(getattr(self, "maintenance_margin", 0.25) or 0.25)
            except Exception:
                mm = 0.25
            mm = max(0.01, min(mm, 1.0))

            marks2 = dict(marks or {})
            pos_shares2: Dict[str, float] = {}
            try:
                for k, p2 in (self.positions or {}).items():
                    if p2 is None:
                        continue
                    sh2 = float(getattr(p2, "shares", 0.0) or 0.0)
                    if sh2 == 0.0:
                        continue
                    pos_shares2[str(k).upper()] = sh2
            except Exception:
                pos_shares2 = {}

            eq2, gross2 = self._compute_equity_gross(cash=float(self.cash), pos_shares=pos_shares2, marks=marks2)
            req2 = float(mm) * float(gross2)
            if gross2 <= 0.0 or eq2 >= req2:
                return

            deficit = float(req2 - eq2)
            qty = max(1.0, abs(sh0) * 0.25)
            try:
                qty = max(qty, deficit / float(px))
            except Exception:
                qty = qty
            qty = min(abs(sh0), qty)

            comm = 0.0
            try:
                comm = float(getattr(self, "liquidation_commission", 0.0) or 0.0)
            except Exception:
                comm = 0.0
            comm = max(0.0, min(comm, 100000.0))

            if sh0 > 0.0:
                proceeds = float(px) * float(qty) - float(comm)
                self.cash += float(proceeds)
                new_sh = float(sh0 - qty)
                if new_sh <= 0.0:
                    self.positions.pop(tk, None)
                else:
                    pos.shares = float(new_sh)
                fill = {
                    "ticker": tk,
                    "price": float(px),
                    "shares": float(qty),
                    "action": "SELL",
                    "commission": float(comm),
                    "trace_id": None,
                    "trace_ids": [],
                    "expert": "liquidation",
                    "analysis": "maintenance_margin_auto_sell",
                }
                try:
                    self.engine.push_event(Event(type=EventType.FILL, timestamp=datetime.now(), payload=fill))
                except Exception:
                    pass
            else:
                cost = float(px) * float(qty) + float(comm)
                self.cash -= float(cost)
                new_sh = float(sh0 + qty)
                if new_sh >= 0.0:
                    self.positions.pop(tk, None)
                else:
                    pos.shares = float(new_sh)
                fill = {
                    "ticker": tk,
                    "price": float(px),
                    "shares": float(qty),
                    "action": "BUY",
                    "commission": float(comm),
                    "trace_id": None,
                    "trace_ids": [],
                    "expert": "liquidation",
                    "analysis": "maintenance_margin_auto_cover",
                }
                try:
                    self.engine.push_event(Event(type=EventType.FILL, timestamp=datetime.now(), payload=fill))
                except Exception:
                    pass

            try:
                marks[tk] = float(px)
            except Exception:
                pass

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

        def _rej_rules(reason: str) -> None:
            _rej(f"[RULES] reject {action} {ticker} x{shares:g} @ {price:.4f}: {reason}")

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

        dt_et = None
        try:
            ts_in = signal.get("timestamp")
            if isinstance(ts_in, str) and ts_in.strip():
                dt_et = datetime.fromisoformat(ts_in.strip().replace("Z", "+00:00"))
                if dt_et.tzinfo is None:
                    dt_et = dt_et.replace(tzinfo=timezone.utc)
                tz0 = self._us_tzinfo()
                if tz0 is not None:
                    dt_et = dt_et.astimezone(tz0)
        except Exception:
            dt_et = None

        if dt_et is None:
            try:
                ts0 = float(getattr(self, "_last_md_ts", None) or 0.0)
                dt_et = self._us_now_et(ts=ts0 if ts0 > 0 else None)
            except Exception:
                dt_et = self._us_now_et()

        if bool(getattr(self, "us_rules_enabled", False)):
            try:
                self._us_settlement_process(dt_et)
            except Exception:
                pass

            if not self._us_is_open_time(dt_et):
                _rej_rules(f"market_closed session={self._us_session(dt_et)}")
                return

            if bool(getattr(self, "us_circuit_breaker_enabled", False)):
                halted, why_h = self._us_is_market_halted(dt_et)
                if halted:
                    _rej_rules(f"market_halt {why_h}")
                    return

            if bool(getattr(self, "us_luld_enabled", False)):
                halted2, why2 = self._us_is_symbol_halted(ticker, dt_et)
                if halted2:
                    _rej_rules(f"symbol_halt {why2}")
                    return

            if self._us_pdt_restricted(dt_et):
                try:
                    cur_sh = float(getattr(self.positions.get(ticker), "shares", 0.0) or 0.0) if self.positions.get(ticker) is not None else 0.0
                except Exception:
                    cur_sh = 0.0
                opening = (abs(cur_sh) < 1e-12)
                if opening:
                    _rej_rules("PDT_restricted")
                    return

        commission = float(signal.get("commission") or 0.0)
        notional = price * abs(shares)

        if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
            if action == "BUY":
                try:
                    need = float(notional + commission)
                except Exception:
                    need = 0.0
                try:
                    settled = float(getattr(self, "_us_settled_cash", float(self.cash)) or float(self.cash))
                except Exception:
                    settled = float(self.cash)
                if need > settled + 1e-9:
                    _rej_rules(f"unsettled_cash need={need:.2f} settled={settled:.2f}")
                    return

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

        if bool(getattr(self, "us_rules_enabled", False)):
            try:
                cur_sh = float(getattr(self.positions.get(ticker), "shares", 0.0) or 0.0) if self.positions.get(ticker) is not None else 0.0
            except Exception:
                cur_sh = 0.0

            is_short_inc = False
            try:
                if action == "SELL":
                    if cur_sh <= 0.0:
                        is_short_inc = True
            except Exception:
                is_short_inc = False

            if is_short_inc:
                if not bool(getattr(self, "allow_short", True)):
                    _rej_rules("short_disabled")
                    return

                if bool(getattr(self, "us_locate_required", False)):
                    try:
                        lim = float(getattr(self, "us_locate_max_shares", 0.0) or 0.0)
                    except Exception:
                        lim = 0.0
                    if lim > 0.0:
                        remain = float(self._us_locate_remaining.get(ticker, lim) or lim)
                        inc = float(abs(shares) or 0.0)
                        if inc > remain + 1e-9:
                            _rej_rules(f"locate_failed remain={remain:g} need={inc:g}")
                            return
                        self._us_locate_remaining[ticker] = float(remain - inc)

                if bool(getattr(self, "us_rule201_enabled", False)):
                    st = self._us_rule201_active.get(ticker)
                    active201 = bool(isinstance(st, dict) and st.get("active", False))
                    if active201:
                        try:
                            last_px = float(self.last_price.get(ticker, 0.0) or 0.0)
                        except Exception:
                            last_px = 0.0
                        tick = float(getattr(self, "us_rule201_tick_size", 0.01) or 0.01)
                        tick = max(0.0001, min(tick, 10.0))
                        if not bool(getattr(self, "us_rule201_auto_price_improve", True)):
                            _rej_rules("Rule201_active")
                            return
                        if last_px > 0.0 and float(price) <= float(last_px):
                            price = float(last_px + tick)
                            notional = float(price) * abs(shares)
                            try:
                                signal["price"] = float(price)
                            except Exception:
                                pass
                            self._rules_log(f"[RULES] Rule201 price_improve {ticker} -> {price:.4f}", priority=1)

        if action == "BUY":
            total_cost = notional + commission
            pos = self.positions.get(ticker)
            if pos is not None and float(pos.shares) < 0:
                # Cover short first
                cover_shares = min(abs(float(pos.shares)), abs(shares))
                total_cost = (price * cover_shares) + commission
                entry_price = float(pos.avg_price)
                self.cash -= total_cost
                if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
                    try:
                        self._us_settled_cash = float(getattr(self, "_us_settled_cash", 0.0) or 0.0) - float(total_cost)
                    except Exception:
                        pass
                remaining = float(pos.shares) + float(cover_shares)
                if remaining >= 0:
                    self.positions.pop(ticker, None)
                    try:
                        if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_pdt_enabled", False)):
                            d0 = dt_et.date()
                            opened = self._us_pos_open_date.get(ticker)
                            if opened is not None and opened == d0:
                                self._us_pdt_day_trades.append(d0)
                                self._rules_log(
                                    f"[RULES] PDT day_trade ticker={ticker} date={d0.isoformat()} count={len(self._us_pdt_day_trades)}",
                                    priority=1,
                                )
                            self._us_pos_open_date.pop(ticker, None)
                    except Exception:
                        pass
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
                    if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
                        try:
                            self._us_settled_cash = float(getattr(self, "_us_settled_cash", 0.0) or 0.0) - float(total_cost2)
                        except Exception:
                            pass
                    self.positions[ticker] = Position(ticker=ticker, shares=extra, avg_price=price, trace_ids=list(trace_ids))
                    try:
                        if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_pdt_enabled", False)):
                            self._us_pos_open_date[ticker] = dt_et.date()
                    except Exception:
                        pass
            else:
                self.cash -= total_cost
                if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
                    try:
                        self._us_settled_cash = float(getattr(self, "_us_settled_cash", 0.0) or 0.0) - float(total_cost)
                    except Exception:
                        pass
                if pos is None:
                    self.positions[ticker] = Position(ticker=ticker, shares=shares, avg_price=price, trace_ids=list(trace_ids))
                    try:
                        if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_pdt_enabled", False)):
                            self._us_pos_open_date[ticker] = dt_et.date()
                    except Exception:
                        pass
                else:
                    new_shares = pos.shares + shares
                    if new_shares == 0:
                        self.positions.pop(ticker, None)
                        try:
                            if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_pdt_enabled", False)):
                                self._us_pos_open_date.pop(ticker, None)
                        except Exception:
                            pass
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
                if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
                    try:
                        d0 = dt_et.date()
                        sd = d0 + timedelta(days=int(getattr(self, "us_settlement_days", 1) or 1))
                        self._us_unsettled.append({"settle_date": sd, "amount": float(proceeds)})
                    except Exception:
                        pass
                self.positions[ticker] = Position(ticker=ticker, shares=-float(sell_shares), avg_price=price, trace_ids=list(trace_ids))
                try:
                    if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_pdt_enabled", False)):
                        self._us_pos_open_date[ticker] = dt_et.date()
                except Exception:
                    pass
            elif float(pos.shares) <= 0:
                sell_shares = abs(shares)
                proceeds = price * sell_shares - commission
                self.cash += proceeds
                if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
                    try:
                        d0 = dt_et.date()
                        sd = d0 + timedelta(days=int(getattr(self, "us_settlement_days", 1) or 1))
                        self._us_unsettled.append({"settle_date": sd, "amount": float(proceeds)})
                    except Exception:
                        pass
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
                if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_settlement_enabled", False)):
                    try:
                        d0 = dt_et.date()
                        sd = d0 + timedelta(days=int(getattr(self, "us_settlement_days", 1) or 1))
                        self._us_unsettled.append({"settle_date": sd, "amount": float(proceeds)})
                    except Exception:
                        pass
                remaining = float(pos.shares) - float(sell_shares)
                if remaining <= 0:
                    self.positions.pop(ticker, None)
                else:
                    pos.shares = float(remaining)

                try:
                    if bool(getattr(self, "us_rules_enabled", False)) and bool(getattr(self, "us_pdt_enabled", False)):
                        d0 = dt_et.date()
                        opened = self._us_pos_open_date.get(ticker)
                        if opened is not None and opened == d0:
                            self._us_pdt_day_trades.append(d0)
                            self._rules_log(f"[RULES] PDT day_trade ticker={ticker} date={d0.isoformat()} count={len(self._us_pdt_day_trades)}", priority=1)
                        self._us_pos_open_date.pop(ticker, None)
                except Exception:
                    pass

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

        fill_shares = shares
        try:
            if action == "SELL":
                # If we sold against an existing long, the executed shares might be capped.
                # Use the computed sell_shares to avoid overstating fills in UI/RL tracking.
                if "sell_shares" in locals():
                    fill_shares = float(sell_shares)
        except Exception:
            fill_shares = shares

        fill = {
            "ticker": ticker,
            "price": price,
            "shares": fill_shares,
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
