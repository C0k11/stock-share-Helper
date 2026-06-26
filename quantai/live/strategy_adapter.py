"""把 `agents.AgentOrchestrator`（纯大脑）接到实盘事件循环的策略适配器。

引擎对策略的契约是 `on_bar(market_data) -> Optional[signal]`（见 `live/engine.py`）。本适配器：

1. 维护每个 ticker 的滚动 OHLCV 缓冲；
2. 用 `quantai.features` 算技术因子，拼成 `AgentContext.features`（含波动率%、可选 macro）；
3. 从注入的 `broker` 读当前持仓/账户，构造 `AgentContext`；
4. 调 `orchestrator.decide(ctx, llm=)` 得 `FinalDecision`；
5. HOLD / 未批准 -> None；否则按"置信度×权益比例"定股数，产出 broker `place_order` 能吃的信号 dict。

**边界**：适配器只做"特征 + 上下文 + 定量"的胶水；决策逻辑在 `agents/`，撮合在 `execution/`。
仓位定量（`_calc_position_size`）忠实迁移自旧 `strategy.py::_calculate_position_size`。
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional

import pandas as pd

from quantai.agents.base import Account, AgentContext, Position
from quantai.features import compute_technical_features


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class AgentStrategy:
    """Orchestrator -> 引擎策略接口的适配器。"""

    def __init__(
        self,
        orchestrator: Any,
        broker: Any = None,
        *,
        llm: Any = None,
        max_history: int = 300,
        min_history: int = 30,
        position_risk_pct: float = 0.05,
        max_shares: int = 2000,
        default_equity: float = 50000.0,
        allow_short: bool = False,
        macro: Optional[Dict[str, float]] = None,
        on_log: Optional[Any] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.broker = broker
        self.llm = llm
        self.max_history = max(2, int(max_history))
        self.min_history = max(2, int(min_history))
        self.position_risk_pct = max(0.005, min(float(position_risk_pct), 0.5))
        self.max_shares = max(1, min(int(max_shares), 200000))
        self.default_equity = float(default_equity)
        self.allow_short = bool(allow_short)
        self.macro: Dict[str, float] = dict(macro or {})
        self._on_log = on_log if callable(on_log) else (lambda _m, _p=2: None)
        self._history: Dict[str, Deque[dict]] = {}

    # ----------------------------------------------------------------- #
    # 外部喂宏观快照（VIX/TNX）；集成层按日更新
    # ----------------------------------------------------------------- #
    def update_macro(self, *, vix: Optional[float] = None, tnx: Optional[float] = None) -> None:
        if vix is not None:
            self.macro["vix"] = float(vix)
        if tnx is not None:
            self.macro["tnx"] = float(tnx)

    # ----------------------------------------------------------------- #
    # 引擎回调
    # ----------------------------------------------------------------- #
    def on_bar(self, market_data: dict) -> Optional[dict]:
        md = market_data if isinstance(market_data, dict) else {}
        ticker = str(md.get("ticker") or md.get("symbol") or md.get("code") or "").upper().strip()
        price = _f(md.get("close") if md.get("close") is not None else
                   md.get("price") if md.get("price") is not None else
                   md.get("last") if md.get("last") is not None else md.get("c"))
        if not ticker or price <= 0.0:
            return None

        self._update_history(ticker, md, price)
        if len(self._history[ticker]) < self.min_history:
            return None  # 预热中：历史不足以算特征

        try:
            features = self._build_features(ticker)
        except Exception as exc:  # 特征算不出来就跳过这根
            self._on_log(f"[Strategy] feature error {ticker}: {exc}", 2)
            return None

        ctx = AgentContext(
            ticker=ticker,
            features=features,
            position=self._position(ticker),
            account=self._account(),
            allow_short=self.allow_short,
            asof=self._asof(md),
        )

        decision = self.orchestrator.decide(ctx, llm=self.llm)
        if not getattr(decision, "approved", False):
            return None
        action = str(getattr(decision, "action", "HOLD") or "HOLD").upper()
        if action not in {"BUY", "SELL"}:
            return None

        # 安全闸：不允许做空时，SELL 只能减多头，不能开/加空
        if action == "SELL" and not self.allow_short and ctx.position.shares <= 0.0:
            return None

        shares = self._calc_position_size(price, confidence=self._confidence(decision))
        if shares <= 0:
            return None
        if action == "SELL":
            # 减多头时不超卖现有多仓
            held = ctx.position.shares
            if not self.allow_short and held > 0.0:
                shares = int(min(shares, held))
            if shares <= 0:
                return None

        return {
            "ticker": ticker,
            "action": action,
            "price": price,
            "shares": shares,
            "expert": getattr(decision, "expert", ""),
            "analysis": getattr(decision, "reason", ""),
            "chart_score": getattr(decision, "chart_score", 0),
            "regime": getattr(decision, "regime", ""),
            "macro_label": getattr(decision, "macro_label", "NEUTRAL"),
        }

    # ----------------------------------------------------------------- #
    # 内部辅助
    # ----------------------------------------------------------------- #
    def _update_history(self, ticker: str, md: dict, price: float) -> None:
        buf = self._history.get(ticker)
        if buf is None:
            buf = deque(maxlen=self.max_history)
            self._history[ticker] = buf
        o = _f(md.get("open"), price)
        h = _f(md.get("high"), price)
        l = _f(md.get("low"), price)
        v = _f(md.get("volume"), 0.0)
        buf.append({"open": o, "high": h, "low": l, "close": price, "volume": v})

    def _build_features(self, ticker: str) -> Dict[str, Any]:
        df = pd.DataFrame(list(self._history[ticker]))
        tech_df = compute_technical_features(df)
        latest = tech_df.iloc[-1].to_dict()
        technical = {k: float(v) for k, v in latest.items() if pd.notna(v)}
        vol_ann_pct = float(technical.get("volatility_20d", 0.0)) * 100.0
        features: Dict[str, Any] = {
            "technical": technical,
            "signal": {},
            "volatility_ann_pct": vol_ann_pct,
        }
        if self.macro:
            features["macro"] = dict(self.macro)
        return features

    def _position(self, ticker: str) -> Position:
        br = self.broker
        if br is None:
            return Position()
        pos = (getattr(br, "positions", {}) or {}).get(ticker.upper())
        if pos is None:
            return Position()
        return Position(shares=_f(getattr(pos, "shares", 0.0)), avg_price=_f(getattr(pos, "avg_price", 0.0)))

    def _account(self) -> Account:
        br = self.broker
        if br is None:
            return Account(cash=self.default_equity, equity=self.default_equity)
        cash = _f(getattr(br, "cash", 0.0))
        equity = _f(br.equity()) if hasattr(br, "equity") else cash
        return Account(cash=cash, equity=equity)

    def _equity(self) -> float:
        br = self.broker
        if br is not None and hasattr(br, "equity"):
            eq = _f(br.equity())
            if eq > 0.0:
                return eq
        return self.default_equity

    def _calc_position_size(self, price: float, *, confidence: float = 0.75) -> int:
        """忠实迁移 strategy.py::_calculate_position_size：权益比例 × 置信度缩放。"""
        if price <= 0.0:
            return 0
        eq = self._equity()
        if eq <= 0.0:
            eq = self.default_equity
        c = max(0.1, min(float(confidence), 0.99))
        alloc = eq * self.position_risk_pct * (0.5 + c)
        shares = int(max(1.0, alloc / price))
        return int(max(1, min(shares, self.max_shares)))

    @staticmethod
    def _confidence(decision: Any) -> float:
        trace = getattr(decision, "trace", None)
        if isinstance(trace, dict):
            for key in ("confidence", "conf"):
                if key in trace:
                    return _f(trace[key], 0.75)
        return 0.75

    @staticmethod
    def _asof(md: dict) -> str:
        t = md.get("time")
        if isinstance(t, datetime):
            return t.isoformat()
        if isinstance(t, str):
            return t
        return ""
