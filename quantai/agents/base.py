"""Agent 层的纯数据契约（无 torch / 无重依赖）。

把旧 `src/trading/strategy.py` 里散落的 dict 出入参收敛成有类型的 dataclass，
让每个 agent 的输入/输出可被静态检查、可被单测构造。所有字段都给默认值，
便于测试里只填关心的字段。

设计要点：
- `AgentContext`：一次决策所需的全部只读上下文（特征 + 持仓 + 账户）。
- `ExpertDecision`：专家给出的方向 + 理由 + 元数据。
- `Regime` / `Action`：用常量类避免到处写裸字符串拼写出错。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class Action:
    """交易方向常量（取代散落的裸字符串）。"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLEAR = "CLEAR"

    ALL = frozenset({BUY, SELL, HOLD, CLEAR})

    @staticmethod
    def normalize(value: Any, *, default: str = "HOLD") -> str:
        """把任意输入规整成合法 action；非法 -> default。"""
        s = str(value or "").strip().upper()
        return s if s in Action.ALL else default


class Regime:
    """Planner 输出的市场状态常量。"""

    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    CASH_PRESERVATION = "cash_preservation"


@dataclass
class Position:
    """某 ticker 的当前持仓快照。"""

    shares: float = 0.0
    avg_price: float = 0.0

    @property
    def side(self) -> str:
        if self.shares > 0:
            return "LONG"
        if self.shares < 0:
            return "SHORT"
        return "FLAT"


@dataclass
class Account:
    """账户层面快照（喂给专家 prompt 做仓位感知）。"""

    cash: float = 0.0
    equity: float = 0.0
    gross_exposure: float = 0.0
    leverage: float = 0.0


@dataclass
class AgentContext:
    """一次 on_bar 决策的全部只读上下文。

    `features` 沿用旧结构：{"technical": {...}, "signal": {...}, "volatility_ann_pct": x}，
    这样可与既有 `_compute_features` 输出直接对接，迁移零摩擦。
    """

    ticker: str
    features: Dict[str, Any] = field(default_factory=dict)
    position: Position = field(default_factory=Position)
    account: Account = field(default_factory=Account)
    allow_short: bool = False
    asof: str = ""

    # --- 便捷只读访问器（避免每个 agent 重复 .get 防御） --- #
    @property
    def technical(self) -> Dict[str, Any]:
        t = self.features.get("technical")
        return t if isinstance(t, dict) else {}

    @property
    def signal(self) -> Dict[str, Any]:
        s = self.features.get("signal")
        return s if isinstance(s, dict) else {}

    @property
    def macro(self) -> Dict[str, Any]:
        """全局宏观快照 {"vix": float, "tnx": float}（由集成层按日填入；缺省空 dict）。"""
        m = self.features.get("macro")
        return m if isinstance(m, dict) else {}

    @property
    def volatility_ann_pct(self) -> float:
        try:
            return float(self.features.get("volatility_ann_pct", 0.0) or 0.0)
        except Exception:
            return 0.0


def flatten_features(features: Dict[str, Any]) -> Dict[str, float]:
    """把 {"technical": {...}, "signal": {...}, "volatility_ann_pct": x} 摊平成
    扁平的 `{name: float}`，供 Planner-SFT / Gatekeeper-RL 的 MLP 喂入。

    等价迁移自旧 `strategy.py::_flatten_gate_features`：signal 字段加 `signal_` 前缀，
    非数值字段静默跳过。
    """
    out: Dict[str, float] = {}
    tech = features.get("technical") if isinstance(features.get("technical"), dict) else {}
    sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
    for k, v in tech.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    for k, v in sig.items():
        try:
            out["signal_" + str(k)] = float(v)
        except Exception:
            continue
    try:
        out["volatility_ann_pct"] = float(features.get("volatility_ann_pct") or 0.0)
    except Exception:
        out["volatility_ann_pct"] = 0.0
    return out


@dataclass
class ExpertDecision:
    """专家（scalper/analyst/news）的方向决策。"""

    decision: str = Action.HOLD
    analysis: str = ""
    expert: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "analysis": self.analysis,
            "expert": self.expert,
            "meta": dict(self.meta),
        }


@dataclass
class FinalDecision:
    """编排器（orchestrator）合并所有 agent 后的最终裁决。"""

    action: str = Action.HOLD
    approved: bool = False
    reason: str = ""
    expert: str = ""
    regime: str = ""
    chart_score: int = 0
    macro_label: str = "NEUTRAL"
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "approved": self.approved,
            "reason": self.reason,
            "expert": self.expert,
            "regime": self.regime,
            "chart_score": int(self.chart_score),
            "macro_label": self.macro_label,
            "trace": dict(self.trace),
        }
