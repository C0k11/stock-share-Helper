"""Macro Governor —— 全局宏观风险闸（**诚实占位**）。

旧 `strategy.py::_macro_governor_assess` 是这样的：

    score = random.uniform(0.3, 0.8)   # "Simulated risk score"
    -> NEUTRAL / LOW / DRIVE

即**纯随机**，没有任何真实宏观输入——属于"伪装成功能的半成品"。重构按"代码即真相"
原则**删掉随机**：默认恒返回 `NEUTRAL`（gear=0，不加不减仓），并显式声明"真实宏观信号
尚未实现"。可选地接受一个 `risk_map`（如外部算好的 {regime: gear}）做确定性查表。

详见 docs/backlog.md（macro governor 随机已删，真实信号待实现/待裁决）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


class MacroGovernor:
    """宏观闸：返回 (gear, label)。gear=0 表示中性（默认）。"""

    def __init__(
        self, *, enabled: bool = False, risk_map: Optional[Dict[str, float]] = None
    ) -> None:
        self.enabled = bool(enabled)
        self.risk_map = dict(risk_map or {})

    @classmethod
    def from_config(cls, macro_enabled: bool) -> "MacroGovernor":
        return cls(enabled=bool(macro_enabled))

    def assess(self, label: str = "") -> Tuple[float, str]:
        """返回 (gear, label)。

        - 未启用（默认）：恒 `(0.0, "NEUTRAL")`——确定性、不影响决策。
        - 启用 + 提供了 label 且命中 risk_map：返回该确定性 gear（查表，无随机）。
        - 启用但无可用信号：仍返回中性，并把 label 标为 "NEUTRAL"（不杜撰风险）。
        """
        if not self.enabled:
            return 0.0, "NEUTRAL"
        key = str(label or "").strip()
        if key and key in self.risk_map:
            return float(self.risk_map[key]), key
        return 0.0, "NEUTRAL"
