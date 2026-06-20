"""Macro Governor —— 基于 VIX / 10Y 收益率的**确定性**全局宏观风险闸。

旧 `strategy.py::_macro_governor_assess`：

    score = random.uniform(0.3, 0.8)   # "Simulated risk score"

即**纯随机**、零真实宏观输入——伪装成功能的半成品。本版方针：**删随机 + 接真实信号**。
现在用 `PriceFetcher` 已在抓的 ^VIX / ^TNX（见 `quantai.data.prices`），按阈值确定性映射：

- VIX >= `vix_risk_off`（或 10Y 收益率 >= `tnx_risk_off`）-> RISK_OFF（gear=-1，提示缩风险）；
- VIX <= `vix_risk_on` 且非 risk-off -> RISK_ON（gear=+1）；
- 其余 / 无数据 / 未启用 -> NEUTRAL（gear=0，不影响决策）。

gear/label 作为上下文喂给 System2 critic（论据），不直接改仓位。全程无随机、可复现。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from quantai.config.schema import MacroConfig


class MacroGovernor:
    """宏观闸：返回 (gear, label)。gear<0=RISK_OFF，>0=RISK_ON，=0=NEUTRAL。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        vix_risk_off: float = 28.0,
        vix_risk_on: float = 15.0,
        tnx_risk_off: float = 4.8,
    ) -> None:
        self.enabled = bool(enabled)
        self.vix_risk_off = float(vix_risk_off)
        self.vix_risk_on = float(vix_risk_on)
        self.tnx_risk_off = float(tnx_risk_off)

    @classmethod
    def from_config(cls, cfg: MacroConfig) -> "MacroGovernor":
        return cls(
            enabled=cfg.enabled,
            vix_risk_off=cfg.vix_risk_off,
            vix_risk_on=cfg.vix_risk_on,
            tnx_risk_off=cfg.tnx_risk_off,
        )

    def assess(
        self, vix: Optional[Any] = None, tnx: Optional[Any] = None
    ) -> Tuple[float, str]:
        """由 VIX/10Y 确定性得出 (gear, label)。

        - 未启用 / 无数据：`(0.0, "NEUTRAL")`。
        - VIX 高或 10Y 高：RISK_OFF（-1）；VIX 低且非 risk-off：RISK_ON（+1）；否则 NEUTRAL。
        """
        if not self.enabled:
            return 0.0, "NEUTRAL"
        v = _to_float(vix)
        t = _to_float(tnx)
        risk_off = (v is not None and v >= self.vix_risk_off) or (
            t is not None and t >= self.tnx_risk_off
        )
        if risk_off:
            return -1.0, "RISK_OFF"
        if v is not None and v <= self.vix_risk_on:
            return 1.0, "RISK_ON"
        return 0.0, "NEUTRAL"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
