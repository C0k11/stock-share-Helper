"""quantai.risk —— 风险层（仓位计算 / 波动率管理 / 回撤控制 / 风险闸门 / 预警）。

全部因果（trailing 窗口或纯函数），无 lookahead。

用法：
    from quantai.risk import PositionSizer, VolatilityManager, DrawdownController, RiskGate, RiskAlerts
"""

from .alerts import AlertSeverity, AlertType, RiskAlert, RiskAlerts
from .drawdown import DrawdownController
from .gate import RiskGate
from .sizing import PositionSizer, compute_target_position
from .volatility import VolatilityManager

__all__ = [
    "PositionSizer",
    "compute_target_position",
    "VolatilityManager",
    "DrawdownController",
    "RiskGate",
    "RiskAlerts",
    "RiskAlert",
    "AlertSeverity",
    "AlertType",
]
