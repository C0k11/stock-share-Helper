"""回撤控制：回撤序列 / 当前回撤 / 风险分级 / 最大回撤期间。

从旧 `src/risk/drawdown.py` 迁移，口径不变，加类型标注。
因果：`expanding().max()` 只用历史高点；风险分级是对"已知回撤"的纯映射。
`compute_max_drawdown_period` 是对完整净值曲线的**事后分析**（报告用），非交易决策。
"""

from __future__ import annotations

import pandas as pd


class DrawdownController:
    """回撤监控与风险分级（normal/warning/danger/halt）。"""

    def __init__(
        self,
        max_drawdown: float = 0.10,
        warning_threshold: float = 0.05,
        halt_threshold: float = 0.25,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold

    def compute_drawdown(self, equity_curve: pd.Series) -> pd.DataFrame:
        """peak(历史高点) / drawdown(负数) / drawdown_duration(水下天数)。"""
        result = pd.DataFrame(index=equity_curve.index)
        result["peak"] = equity_curve.expanding().max()
        result["drawdown"] = (equity_curve - result["peak"]) / result["peak"]
        is_underwater = result["drawdown"] < 0
        result["drawdown_duration"] = is_underwater.groupby(
            (~is_underwater).cumsum()
        ).cumsum()
        return result

    def get_current_drawdown(self, equity_curve: pd.Series) -> dict:
        """最新一日的回撤快照 + 历史最大回撤。"""
        dd = self.compute_drawdown(equity_curve)
        current = dd.iloc[-1]
        return {
            "current_drawdown": float(current["drawdown"]),
            "current_duration_days": int(current["drawdown_duration"]),
            "max_historical_drawdown": float(dd["drawdown"].min()),
            "peak_value": float(current["peak"]),
            "current_value": float(equity_curve.iloc[-1]),
        }

    def check_risk_level(self, current_drawdown: float) -> dict:
        """把当前回撤映射到 level/action/position_scale。"""
        dd = abs(current_drawdown)
        if dd >= self.halt_threshold:
            return {
                "level": "halt",
                "action": "halt",
                "position_scale": 0.0,
                "message": f"回撤{dd:.1%}超过暂停阈值{self.halt_threshold:.1%}，建议暂停交易",
            }
        if dd >= self.max_drawdown:
            return {
                "level": "danger",
                "action": "reduce",
                "position_scale": 0.3,
                "message": f"回撤{dd:.1%}超过最大容忍{self.max_drawdown:.1%}，建议大幅减仓",
            }
        if dd >= self.warning_threshold:
            return {
                "level": "warning",
                "action": "reduce",
                "position_scale": 0.7,
                "message": f"回撤{dd:.1%}超过预警阈值{self.warning_threshold:.1%}，建议适度减仓",
            }
        return {
            "level": "normal",
            "action": "none",
            "position_scale": 1.0,
            "message": "回撤在正常范围内",
        }

    def compute_max_drawdown_period(self, equity_curve: pd.Series) -> dict:
        """事后分析：最大回撤的 peak/trough/recovery 日期与天数（报告用）。"""
        dd = self.compute_drawdown(equity_curve)
        max_dd_idx = dd["drawdown"].idxmin()
        max_dd_value = dd.loc[max_dd_idx, "drawdown"]
        peak_idx = dd.loc[:max_dd_idx, "peak"].idxmax()
        after_max_dd = dd.loc[max_dd_idx:]
        recovered = after_max_dd[after_max_dd["drawdown"] >= 0]
        recovery_idx = recovered.index[0] if len(recovered) > 0 else None
        return {
            "max_drawdown": float(max_dd_value),
            "peak_date": str(peak_idx),
            "trough_date": str(max_dd_idx),
            "recovery_date": str(recovery_idx) if recovery_idx is not None else None,
            "drawdown_days": (max_dd_idx - peak_idx).days
            if hasattr(max_dd_idx - peak_idx, "days")
            else None,
            "recovery_days": (recovery_idx - max_dd_idx).days
            if recovery_idx is not None and hasattr(recovery_idx - max_dd_idx, "days")
            else None,
        }
