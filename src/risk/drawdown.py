"""
回撤控制模块
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class DrawdownController:
    """回撤控制器"""
    
    def __init__(
        self,
        max_drawdown: float = 0.10,
        warning_threshold: float = 0.05,
        halt_threshold: float = 0.25
    ):
        """
        Args:
            max_drawdown: 用户风险档位的最大回撤
            warning_threshold: 预警阈值
            halt_threshold: 暂停交易阈值
        """
        self.max_drawdown = max_drawdown
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold
    
    def compute_drawdown(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        计算回撤序列
        
        Returns:
            包含 peak, drawdown, drawdown_duration 的DataFrame
        """
        result = pd.DataFrame(index=equity_curve.index)
        
        # 历史最高点
        result["peak"] = equity_curve.expanding().max()
        
        # 回撤（负数）
        result["drawdown"] = (equity_curve - result["peak"]) / result["peak"]
        
        # 回撤持续时间
        is_underwater = result["drawdown"] < 0
        result["drawdown_duration"] = is_underwater.groupby(
            (~is_underwater).cumsum()
        ).cumsum()
        
        return result
    
    def get_current_drawdown(self, equity_curve: pd.Series) -> Dict:
        """获取当前回撤状态"""
        dd = self.compute_drawdown(equity_curve)
        current = dd.iloc[-1]
        
        # 历史最大回撤
        max_dd = dd["drawdown"].min()
        
        return {
            "current_drawdown": float(current["drawdown"]),
            "current_duration_days": int(current["drawdown_duration"]),
            "max_historical_drawdown": float(max_dd),
            "peak_value": float(current["peak"]),
            "current_value": float(equity_curve.iloc[-1])
        }
    
    def check_risk_level(self, current_drawdown: float) -> Dict:
        """
        检查风险级别
        
        Returns:
            {
                "level": "normal" | "warning" | "danger" | "halt",
                "action": "none" | "reduce" | "hedge" | "halt",
                "position_scale": 0.0 - 1.0
            }
        """
        dd = abs(current_drawdown)
        
        if dd >= self.halt_threshold:
            return {
                "level": "halt",
                "action": "halt",
                "position_scale": 0.0,
                "message": f"回撤{dd:.1%}超过暂停阈值{self.halt_threshold:.1%}，建议暂停交易"
            }
        elif dd >= self.max_drawdown:
            return {
                "level": "danger",
                "action": "reduce",
                "position_scale": 0.3,
                "message": f"回撤{dd:.1%}超过最大容忍{self.max_drawdown:.1%}，建议大幅减仓"
            }
        elif dd >= self.warning_threshold:
            return {
                "level": "warning",
                "action": "reduce",
                "position_scale": 0.7,
                "message": f"回撤{dd:.1%}超过预警阈值{self.warning_threshold:.1%}，建议适度减仓"
            }
        else:
            return {
                "level": "normal",
                "action": "none",
                "position_scale": 1.0,
                "message": "回撤在正常范围内"
            }
    
    def compute_max_drawdown_period(
        self,
        equity_curve: pd.Series
    ) -> Dict:
        """计算最大回撤期间"""
        dd = self.compute_drawdown(equity_curve)
        
        # 最大回撤点
        max_dd_idx = dd["drawdown"].idxmin()
        max_dd_value = dd.loc[max_dd_idx, "drawdown"]
        
        # 找到之前的peak
        peak_idx = dd.loc[:max_dd_idx, "peak"].idxmax()
        
        # 找到恢复点（如果有）
        after_max_dd = dd.loc[max_dd_idx:]
        recovered = after_max_dd[after_max_dd["drawdown"] >= 0]
        recovery_idx = recovered.index[0] if len(recovered) > 0 else None
        
        return {
            "max_drawdown": float(max_dd_value),
            "peak_date": str(peak_idx),
            "trough_date": str(max_dd_idx),
            "recovery_date": str(recovery_idx) if recovery_idx else None,
            "drawdown_days": (max_dd_idx - peak_idx).days if hasattr(max_dd_idx - peak_idx, 'days') else None,
            "recovery_days": (recovery_idx - max_dd_idx).days if recovery_idx and hasattr(recovery_idx - max_dd_idx, 'days') else None
        }
