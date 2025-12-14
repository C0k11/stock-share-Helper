"""
仓位计算模块 - 目标波动率、风险档位映射
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class PositionSizer:
    """仓位计算器"""
    
    def __init__(
        self,
        target_volatility: float = 0.10,
        max_position: float = 1.0,
        min_position: float = 0.0,
        vol_lookback: int = 20
    ):
        """
        Args:
            target_volatility: 目标年化波动率
            max_position: 最大仓位
            min_position: 最小仓位
            vol_lookback: 波动率计算窗口
        """
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.min_position = min_position
        self.vol_lookback = vol_lookback
    
    def compute_vol_target_position(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        基于波动率目标计算仓位
        
        核心逻辑：目标仓位 = 目标波动率 / 实际波动率
        波动率高时降仓，波动率低时加仓
        """
        returns = df["close"].pct_change()
        realized_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # 波动率目标仓位
        raw_position = self.target_volatility / realized_vol
        
        # 限制在[min, max]范围
        position = raw_position.clip(self.min_position, self.max_position)
        
        return position
    
    def compute_signal_position(
        self,
        signal_strength: pd.Series
    ) -> pd.Series:
        """
        基于信号强度计算仓位
        
        信号映射：
        strong_long -> 1.0
        weak_long -> 0.6
        neutral -> 0.3
        weak_short -> 0.1
        strong_short -> 0.0
        """
        mapping = {
            "strong_long": 1.0,
            "weak_long": 0.6,
            "neutral": 0.3,
            "weak_short": 0.1,
            "strong_short": 0.0
        }
        return signal_strength.map(mapping)
    
    def compute_regime_adjustment(
        self,
        regime: pd.Series
    ) -> pd.Series:
        """
        基于风险状态调整仓位系数
        
        Risk-On: 1.0 (不调整)
        Transition: 0.7 (降低30%)
        Risk-Off: 0.4 (降低60%)
        """
        mapping = {
            "risk_on": 1.0,
            "transition": 0.7,
            "risk_off": 0.4
        }
        return regime.map(mapping)
    
    def compute_final_position(
        self,
        df: pd.DataFrame,
        signal_strength: pd.Series,
        regime: pd.Series
    ) -> pd.DataFrame:
        """
        计算最终目标仓位
        
        最终仓位 = min(波动率目标仓位, 信号仓位) * 风险状态系数
        """
        result = pd.DataFrame(index=df.index)
        
        # 各组件
        result["vol_position"] = self.compute_vol_target_position(df)
        result["signal_position"] = self.compute_signal_position(signal_strength)
        result["regime_factor"] = self.compute_regime_adjustment(regime)
        
        # 最终仓位
        result["target_position"] = (
            np.minimum(result["vol_position"], result["signal_position"]) *
            result["regime_factor"]
        ).clip(self.min_position, self.max_position)
        
        return result
    
    def apply_risk_profile(
        self,
        position: float,
        profile: str = "balanced"
    ) -> float:
        """
        应用用户风险档位约束
        
        Args:
            position: 原始目标仓位
            profile: conservative/balanced/aggressive
        """
        profile_limits = {
            "conservative": {"max_equity": 0.4, "scale": 0.6},
            "balanced": {"max_equity": 0.6, "scale": 0.8},
            "aggressive": {"max_equity": 0.8, "scale": 1.0}
        }
        
        limit = profile_limits.get(profile, profile_limits["balanced"])
        scaled = position * limit["scale"]
        return min(scaled, limit["max_equity"])


def compute_target_position(
    df: pd.DataFrame,
    target_vol: float = 0.10
) -> pd.Series:
    """计算目标仓位的便捷函数"""
    sizer = PositionSizer(target_volatility=target_vol)
    return sizer.compute_vol_target_position(df)
