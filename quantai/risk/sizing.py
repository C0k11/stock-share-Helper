"""仓位计算：波动率目标 / 信号强度映射 / 风险状态调整 / 用户风险档位。

从旧 `src/strategy/position.py` 迁移，口径不变，加类型标注、显式 `fill_method=None`。
因果：波动率目标用 trailing rolling std；最终仓位在 t 处只用 <= t 的信息。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZE = np.sqrt(252)

_SIGNAL_TO_POSITION = {
    "strong_long": 1.0,
    "weak_long": 0.6,
    "neutral": 0.3,
    "weak_short": 0.1,
    "strong_short": 0.0,
}

_REGIME_FACTOR = {
    "risk_on": 1.0,
    "transition": 0.7,
    "risk_off": 0.4,
}

_PROFILE_LIMITS = {
    "conservative": {"max_equity": 0.4, "scale": 0.6},
    "balanced": {"max_equity": 0.6, "scale": 0.8},
    "aggressive": {"max_equity": 0.8, "scale": 1.0},
}


class PositionSizer:
    """把信号/波动率/风险状态综合成目标仓位。"""

    def __init__(
        self,
        target_volatility: float = 0.10,
        max_position: float = 1.0,
        min_position: float = 0.0,
        vol_lookback: int = 20,
    ) -> None:
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.min_position = min_position
        self.vol_lookback = vol_lookback

    def compute_vol_target_position(self, df: pd.DataFrame) -> pd.Series:
        """波动率目标仓位 = 目标波动率 / 实现波动率，截断到 [min, max]。"""
        returns = df["close"].pct_change(fill_method=None)
        realized_vol = returns.rolling(self.vol_lookback).std() * _ANNUALIZE
        raw_position = self.target_volatility / realized_vol
        return raw_position.clip(self.min_position, self.max_position)

    def compute_signal_position(self, signal_strength: pd.Series) -> pd.Series:
        """信号强度标签 -> 仓位。"""
        return signal_strength.map(_SIGNAL_TO_POSITION)

    def compute_regime_adjustment(self, regime: pd.Series) -> pd.Series:
        """风险状态 -> 仓位系数。"""
        return regime.map(_REGIME_FACTOR)

    def compute_final_position(
        self,
        df: pd.DataFrame,
        signal_strength: pd.Series,
        regime: pd.Series,
    ) -> pd.DataFrame:
        """最终仓位 = min(波动率目标, 信号仓位) * 风险状态系数，再截断。"""
        result = pd.DataFrame(index=df.index)
        result["vol_position"] = self.compute_vol_target_position(df)
        result["signal_position"] = self.compute_signal_position(signal_strength)
        result["regime_factor"] = self.compute_regime_adjustment(regime)
        result["target_position"] = (
            np.minimum(result["vol_position"], result["signal_position"])
            * result["regime_factor"]
        ).clip(self.min_position, self.max_position)
        return result

    def apply_risk_profile(self, position: float, profile: str = "balanced") -> float:
        """套用用户风险档位（保守/平衡/激进）的缩放与权益上限。"""
        limit = _PROFILE_LIMITS.get(profile, _PROFILE_LIMITS["balanced"])
        scaled = position * limit["scale"]
        return min(scaled, limit["max_equity"])


def compute_target_position(df: pd.DataFrame, target_vol: float = 0.10) -> pd.Series:
    """便捷函数：仅按波动率目标计算仓位序列。"""
    return PositionSizer(target_volatility=target_vol).compute_vol_target_position(df)
