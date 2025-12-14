"""
信号生成模块 - 趋势/动量信号
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 200,
        momentum_lookback: int = 63
    ):
        """
        Args:
            ma_short: 短期均线
            ma_long: 长期均线
            momentum_lookback: 动量回看周期
        """
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.momentum_lookback = momentum_lookback
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 价格数据，需包含close
        
        Returns:
            包含各类信号的DataFrame
        """
        signals = pd.DataFrame(index=df.index)
        close = df["close"]
        
        # 1. 趋势信号：价格 vs 长期均线
        ma_long = close.rolling(self.ma_long).mean()
        signals["trend_signal"] = np.where(close > ma_long, 1, -1)
        
        # 2. 动量信号：过去N日收益
        momentum = close.pct_change(self.momentum_lookback)
        signals["momentum_signal"] = np.where(momentum > 0, 1, -1)
        
        # 3. 均线交叉信号
        ma_short = close.rolling(self.ma_short).mean()
        signals["ma_cross_signal"] = np.where(ma_short > ma_long, 1, -1)
        
        # 4. 突破信号
        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        signals["breakout_signal"] = 0
        signals.loc[close >= high_20, "breakout_signal"] = 1
        signals.loc[close <= low_20, "breakout_signal"] = -1
        
        # 5. 综合信号（简单平均）
        signals["composite_signal"] = (
            signals["trend_signal"] * 0.3 +
            signals["momentum_signal"] * 0.3 +
            signals["ma_cross_signal"] * 0.2 +
            signals["breakout_signal"] * 0.2
        )
        
        # 6. 离散化：强多/弱多/中性/弱空/强空
        signals["signal_strength"] = pd.cut(
            signals["composite_signal"],
            bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            labels=["strong_short", "weak_short", "neutral", "weak_long", "strong_long"]
        )
        
        return signals
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict:
        """获取当前信号"""
        signals = self.generate(df)
        latest = signals.iloc[-1]
        
        return {
            "date": str(signals.index[-1]),
            "trend": int(latest["trend_signal"]),
            "momentum": int(latest["momentum_signal"]),
            "ma_cross": int(latest["ma_cross_signal"]),
            "breakout": int(latest["breakout_signal"]),
            "composite": float(latest["composite_signal"]),
            "strength": str(latest["signal_strength"])
        }


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """生成信号的便捷函数"""
    return SignalGenerator().generate(df)
