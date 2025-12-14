"""
风险状态检测模块 - 判断当前市场处于Risk-On/Risk-Off/Transition
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from enum import Enum
from loguru import logger


class MarketRegime(Enum):
    """市场风险状态"""
    RISK_ON = "risk_on"         # 风险偏好，进攻
    RISK_OFF = "risk_off"       # 风险规避，防守
    TRANSITION = "transition"   # 过渡期，谨慎


class RegimeDetector:
    """风险状态检测器"""
    
    def __init__(
        self,
        vix_high: float = 25.0,
        vix_low: float = 15.0,
        trend_ma: int = 50
    ):
        """
        Args:
            vix_high: VIX高阈值，超过则Risk-Off
            vix_low: VIX低阈值，低于则Risk-On
            trend_ma: 趋势判断用的均线周期
        """
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.trend_ma = trend_ma
    
    def detect(
        self,
        spy_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        检测市场风险状态
        
        Args:
            spy_data: SPY价格数据
            vix_data: VIX数据（可选）
        
        Returns:
            包含regime列的DataFrame
        """
        result = pd.DataFrame(index=spy_data.index)
        
        # 1. 趋势信号：价格 vs MA
        ma = spy_data["close"].rolling(self.trend_ma).mean()
        result["trend_signal"] = (spy_data["close"] > ma).astype(int)
        
        # 2. VIX信号
        if vix_data is not None and "close" in vix_data.columns:
            # 对齐索引
            vix_aligned = vix_data["close"].reindex(spy_data.index, method="ffill")
            result["vix"] = vix_aligned
            result["vix_signal"] = 0
            result.loc[vix_aligned < self.vix_low, "vix_signal"] = 1   # Risk-On
            result.loc[vix_aligned > self.vix_high, "vix_signal"] = -1 # Risk-Off
        else:
            result["vix_signal"] = 0
        
        # 3. 波动率变化信号
        returns = spy_data["close"].pct_change()
        vol_short = returns.rolling(10).std() * np.sqrt(252)
        vol_long = returns.rolling(60).std() * np.sqrt(252)
        result["vol_expansion"] = (vol_short > vol_long * 1.2).astype(int) * -1
        
        # 4. 综合判断
        result["score"] = (
            result["trend_signal"] + 
            result["vix_signal"] + 
            result["vol_expansion"]
        )
        
        # 映射到Regime
        result["regime"] = MarketRegime.TRANSITION.value
        result.loc[result["score"] >= 1, "regime"] = MarketRegime.RISK_ON.value
        result.loc[result["score"] <= -1, "regime"] = MarketRegime.RISK_OFF.value
        
        return result
    
    def get_current_regime(
        self,
        spy_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """获取当前风险状态"""
        result = self.detect(spy_data, vix_data)
        latest = result.iloc[-1]
        
        return {
            "regime": latest["regime"],
            "score": latest["score"],
            "trend_signal": latest["trend_signal"],
            "vix_signal": latest["vix_signal"],
            "vol_expansion": latest["vol_expansion"],
            "vix": latest.get("vix", None),
            "date": str(result.index[-1])
        }
    
    def get_regime_history(
        self,
        spy_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """获取历史风险状态序列"""
        return self.detect(spy_data, vix_data)[["regime", "score"]]


def detect_regime(
    spy_data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None
) -> str:
    """检测当前风险状态的便捷函数"""
    detector = RegimeDetector()
    result = detector.get_current_regime(spy_data, vix_data)
    return result["regime"]
