"""
波动率管理模块
"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger


class VolatilityManager:
    """波动率管理器"""
    
    def __init__(
        self,
        target_volatility: float = 0.10,
        vol_ceiling: float = 0.25,
        lookback_short: int = 20,
        lookback_long: int = 60
    ):
        """
        Args:
            target_volatility: 目标年化波动率
            vol_ceiling: 波动率上限
            lookback_short: 短期波动率窗口
            lookback_long: 长期波动率窗口
        """
        self.target_volatility = target_volatility
        self.vol_ceiling = vol_ceiling
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
    
    def compute_realized_volatility(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """计算历史实现波动率（年化）"""
        returns = prices.pct_change()
        vol = returns.rolling(window).std() * np.sqrt(252)
        return vol
    
    def compute_vol_regime(self, prices: pd.Series) -> pd.DataFrame:
        """
        计算波动率状态
        
        Returns:
            包含 vol_short, vol_long, vol_ratio, vol_regime 的DataFrame
        """
        result = pd.DataFrame(index=prices.index)
        
        result["vol_short"] = self.compute_realized_volatility(prices, self.lookback_short)
        result["vol_long"] = self.compute_realized_volatility(prices, self.lookback_long)
        result["vol_ratio"] = result["vol_short"] / result["vol_long"]
        
        # 波动率状态
        result["vol_regime"] = "normal"
        result.loc[result["vol_ratio"] > 1.3, "vol_regime"] = "expanding"
        result.loc[result["vol_ratio"] < 0.7, "vol_regime"] = "contracting"
        result.loc[result["vol_short"] > self.vol_ceiling, "vol_regime"] = "extreme"
        
        return result
    
    def get_position_scalar(self, current_vol: float) -> float:
        """
        基于当前波动率计算仓位缩放因子
        
        目标：让组合波动率稳定在目标水平
        """
        if current_vol <= 0:
            return 1.0
        
        scalar = self.target_volatility / current_vol
        
        # 限制缩放范围
        scalar = np.clip(scalar, 0.2, 1.5)
        
        return float(scalar)
    
    def check_vol_alert(self, prices: pd.Series) -> Dict:
        """检查波动率预警"""
        vol_data = self.compute_vol_regime(prices)
        current = vol_data.iloc[-1]
        
        alerts = []
        
        # 波动率扩张
        if current["vol_regime"] == "expanding":
            alerts.append({
                "type": "vol_expansion",
                "severity": "medium",
                "message": f"短期波动率({current['vol_short']:.1%})高于长期水平({current['vol_long']:.1%})"
            })
        
        # 极端波动
        if current["vol_regime"] == "extreme":
            alerts.append({
                "type": "vol_extreme",
                "severity": "high",
                "message": f"波动率({current['vol_short']:.1%})超过上限({self.vol_ceiling:.1%})"
            })
        
        return {
            "current_vol": float(current["vol_short"]),
            "vol_regime": current["vol_regime"],
            "position_scalar": self.get_position_scalar(current["vol_short"]),
            "alerts": alerts
        }
