"""
技术因子模块 - 均线、动量、波动率等
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class TechnicalFeatures:
    """技术指标计算"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 价格DataFrame，需包含 close, high, low, volume
        """
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
    
    def add_moving_averages(
        self,
        windows: list = [5, 10, 20, 50, 200]
    ) -> "TechnicalFeatures":
        """添加移动平均线"""
        for w in windows:
            self.features[f"ma_{w}"] = self.df["close"].rolling(w).mean()
            self.features[f"ma_{w}_slope"] = self.features[f"ma_{w}"].pct_change(5)
        
        # 价格相对于均线的位置
        if 20 in windows:
            self.features["price_vs_ma20"] = self.df["close"] / self.features["ma_20"] - 1
        if 200 in windows:
            self.features["price_vs_ma200"] = self.df["close"] / self.features["ma_200"] - 1
        
        return self
    
    def add_momentum(
        self,
        windows: list = [5, 10, 21, 63, 126, 252]
    ) -> "TechnicalFeatures":
        """添加动量指标"""
        for w in windows:
            self.features[f"return_{w}d"] = self.df["close"].pct_change(w)
        
        # 动量排名（用于跨标的比较）
        if 63 in windows and 252 in windows:
            # 3-12个月动量（经典动量因子）
            self.features["momentum_3_12m"] = (
                self.features["return_63d"] * 0.5 + 
                self.features["return_252d"] * 0.5
            )
        
        return self
    
    def add_volatility(
        self,
        windows: list = [10, 20, 60]
    ) -> "TechnicalFeatures":
        """添加波动率指标"""
        returns = self.df["close"].pct_change()
        
        for w in windows:
            # 历史波动率（年化）
            self.features[f"volatility_{w}d"] = returns.rolling(w).std() * np.sqrt(252)
        
        # 波动率变化率
        if 20 in windows:
            self.features["vol_change"] = (
                self.features["volatility_20d"] / 
                self.features["volatility_20d"].shift(20) - 1
            )
        
        # 波动率相对水平
        if 20 in windows and 60 in windows:
            self.features["vol_ratio"] = (
                self.features["volatility_20d"] / 
                self.features["volatility_60d"]
            )
        
        return self
    
    def add_drawdown(self) -> "TechnicalFeatures":
        """添加回撤指标"""
        # 历史最高点
        self.features["rolling_max"] = self.df["close"].expanding().max()
        
        # 当前回撤
        self.features["drawdown"] = (
            self.df["close"] / self.features["rolling_max"] - 1
        )
        
        # 最近N日最大回撤
        for w in [20, 60, 252]:
            rolling_max = self.df["close"].rolling(w).max()
            rolling_min = self.df["close"].rolling(w).min()
            self.features[f"max_drawdown_{w}d"] = (rolling_min - rolling_max) / rolling_max
        
        return self
    
    def add_trend_signals(self) -> "TechnicalFeatures":
        """添加趋势信号"""
        # 均线多头排列
        if all(f"ma_{w}" in self.features.columns for w in [20, 50, 200]):
            self.features["trend_alignment"] = (
                (self.features["ma_20"] > self.features["ma_50"]).astype(int) +
                (self.features["ma_50"] > self.features["ma_200"]).astype(int)
            ) / 2
        
        # 价格突破
        self.features["breakout_20d_high"] = (
            self.df["close"] >= self.df["high"].rolling(20).max()
        ).astype(int)
        
        self.features["breakdown_20d_low"] = (
            self.df["close"] <= self.df["low"].rolling(20).min()
        ).astype(int)
        
        return self
    
    def add_all(self) -> "TechnicalFeatures":
        """添加所有技术指标"""
        return (
            self.add_moving_averages()
            .add_momentum()
            .add_volatility()
            .add_drawdown()
            .add_trend_signals()
        )
    
    def get_features(self) -> pd.DataFrame:
        """获取计算后的特征DataFrame"""
        return self.features
    
    def get_latest(self) -> dict:
        """获取最新一行特征"""
        return self.features.iloc[-1].to_dict()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术特征的便捷函数"""
    return TechnicalFeatures(df).add_all().get_features()
