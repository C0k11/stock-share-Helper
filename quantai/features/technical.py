"""技术因子：均线 / 动量 / 波动率 / 回撤 / 趋势信号。

从旧 `src/features/technical.py` 迁移，**计算口径保持不变**（列名、窗口一致）；
改动仅限工程质量：加类型标注、消除可变默认参数。

无 lookahead：所有窗口都是**因果**的——
`rolling(w)` / `expanding()` / `pct_change(w)` / `shift(+n)` 只用 <= t 的数据。
`tests/features/test_no_lookahead.py` 用「截断序列在 t 处特征值不变」断言了这一点。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

_DEFAULT_MA: tuple[int, ...] = (5, 10, 20, 50, 200)
_DEFAULT_MOMENTUM: tuple[int, ...] = (5, 10, 21, 63, 126, 252)
_DEFAULT_VOL: tuple[int, ...] = (10, 20, 60)
_TRADING_DAYS = 252
_ANNUALIZE = np.sqrt(_TRADING_DAYS)


class TechnicalFeatures:
    """链式构建器：`TechnicalFeatures(df).add_all().get_features()`。

    输入 df 需含列 close / high / low（volume 可选）。
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)

    def add_moving_averages(
        self, windows: Optional[Sequence[int]] = None
    ) -> "TechnicalFeatures":
        """各周期均线、均线斜率(5日 pct_change)、价格相对 ma20/ma200 的位置。"""
        ws = tuple(windows) if windows is not None else _DEFAULT_MA
        close = self.df["close"]
        for w in ws:
            self.features[f"ma_{w}"] = close.rolling(w).mean()
            self.features[f"ma_{w}_slope"] = self.features[f"ma_{w}"].pct_change(5, fill_method=None)
        if 20 in ws:
            self.features["price_vs_ma20"] = close / self.features["ma_20"] - 1
        if 200 in ws:
            self.features["price_vs_ma200"] = close / self.features["ma_200"] - 1
        return self

    def add_momentum(
        self, windows: Optional[Sequence[int]] = None
    ) -> "TechnicalFeatures":
        """各周期 N 日收益；经典 3-12 月动量因子。"""
        ws = tuple(windows) if windows is not None else _DEFAULT_MOMENTUM
        close = self.df["close"]
        for w in ws:
            self.features[f"return_{w}d"] = close.pct_change(w, fill_method=None)
        if 63 in ws and 252 in ws:
            self.features["momentum_3_12m"] = (
                self.features["return_63d"] * 0.5 + self.features["return_252d"] * 0.5
            )
        return self

    def add_volatility(
        self, windows: Optional[Sequence[int]] = None
    ) -> "TechnicalFeatures":
        """年化历史波动率；波动率变化率(对比20日前)；20/60日波动率比。"""
        ws = tuple(windows) if windows is not None else _DEFAULT_VOL
        returns = self.df["close"].pct_change(fill_method=None)
        for w in ws:
            self.features[f"volatility_{w}d"] = returns.rolling(w).std() * _ANNUALIZE
        if 20 in ws:
            self.features["vol_change"] = (
                self.features["volatility_20d"] / self.features["volatility_20d"].shift(20) - 1
            )
        if 20 in ws and 60 in ws:
            self.features["vol_ratio"] = (
                self.features["volatility_20d"] / self.features["volatility_60d"]
            )
        return self

    def add_drawdown(self) -> "TechnicalFeatures":
        """累计高点回撤；近 N 日区间最大回撤。"""
        close = self.df["close"]
        self.features["rolling_max"] = close.expanding().max()
        self.features["drawdown"] = close / self.features["rolling_max"] - 1
        for w in (20, 60, 252):
            rolling_max = close.rolling(w).max()
            rolling_min = close.rolling(w).min()
            self.features[f"max_drawdown_{w}d"] = (rolling_min - rolling_max) / rolling_max
        return self

    def add_trend_signals(self) -> "TechnicalFeatures":
        """均线多头排列度；20 日突破/跌破标志。

        注：breakout_20d_high 的 20 日高**含当日**（口径与旧版一致），
        因此偏保守，几乎只在"当日收于 20 日新高"时触发。
        """
        if all(f"ma_{w}" in self.features.columns for w in (20, 50, 200)):
            self.features["trend_alignment"] = (
                (self.features["ma_20"] > self.features["ma_50"]).astype(int)
                + (self.features["ma_50"] > self.features["ma_200"]).astype(int)
            ) / 2
        close = self.df["close"]
        self.features["breakout_20d_high"] = (
            close >= self.df["high"].rolling(20).max()
        ).astype(int)
        self.features["breakdown_20d_low"] = (
            close <= self.df["low"].rolling(20).min()
        ).astype(int)
        return self

    def add_all(self) -> "TechnicalFeatures":
        return (
            self.add_moving_averages()
            .add_momentum()
            .add_volatility()
            .add_drawdown()
            .add_trend_signals()
        )

    def get_features(self) -> pd.DataFrame:
        return self.features

    def get_latest(self) -> dict:
        return self.features.iloc[-1].to_dict()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """便捷函数：一次性算出全部技术因子。"""
    return TechnicalFeatures(df).add_all().get_features()
