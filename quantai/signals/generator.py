"""规则信号生成：趋势 / 动量 / 均线交叉 / 突破 -> 综合信号与离散强度。

从旧 `src/strategy/signals.py` 迁移，口径不变，加类型标注、显式 `fill_method=None`。
无 lookahead：全部用因果窗口；信号在 t 处只反映 <= t 的信息，
真正的成交时点由 backtest 的 next_open 保证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SignalGenerator:
    """把价格序列转成方向信号(+1/-1)与离散强度标签。"""

    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 200,
        momentum_lookback: int = 63,
    ) -> None:
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.momentum_lookback = momentum_lookback

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成 trend/momentum/ma_cross/breakout/composite 信号与 signal_strength。"""
        signals = pd.DataFrame(index=df.index)
        close = df["close"]

        ma_long = close.rolling(self.ma_long).mean()
        signals["trend_signal"] = np.where(close > ma_long, 1, -1)

        momentum = close.pct_change(self.momentum_lookback, fill_method=None)
        signals["momentum_signal"] = np.where(momentum > 0, 1, -1)

        ma_short = close.rolling(self.ma_short).mean()
        signals["ma_cross_signal"] = np.where(ma_short > ma_long, 1, -1)

        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        signals["breakout_signal"] = 0
        signals.loc[close >= high_20, "breakout_signal"] = 1
        signals.loc[close <= low_20, "breakout_signal"] = -1

        signals["composite_signal"] = (
            signals["trend_signal"] * 0.3
            + signals["momentum_signal"] * 0.3
            + signals["ma_cross_signal"] * 0.2
            + signals["breakout_signal"] * 0.2
        )

        signals["signal_strength"] = pd.cut(
            signals["composite_signal"],
            bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            labels=["strong_short", "weak_short", "neutral", "weak_long", "strong_long"],
        )
        return signals

    def get_current_signal(self, df: pd.DataFrame) -> dict:
        """最新一日的信号快照。"""
        signals = self.generate(df)
        latest = signals.iloc[-1]
        return {
            "date": str(signals.index[-1]),
            "trend": int(latest["trend_signal"]),
            "momentum": int(latest["momentum_signal"]),
            "ma_cross": int(latest["ma_cross_signal"]),
            "breakout": int(latest["breakout_signal"]),
            "composite": float(latest["composite_signal"]),
            "strength": str(latest["signal_strength"]),
        }
