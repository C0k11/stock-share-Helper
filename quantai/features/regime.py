"""市场状态(regime)检测：Risk-On / Risk-Off / Transition。

从旧 `src/features/regime.py` 迁移，口径不变，加类型标注。
默认阈值与 `quantai.config` 的 `strategy.regime` 对应（vix_high=25 / vix_low=15 / trend_ma=50）。

无 lookahead：趋势/波动率用因果 rolling；
VIX 对齐用 `reindex(..., method="ffill")`（前向填充，只用过去值）。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

_ANNUALIZE = np.sqrt(252)


class MarketRegime(Enum):
    """市场风险状态。"""

    RISK_ON = "risk_on"        # 风险偏好，进攻
    RISK_OFF = "risk_off"      # 风险规避，防守
    TRANSITION = "transition"  # 过渡期，谨慎


class RegimeDetector:
    """基于 趋势 + VIX + 波动率扩张 的三票打分，映射到 regime。"""

    def __init__(
        self,
        vix_high: float = 25.0,
        vix_low: float = 15.0,
        trend_ma: int = 50,
    ) -> None:
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.trend_ma = trend_ma

    def detect(
        self,
        spy_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """逐日打分并映射 regime，返回含 trend_signal/vix_signal/vol_expansion/score/regime 的表。"""
        result = pd.DataFrame(index=spy_data.index)
        close = spy_data["close"]

        # 1. 趋势：价格 vs 均线
        ma = close.rolling(self.trend_ma).mean()
        result["trend_signal"] = (close > ma).astype(int)

        # 2. VIX：低于 low -> Risk-On(+1)，高于 high -> Risk-Off(-1)
        if vix_data is not None and "close" in vix_data.columns:
            vix_aligned = vix_data["close"].reindex(spy_data.index, method="ffill")
            result["vix"] = vix_aligned
            result["vix_signal"] = 0
            result.loc[vix_aligned < self.vix_low, "vix_signal"] = 1
            result.loc[vix_aligned > self.vix_high, "vix_signal"] = -1
        else:
            result["vix_signal"] = 0

        # 3. 波动率扩张：短期年化波动 > 长期 *1.2 -> 防守(-1)
        returns = close.pct_change(fill_method=None)
        vol_short = returns.rolling(10).std() * _ANNUALIZE
        vol_long = returns.rolling(60).std() * _ANNUALIZE
        result["vol_expansion"] = (vol_short > vol_long * 1.2).astype(int) * -1

        # 4. 综合打分 -> regime
        result["score"] = (
            result["trend_signal"] + result["vix_signal"] + result["vol_expansion"]
        )
        result["regime"] = MarketRegime.TRANSITION.value
        result.loc[result["score"] >= 1, "regime"] = MarketRegime.RISK_ON.value
        result.loc[result["score"] <= -1, "regime"] = MarketRegime.RISK_OFF.value
        return result

    def get_current_regime(
        self,
        spy_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
    ) -> dict:
        """返回最新一日的 regime 快照。"""
        result = self.detect(spy_data, vix_data)
        latest = result.iloc[-1]
        return {
            "regime": latest["regime"],
            "score": latest["score"],
            "trend_signal": latest["trend_signal"],
            "vix_signal": latest["vix_signal"],
            "vol_expansion": latest["vol_expansion"],
            "vix": latest.get("vix", None),
            "date": str(result.index[-1]),
        }

    def get_regime_history(
        self,
        spy_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """只取 regime + score 两列的历史序列。"""
        return self.detect(spy_data, vix_data)[["regime", "score"]]


def detect_regime(
    spy_data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None,
) -> str:
    """便捷函数：返回最新一日的 regime 字符串。"""
    return RegimeDetector().get_current_regime(spy_data, vix_data)["regime"]
