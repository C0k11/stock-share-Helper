"""波动率管理：实现波动率 / 波动率状态 / 仓位缩放 / 波动率预警。

从旧 `src/risk/volatility.py` 迁移，口径不变，加类型标注、显式 `fill_method=None`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZE = np.sqrt(252)


class VolatilityManager:
    """围绕目标波动率做仓位缩放与状态判断。"""

    def __init__(
        self,
        target_volatility: float = 0.10,
        vol_ceiling: float = 0.25,
        lookback_short: int = 20,
        lookback_long: int = 60,
    ) -> None:
        self.target_volatility = target_volatility
        self.vol_ceiling = vol_ceiling
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def compute_realized_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """年化实现波动率（trailing rolling std）。"""
        returns = prices.pct_change(fill_method=None)
        return returns.rolling(window).std() * _ANNUALIZE

    def compute_vol_regime(self, prices: pd.Series) -> pd.DataFrame:
        """短/长期波动率、比值、状态(normal/expanding/contracting/extreme)。"""
        result = pd.DataFrame(index=prices.index)
        result["vol_short"] = self.compute_realized_volatility(prices, self.lookback_short)
        result["vol_long"] = self.compute_realized_volatility(prices, self.lookback_long)
        result["vol_ratio"] = result["vol_short"] / result["vol_long"]
        result["vol_regime"] = "normal"
        result.loc[result["vol_ratio"] > 1.3, "vol_regime"] = "expanding"
        result.loc[result["vol_ratio"] < 0.7, "vol_regime"] = "contracting"
        result.loc[result["vol_short"] > self.vol_ceiling, "vol_regime"] = "extreme"
        return result

    def get_position_scalar(self, current_vol: float) -> float:
        """仓位缩放因子 = 目标波动率 / 当前波动率，截断到 [0.2, 1.5]。"""
        if current_vol <= 0:
            return 1.0
        return float(np.clip(self.target_volatility / current_vol, 0.2, 1.5))

    def check_vol_alert(self, prices: pd.Series) -> dict:
        """基于最新一日的波动率状态给出预警与仓位缩放建议。"""
        vol_data = self.compute_vol_regime(prices)
        current = vol_data.iloc[-1]
        alerts = []
        if current["vol_regime"] == "expanding":
            alerts.append(
                {
                    "type": "vol_expansion",
                    "severity": "medium",
                    "message": f"短期波动率({current['vol_short']:.1%})高于长期水平({current['vol_long']:.1%})",
                }
            )
        if current["vol_regime"] == "extreme":
            alerts.append(
                {
                    "type": "vol_extreme",
                    "severity": "high",
                    "message": f"波动率({current['vol_short']:.1%})超过上限({self.vol_ceiling:.1%})",
                }
            )
        return {
            "current_vol": float(current["vol_short"]),
            "vol_regime": current["vol_regime"],
            "position_scalar": self.get_position_scalar(current["vol_short"]),
            "alerts": alerts,
        }
