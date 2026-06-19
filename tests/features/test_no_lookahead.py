"""无 lookahead 证明：截断不变性（truncation invariance）。

核心思想：若某特征在 t 处的值**只依赖 <= t 的数据**，那么把序列在 t 之后截断、
重新计算，t 处的值必须**完全一致**。若存在偷看未来（如 centered window / shift(-n)），
截断会改变 t 处的值，测试即失败。
"""

from __future__ import annotations

import pandas as pd

from quantai.features.regime import RegimeDetector
from quantai.features.technical import compute_technical_features


def test_technical_features_are_causal(prices: pd.DataFrame) -> None:
    full = compute_technical_features(prices)
    for k in (150, 220, 280):
        truncated = compute_technical_features(prices.iloc[:k])
        idx = prices.index[k - 1]
        pd.testing.assert_series_equal(full.loc[idx], truncated.loc[idx])


def test_regime_is_causal(prices: pd.DataFrame) -> None:
    vix = pd.DataFrame(
        {"close": (prices["close"] / prices["close"].iloc[0] * 15.0).to_numpy()},
        index=prices.index,
    )
    detector = RegimeDetector()
    full = detector.detect(prices, vix)
    for k in (150, 220, 280):
        truncated = detector.detect(prices.iloc[:k], vix.iloc[:k])
        idx = prices.index[k - 1]
        pd.testing.assert_series_equal(full.loc[idx], truncated.loc[idx])
