"""analysis/ 无 lookahead 证明：截断不变性（与 features 层同一方法论）。

若指标在 t 处的值只依赖 <= t 的数据，把序列在 t 后截断重算，t 处的值必须完全一致。
覆盖 trend / volatility / stats 全部产 Series/DataFrame 的指标。
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantai.analysis import (
    atr,
    bollinger,
    drawdown_curve,
    macd,
    macd_cross,
    moving_average_system,
    pullback,
    realized_volatility,
    rolling_beta,
    rolling_correlation,
    rsi,
    stochastic,
)

_CUTS = (150, 220, 280)


def _assert_causal_frame(full: pd.DataFrame, recompute, prices: pd.DataFrame) -> None:
    for k in _CUTS:
        truncated = recompute(prices.iloc[:k])
        idx = prices.index[k - 1]
        pd.testing.assert_series_equal(full.loc[idx], truncated.loc[idx])


@pytest.mark.parametrize(
    "fn",
    [
        lambda df: macd(df["close"]),
        lambda df: macd_cross(df["close"]),
        lambda df: moving_average_system(df["close"]),
        lambda df: pullback(df["close"], df["high"]),
        lambda df: bollinger(df["close"]),
        lambda df: stochastic(df["high"], df["low"], df["close"]),
    ],
    ids=["macd", "macd_cross", "ma_system", "pullback", "bollinger", "stochastic"],
)
def test_frame_indicators_are_causal(prices: pd.DataFrame, fn) -> None:
    _assert_causal_frame(fn(prices), fn, prices)


@pytest.mark.parametrize(
    "fn",
    [
        lambda df: rsi(df["close"]),
        lambda df: realized_volatility(df["close"]),
        lambda df: atr(df["high"], df["low"], df["close"]),
        lambda df: drawdown_curve(df["close"]),
        lambda df: rolling_correlation(df["close"], df["open"]),
        lambda df: rolling_beta(
            df["close"].pct_change(fill_method=None),
            df["open"].pct_change(fill_method=None),
        ),
    ],
    ids=["rsi", "realized_vol", "atr", "drawdown", "rolling_corr", "rolling_beta"],
)
def test_series_indicators_are_causal(prices: pd.DataFrame, fn) -> None:
    full = fn(prices)
    for k in _CUTS:
        truncated = fn(prices.iloc[:k])
        idx = prices.index[k - 1]
        f, t = full.loc[idx], truncated.loc[idx]
        assert (f == t) or (pd.isna(f) and pd.isna(t)), f"lookahead at {idx}: {f} != {t}"
