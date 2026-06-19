"""quantai.features.technical 的口径与工程质量测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from quantai.features.technical import TechnicalFeatures, compute_technical_features


def test_add_all_produces_expected_columns(prices: pd.DataFrame) -> None:
    feats = compute_technical_features(prices)
    for col in (
        "ma_20", "ma_200", "ma_20_slope", "price_vs_ma20",
        "return_63d", "momentum_3_12m",
        "volatility_20d", "vol_ratio",
        "drawdown", "max_drawdown_60d",
        "trend_alignment", "breakout_20d_high", "breakdown_20d_low",
    ):
        assert col in feats.columns


def test_ma_is_trailing_mean(prices: pd.DataFrame) -> None:
    feats = TechnicalFeatures(prices).add_moving_averages().get_features()
    i = 50
    expected = prices["close"].iloc[i - 19 : i + 1].mean()
    assert feats["ma_20"].iloc[i] == pytest.approx(expected)


def test_momentum_matches_pct_change(prices: pd.DataFrame) -> None:
    feats = TechnicalFeatures(prices).add_momentum().get_features()
    expected = prices["close"].iloc[100] / prices["close"].iloc[95] - 1
    assert feats["return_5d"].iloc[100] == pytest.approx(expected)


def test_drawdown_is_non_positive(prices: pd.DataFrame) -> None:
    feats = TechnicalFeatures(prices).add_drawdown().get_features()
    assert (feats["drawdown"].dropna() <= 1e-9).all()


def test_breakout_flags_are_binary(prices: pd.DataFrame) -> None:
    feats = TechnicalFeatures(prices).add_trend_signals().get_features()
    assert set(feats["breakout_20d_high"].unique()) <= {0, 1}


def test_custom_windows_do_not_leak_defaults(make_prices) -> None:
    """显式传 windows 应只产出该窗口，不混入默认窗口（可变默认参数已修）。"""
    p = make_prices(120, 1)
    feats = TechnicalFeatures(p).add_moving_averages([10]).get_features()
    assert "ma_10" in feats.columns
    assert "ma_200" not in feats.columns


def test_get_latest_returns_last_row(prices: pd.DataFrame) -> None:
    tf = TechnicalFeatures(prices).add_moving_averages()
    latest = tf.get_latest()
    assert latest["ma_20"] == pytest.approx(tf.get_features()["ma_20"].iloc[-1], nan_ok=True)
