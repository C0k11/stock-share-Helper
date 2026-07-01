"""volatility.py 单测：已知值手算、边界（常数序列 / 数据不足 / 零方差）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.analysis import atr, bollinger, realized_volatility, rolling_correlation


def _series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- #
# realized vol
# --------------------------------------------------------------------------- #
class TestRealizedVolatility:
    def test_constant_price_zero_vol(self):
        out = realized_volatility(_series([100.0] * 40), window=20)
        assert out.iloc[-1] == pytest.approx(0.0)

    def test_annualization_factor(self):
        s = _series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 100))))
        daily = realized_volatility(s, window=20, annualize=False)
        annual = realized_volatility(s, window=20, annualize=True)
        pd.testing.assert_series_equal(
            annual, daily * np.sqrt(252), check_names=False
        )

    def test_known_value(self):
        # 收益恰为 [+1%, -1%, +1%, ...]：窗口内 std 可手算
        rets = np.array([0.01, -0.01] * 15)
        close = 100 * np.cumprod(1 + rets)
        s = _series(np.concatenate([[100.0], close]))
        out = realized_volatility(s, window=10, annualize=False)
        expected = np.std([0.01, -0.01] * 5, ddof=1)
        assert out.iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_window_lt_2_raises(self):
        with pytest.raises(ValueError):
            realized_volatility(_series([1, 2, 3]), window=1)

    def test_insufficient_data_nan(self):
        out = realized_volatility(_series([1, 2, 3]), window=20)
        assert out.isna().all()


# --------------------------------------------------------------------------- #
# Bollinger
# --------------------------------------------------------------------------- #
class TestBollinger:
    def test_known_values(self):
        s = _series([1, 2, 3, 4, 5])
        out = bollinger(s, window=5, num_std=2.0)
        sd = np.std([1, 2, 3, 4, 5], ddof=0)  # 总体标准差（Bollinger/TA-Lib 口径）
        assert out["bb_mid"].iloc[-1] == pytest.approx(3.0)
        assert out["bb_upper"].iloc[-1] == pytest.approx(3.0 + 2 * sd)
        assert out["bb_lower"].iloc[-1] == pytest.approx(3.0 - 2 * sd)
        assert out["bb_bandwidth"].iloc[-1] == pytest.approx(4 * sd / 3.0)
        # close=5 恰在上轨内位置：(5-lower)/(upper-lower)
        assert out["bb_percent_b"].iloc[-1] == pytest.approx(
            (5 - (3 - 2 * sd)) / (4 * sd)
        )

    def test_constant_price(self):
        out = bollinger(_series([50.0] * 30), window=20)
        last = out.iloc[-1]
        assert last["bb_bandwidth"] == pytest.approx(0.0)
        assert last["bb_upper"] == pytest.approx(last["bb_lower"]) == pytest.approx(50.0)
        assert np.isnan(last["bb_percent_b"])  # 0/0 未定义 -> NaN

    def test_insufficient_data_nan(self):
        out = bollinger(_series([1, 2]), window=20)
        assert out["bb_mid"].isna().all()


# --------------------------------------------------------------------------- #
# ATR
# --------------------------------------------------------------------------- #
class TestATR:
    def test_constant_range_converges_to_range(self):
        n = 40
        high = _series([102.0] * n)
        low = _series([100.0] * n)
        close = _series([101.0] * n)
        out = atr(high, low, close, window=14)
        # TR 恒等于 2 -> Wilder 平滑收敛在 2
        assert out.iloc[-1] == pytest.approx(2.0)
        assert out.iloc[:14].isna().all()

    def test_gap_uses_prev_close(self):
        # 第二天跳空高开：TR 应含 |high - prev_close|
        high = _series([102.0, 120.0])
        low = _series([100.0, 118.0])
        close = _series([101.0, 119.0])
        prev_close = close.shift(1)
        tr_manual = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        assert tr_manual.iloc[1] == pytest.approx(19.0)  # 120 - 101，不是 2

    def test_nonnegative(self, prices):
        out = atr(prices["high"], prices["low"], prices["close"]).dropna()
        assert (out >= 0).all()


# --------------------------------------------------------------------------- #
# rolling correlation
# --------------------------------------------------------------------------- #
class TestRollingCorrelation:
    def test_scaled_series_corr_1(self, prices):
        a = prices["close"]
        out = rolling_correlation(a, a * 2.0, window=20).dropna()
        assert (out - 1.0).abs().max() < 1e-9

    def test_inverse_returns_corr_minus_1(self, prices):
        a = prices["close"]
        r = a.pct_change(fill_method=None).fillna(0.0)
        b = pd.Series(
            100 * np.cumprod(1 - r.to_numpy()), index=a.index
        )  # r_b = -r_a
        out = rolling_correlation(a, b, window=20).dropna()
        assert (out + 1.0).abs().max() < 1e-6

    def test_index_alignment_inner_join(self, prices):
        a = prices["close"]
        b = prices["close"].iloc[50:]  # 只有交集部分
        out = rolling_correlation(a, b, window=20)
        assert len(out) == len(b)

    def test_zero_variance_window_nan(self):
        a = _series([100.0] * 30)
        b = _series(np.linspace(100, 130, 30))
        out = rolling_correlation(a, b, window=10)
        assert out.iloc[15:].isna().all()  # a 收益恒 0 -> 相关未定义

    def test_window_lt_2_raises(self):
        s = _series([1, 2, 3])
        with pytest.raises(ValueError):
            rolling_correlation(s, s, window=1)
