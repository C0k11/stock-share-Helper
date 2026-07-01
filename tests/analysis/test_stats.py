"""stats.py 单测：口径复用（Sharpe 与 backtest.metrics 同源）、已知值手算、边界。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.analysis import (
    beta,
    drawdown_curve,
    drawdown_stats,
    return_distribution,
    returns_from_prices,
    rolling_beta,
    sharpe_ratio,
    sortino_ratio,
)
from quantai.backtest import metrics as bt_metrics


def _series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- #
# 口径一致性：分析层与回测层必须同一个 Sharpe/回撤
# --------------------------------------------------------------------------- #
class TestMetricsReuse:
    def test_sharpe_is_backtest_sharpe(self):
        assert sharpe_ratio is bt_metrics.sharpe_ratio

    def test_max_drawdown_is_backtest_max_drawdown(self):
        from quantai.analysis import max_drawdown

        assert max_drawdown is bt_metrics.max_drawdown


# --------------------------------------------------------------------------- #
# 收益分布
# --------------------------------------------------------------------------- #
class TestReturnDistribution:
    def test_known_small_sample(self):
        r = pd.Series([0.01, -0.02, 0.03])
        d = return_distribution(r)
        assert d.n_days == 3
        assert d.mean_daily == pytest.approx(np.mean([0.01, -0.02, 0.03]))
        assert d.std_daily == pytest.approx(np.std([0.01, -0.02, 0.03], ddof=1))
        assert d.ann_return == pytest.approx(d.mean_daily * 252)
        assert d.ann_volatility == pytest.approx(d.std_daily * np.sqrt(252))
        assert d.best_day == pytest.approx(0.03)
        assert d.worst_day == pytest.approx(-0.02)
        assert d.positive_share == pytest.approx(2 / 3)
        # 历史 VaR：pandas 线性插值分位
        assert d.var_95 == pytest.approx(r.quantile(0.05))
        assert d.cvar_95 == pytest.approx(-0.02)  # 尾部只有最差那天

    def test_empty_series(self):
        d = return_distribution(pd.Series([], dtype=float))
        assert d.n_days == 0
        assert np.isnan(d.mean_daily) and np.isnan(d.var_95)

    def test_nan_dropped(self):
        d = return_distribution(pd.Series([0.01, np.nan, -0.01]))
        assert d.n_days == 2

    def test_returns_from_prices(self):
        out = returns_from_prices(_series([100, 110, 99]))
        assert out.tolist() == pytest.approx([0.10, -0.10])


# --------------------------------------------------------------------------- #
# 回撤
# --------------------------------------------------------------------------- #
class TestDrawdown:
    def test_curve_known_values(self):
        eq = _series([100, 120, 90, 95, 130, 110])
        dd = drawdown_curve(eq)
        assert dd.iloc[0] == 0.0
        assert dd.iloc[2] == pytest.approx(90 / 120 - 1)  # -25%
        assert dd.iloc[4] == 0.0  # 创新高回到 0

    def test_stats_known_values(self):
        eq = _series([100, 120, 90, 95, 130, 110])
        st = drawdown_stats(eq)
        assert st.max_drawdown == pytest.approx(-0.25)
        assert st.peak_date == str(eq.index[1])
        assert st.trough_date == str(eq.index[2])
        assert st.longest_underwater_days == 2  # 90、95 两天水下（末段 110 单独 1 天）

    def test_monotonic_up_no_drawdown(self):
        st = drawdown_stats(_series([1, 2, 3, 4]))
        assert st.max_drawdown == pytest.approx(0.0)
        assert st.longest_underwater_days == 0

    def test_empty(self):
        st = drawdown_stats(pd.Series([], dtype=float))
        assert np.isnan(st.max_drawdown)
        assert st.longest_underwater_days == 0


# --------------------------------------------------------------------------- #
# Sortino
# --------------------------------------------------------------------------- #
class TestSortino:
    def test_known_value_rf0(self):
        r = pd.Series([0.02, -0.01])
        # downside_dev = sqrt(mean([0, -0.01]^2)) * sqrt(252)；mean_excess = 0.005
        expected = (0.005 * 252) / (np.sqrt(np.mean([0.0, 0.0001])) * np.sqrt(252))
        assert sortino_ratio(r, risk_free_rate=0.0) == pytest.approx(expected)

    def test_no_downside_is_nan(self):
        r = pd.Series([0.01, 0.02, 0.03])
        assert np.isnan(sortino_ratio(r, risk_free_rate=0.0))

    def test_empty_is_nan(self):
        assert np.isnan(sortino_ratio(pd.Series([], dtype=float)))

    def test_penalizes_downside_more_than_sharpe_direction(self):
        # 左偏（偶发大跌）序列的 Sortino 应低于右偏（偶发大涨）同均值序列
        rng = np.random.default_rng(7)
        base = rng.normal(0.001, 0.005, 500)
        left = pd.Series(np.where(rng.random(500) < 0.02, -0.05, base))
        right = pd.Series(np.where(rng.random(500) < 0.02, +0.05, base))
        assert sortino_ratio(left, 0.0) < sortino_ratio(right, 0.0)


# --------------------------------------------------------------------------- #
# beta
# --------------------------------------------------------------------------- #
class TestBeta:
    def test_scaled_beta_2(self):
        rng = np.random.default_rng(3)
        rb = pd.Series(rng.normal(0, 0.01, 200), index=pd.bdate_range("2023-01-01", periods=200))
        assert beta(rb * 2.0, rb) == pytest.approx(2.0)

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(4)
        idx = pd.bdate_range("2023-01-01", periods=2000)
        ra = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
        rb = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
        assert abs(beta(ra, rb)) < 0.1

    def test_constant_benchmark_nan(self):
        idx = pd.bdate_range("2023-01-01", periods=50)
        ra = pd.Series(np.random.default_rng(5).normal(0, 0.01, 50), index=idx)
        rb = pd.Series(0.0, index=idx)
        assert np.isnan(beta(ra, rb))

    def test_too_few_samples_nan(self):
        idx = pd.bdate_range("2023-01-01", periods=1)
        assert np.isnan(beta(pd.Series([0.01], index=idx), pd.Series([0.02], index=idx)))

    def test_alignment_inner_join(self):
        idx = pd.bdate_range("2023-01-01", periods=100)
        rng = np.random.default_rng(6)
        rb = pd.Series(rng.normal(0, 0.01, 100), index=idx)
        ra = (rb * 1.5).iloc[30:]  # 只有部分重叠
        assert beta(ra, rb) == pytest.approx(1.5)

    def test_rolling_beta_scaled(self):
        rng = np.random.default_rng(8)
        idx = pd.bdate_range("2023-01-01", periods=150)
        rb = pd.Series(rng.normal(0, 0.01, 150), index=idx)
        out = rolling_beta(rb * 2.0, rb, window=30).dropna()
        assert (out - 2.0).abs().max() < 1e-9

    def test_rolling_beta_window_raises(self):
        s = pd.Series([0.01, 0.02])
        with pytest.raises(ValueError):
            rolling_beta(s, s, window=1)
