"""trend.py 单测：已知值手算、方向性行为、边界（数据不足 / NaN / 常数序列）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.analysis import (
    ema,
    macd,
    macd_cross,
    momentum,
    moving_average_system,
    pullback,
    roc,
    rsi,
    rsi_signals,
    sma,
    stochastic,
)


def _series(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --------------------------------------------------------------------------- #
# 均线
# --------------------------------------------------------------------------- #
class TestMovingAverages:
    def test_sma_known_values(self):
        s = _series([1, 2, 3, 4])
        out = sma(s, 2)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1:].tolist() == [1.5, 2.5, 3.5]

    def test_ema_known_recursion(self):
        # span=3 -> alpha=0.5：EMA = [2, 3, 5.5]
        s = _series([2, 4, 8])
        out = ema(s, 3)
        assert out.tolist() == pytest.approx([2.0, 3.0, 5.5])

    def test_ma_system_alignment_uptrend(self):
        s = _series(np.linspace(100, 200, 120))
        out = moving_average_system(s, windows=(5, 10, 20, 50))
        # 单调上涨 warmup 后短均线必在长均线上方 -> 完美多头排列
        assert out["ma_alignment"].iloc[-1] == 1.0
        # 最长窗口未热身处 alignment 必须 NaN（不装懂）
        assert np.isnan(out["ma_alignment"].iloc[10])

    def test_ma_system_alignment_downtrend(self):
        s = _series(np.linspace(200, 100, 120))
        out = moving_average_system(s, windows=(5, 20, 50))
        assert out["ma_alignment"].iloc[-1] == 0.0

    def test_ma_system_empty_windows_raises(self):
        with pytest.raises(ValueError):
            moving_average_system(_series([1, 2, 3]), windows=())

    def test_insufficient_data_all_nan(self):
        out = sma(_series([1, 2, 3]), 10)
        assert out.isna().all()


# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #
class TestMACD:
    def test_constant_price_macd_zero(self):
        s = _series([50.0] * 60)
        out = macd(s)
        assert (out["macd"].abs() < 1e-12).all()
        assert (out["macd_histogram"].abs() < 1e-12).all()

    def test_matches_ema_definition(self):
        s = _series(np.linspace(100, 130, 80) + np.sin(np.arange(80)))
        out = macd(s, fast=12, slow=26, signal=9)
        expected = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        pd.testing.assert_series_equal(out["macd"], expected, check_names=False)

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValueError):
            macd(_series([1, 2, 3]), fast=26, slow=12)

    def test_up_down_up_produces_death_then_golden_cross(self):
        # warmup(前 slow=26 根被抑制)之后：上升->下跌段出死叉，再反转出金叉
        up1 = np.linspace(100, 130, 50)
        down = np.linspace(130, 100, 40)
        up2 = np.linspace(100, 140, 50)
        s = _series(np.concatenate([up1, down[1:], up2[1:]]))
        out = macd_cross(s)
        assert out["death_cross"].iloc[50:90].any()  # 下跌段出死叉
        assert out["golden_cross"].iloc[90:].any()  # 反转段出金叉
        # 金叉死叉不能同一天同时为 True
        assert not (out["golden_cross"] & out["death_cross"]).any()

    def test_warmup_period_suppressed(self):
        # 审计实锤的 bar-1 假信号：EMA 种子期 macd~signal，bar 1 任何
        # 变动都"穿越"。前 slow 根必须恒 False——用 50 个随机序列扫。
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = _series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 60))))
            out = macd_cross(s)
            assert not out.iloc[:26].any().any(), "warmup 期不得产信号"

    def test_first_row_no_cross(self):
        out = macd_cross(_series(np.linspace(1, 2, 50)))
        assert not out.iloc[0].any()


# --------------------------------------------------------------------------- #
# 动量
# --------------------------------------------------------------------------- #
class TestMomentum:
    def test_roc_known_value(self):
        out = roc(_series([100, 110]), window=1)
        assert out.iloc[1] == pytest.approx(10.0)

    def test_momentum_known_value(self):
        out = momentum(_series([100, 110, 95]), window=2)
        assert out.iloc[2] == pytest.approx(-5.0)

    def test_short_series_nan(self):
        out = roc(_series([1, 2]), window=10)
        assert out.isna().all()


# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #
class TestRSI:
    def test_all_up_is_100(self):
        s = _series(np.linspace(100, 200, 60))
        out = rsi(s, 14)
        assert out.iloc[-1] == pytest.approx(100.0)

    def test_all_down_is_0(self):
        s = _series(np.linspace(200, 100, 60))
        out = rsi(s, 14)
        assert out.iloc[-1] == pytest.approx(0.0)

    def test_flat_is_neutral_50(self):
        s = _series([80.0] * 40)
        out = rsi(s, 14)
        assert out.iloc[-1] == pytest.approx(50.0)

    def test_range_bounds(self, prices):
        out = rsi(prices["close"], 14).dropna()
        assert ((out >= 0) & (out <= 100)).all()

    def test_warmup_is_nan(self):
        out = rsi(_series(np.linspace(1, 2, 30)), 14)
        assert out.iloc[:14].isna().all()
        assert out.iloc[14:].notna().all()

    def test_signals_thresholds(self):
        vals = pd.Series([np.nan, 25.0, 50.0, 75.0])
        out = rsi_signals(vals, oversold=30, overbought=70)
        assert out["rsi_oversold"].tolist() == [False, True, False, False]
        assert out["rsi_overbought"].tolist() == [False, False, False, True]

    def test_signals_bad_thresholds_raise(self):
        with pytest.raises(ValueError):
            rsi_signals(pd.Series([50.0]), oversold=70, overbought=30)


# --------------------------------------------------------------------------- #
# 随机指标
# --------------------------------------------------------------------------- #
class TestStochastic:
    def test_close_at_high_gives_100(self):
        n = 30
        close = _series(np.linspace(100, 130, n))  # 单调涨：收盘即区间最高
        out = stochastic(close, close, close, k_window=14, d_window=1, smooth_k=1)
        assert out["stoch_k"].iloc[-1] == pytest.approx(100.0)

    def test_close_at_low_gives_0(self):
        close = _series(np.linspace(130, 100, 30))
        out = stochastic(close, close, close, k_window=14, d_window=1, smooth_k=1)
        assert out["stoch_k"].iloc[-1] == pytest.approx(0.0)

    def test_zero_range_gives_nan(self):
        s = _series([100.0] * 30)
        out = stochastic(s, s, s, k_window=14)
        assert out["stoch_k"].iloc[-1] != out["stoch_k"].iloc[-1]  # NaN

    def test_bounds(self, prices):
        out = stochastic(prices["high"], prices["low"], prices["close"]).dropna()
        assert ((out >= 0) & (out <= 100)).all().all()


# --------------------------------------------------------------------------- #
# 回踩检测
# --------------------------------------------------------------------------- #
class TestPullback:
    def _trend_with_dip(self) -> pd.DataFrame:
        # 100 天线性上涨 100->160，然后 5 天回落 5%，再涨
        up1 = np.linspace(100, 160, 100)
        dip = np.linspace(160, 152, 6)[1:]  # -5%
        up2 = np.linspace(152, 170, 20)[1:]
        close = np.concatenate([up1, dip, up2])
        df = pd.DataFrame(
            {"close": close, "high": close * 1.002},
            index=pd.bdate_range("2023-01-01", periods=len(close)),
        )
        return df

    def test_detects_dip_in_uptrend(self):
        df = self._trend_with_dip()
        out = pullback(df["close"], df["high"])
        assert out["pullback"].iloc[100:105].any()  # 回落段触发
        assert out["in_uptrend"].iloc[100:105].all()  # 趋势未破坏

    def test_no_pullback_in_downtrend(self):
        close = _series(np.linspace(200, 100, 120))
        out = pullback(close, close * 1.002)
        assert not out["pullback"].any()

    def test_no_pullback_without_retrace(self):
        close = _series(np.linspace(100, 200, 120))  # 一路涨不回头
        out = pullback(close, close)  # high=close：收盘价即当日高点
        assert not out["pullback"].any()

    def test_bad_retrace_params_raise(self):
        s = _series([1, 2, 3])
        with pytest.raises(ValueError):
            pullback(s, s, min_retrace=0.2, max_retrace=0.1)

    def test_warmup_is_false_not_nan(self):
        df = self._trend_with_dip()
        out = pullback(df["close"], df["high"])
        assert out["pullback"].dtype == bool
        assert not out["pullback"].iloc[:10].any()


# --------------------------------------------------------------------------- #
# NaN 洞口径（审计后收紧：洞上必须 NaN，不许穿透旧值"编数字"）
# --------------------------------------------------------------------------- #
class TestNaNSafety:
    def test_functions_survive_nan_holes(self, prices):
        close = prices["close"].copy()
        close.iloc[50:53] = np.nan
        # 不崩 + 索引保持
        for out in (
            sma(close, 20),
            ema(close, 12),
            rsi(close, 14),
            roc(close, 10),
            macd(close)["macd"],
        ):
            assert len(out) == len(close)

    def test_ewm_family_nan_at_holes(self, prices):
        """EWM 族（ema/macd/rsi）洞上输出 NaN--pandas ewm 默认穿透旧值，已显式掩掉。"""
        close = prices["close"].copy()
        close.iloc[50:53] = np.nan
        assert ema(close, 12).iloc[50:53].isna().all()
        m = macd(close)
        assert m.iloc[50:53].isna().all().all()
        # RSI：洞上 + 洞后首根（跨洞差分未知）都 NaN
        r = rsi(close, 14)
        assert r.iloc[50:54].isna().all()
        assert r.iloc[55:].notna().all()  # 洞后恢复递推

    def test_rsi_hole_does_not_freeze_at_stale_value(self):
        """审查场景：涨到顶 -> 洞 -> 崩盘。旧实现 RSI 永远停在 100；新实现洞区 NaN。"""
        up = np.linspace(100, 130, 30)
        s = _series(np.concatenate([up, [np.nan], [90.0] * 20]))
        r = rsi(s, 14)
        assert np.isnan(r.iloc[30])  # 洞上 NaN（旧实现 = 100.0 假数字）
        assert np.isnan(r.iloc[31])  # 洞后首根差分未知 -> NaN
        assert r.iloc[35:].notna().all()

    def test_macd_cross_no_signal_at_or_after_hole(self, prices):
        close = prices["close"].copy()
        close.iloc[100:103] = np.nan
        out = macd_cross(close)
        assert not out.iloc[100:104].any().any()  # 洞及洞后首根不产信号


# --------------------------------------------------------------------------- #
# 参数校验（window=0 家族：拒绝而非静默全 NaN / ZeroDivisionError）
# --------------------------------------------------------------------------- #
class TestParamValidation:
    def test_window_zero_or_negative_raises(self):
        s = _series(np.linspace(1, 2, 30))
        with pytest.raises(ValueError):
            sma(s, 0)
        with pytest.raises(ValueError):
            ema(s, 0)
        with pytest.raises(ValueError):
            rsi(s, 0)
        with pytest.raises(ValueError):
            roc(s, 0)
        with pytest.raises(ValueError):
            momentum(s, -5)  # 负窗口 = lookahead，必须拒绝
        with pytest.raises(ValueError):
            stochastic(s, s, s, k_window=0)
        with pytest.raises(ValueError):
            stochastic(s, s, s, smooth_k=0)
        with pytest.raises(ValueError):
            pullback(s, s, lookback=0)
        with pytest.raises(ValueError):
            moving_average_system(s, windows=(0, 5))

    def test_ma_alignment_column_always_present(self):
        s = _series(np.linspace(1, 2, 30))
        out = moving_average_system(s, windows=(5,))
        assert "ma_alignment" in out.columns  # 单窗口：列存在、全 NaN（schema 稳定）
        assert out["ma_alignment"].isna().all()
        out2 = moving_average_system(s, windows=(5, 5, 5))  # 去重后单窗口同理
        assert out2["ma_alignment"].isna().all()
