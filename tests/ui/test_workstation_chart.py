"""workstation_figure 单测：面板随开关增减、叠加层、VWAP 日内重置、线/蜡烛切换。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from quantai.ui.charts import workstation_figure


def _daily(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))),
        index=pd.bdate_range("2025-01-06", periods=n),
    )
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.integers(1e6, 5e6, n)}
    )


def _intraday(bars_per_day=30, days=2) -> pd.DataFrame:
    idx = []
    for d in range(days):
        day = pd.Timestamp("2026-07-06") + pd.Timedelta(days=d)
        idx += [day + pd.Timedelta(minutes=5 * i) for i in range(bars_per_day)]
    n = len(idx)
    close = pd.Series(100 + np.sin(np.arange(n) / 5), index=pd.DatetimeIndex(idx))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": [1000.0] * n}
    )


class TestPanes:
    def test_all_panes_on(self):
        fig = workstation_figure(_daily(), "SPY", ma_windows=(20,), show_volume=True,
                                 show_rsi=True, show_macd=True)
        # 4 窗格：subplot 布局有 yaxis..yaxis4
        assert fig.layout.yaxis4 is not None
        kinds = [t.type for t in fig.data]
        assert "candlestick" in kinds and kinds.count("bar") == 2  # volume + macd hist

    def test_price_only(self):
        fig = workstation_figure(_daily(), "SPY", show_volume=False, show_rsi=False, show_macd=False)
        assert len(fig.data) == 1
        assert not hasattr(fig.layout, "yaxis2") or fig.layout.yaxis2.domain is None

    def test_line_kind(self):
        fig = workstation_figure(_daily(), "SPY", kind="line", show_volume=False,
                                 show_rsi=False, show_macd=False)
        assert fig.data[0].type == "scatter"


class TestOverlays:
    def test_ma_and_bollinger_traces(self):
        fig = workstation_figure(_daily(), "SPY", ma_windows=(20, 50), show_bollinger=True,
                                 show_volume=False, show_rsi=False, show_macd=False)
        names = [t.name for t in fig.data]
        assert "MA20" in names and "MA50" in names
        assert names.count("bb_upper") + names.count("bb_lower") == 2

    def test_vwap_resets_each_session(self):
        df = _intraday(bars_per_day=30, days=2)
        fig = workstation_figure(df, "SPY", kind="line", show_vwap=True,
                                 show_volume=False, show_rsi=False, show_macd=False)
        vwap = next(t for t in fig.data if t.name == "VWAP")
        y = np.asarray(vwap.y, dtype=float)
        # 等量成交下 VWAP=典型价累计均值：第二天首根应重置为当根典型价（非跨日累计）
        day2_first = 30
        typical = (df["high"] + df["low"] + df["close"]) / 3
        assert y[day2_first] == pytest.approx(float(typical.iloc[day2_first]))

    def test_vwap_skipped_without_volume(self):
        df = _daily().drop(columns=["volume"])
        fig = workstation_figure(df, "SPY", show_vwap=True, show_volume=False,
                                 show_rsi=False, show_macd=False)
        assert all(t.name != "VWAP" for t in fig.data)
