"""charts.py 单测：纯函数产 Figure，离线断言 trace 结构与暗色主题。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from quantai.ui.charts import (
    DARK_LAYOUT,
    DOWN_COLOR,
    UP_COLOR,
    candlestick_figure,
    pnl_bar_figure,
    rsi_macd_figure,
)


def _ohlcv(n=120, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))),
        index=pd.bdate_range("2025-01-06", periods=n),
    )
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.integers(1e6, 5e6, n)}
    )


class TestCandlestick:
    def test_traces_structure(self):
        fig = candlestick_figure(_ohlcv(), "AAA", ma_windows=(20, 50), show_bollinger=True)
        kinds = [t.type for t in fig.data]
        assert kinds.count("candlestick") == 1
        assert kinds.count("scatter") == 4  # MA20 + MA50 + BB 上下轨
        assert kinds.count("bar") == 1  # 成交量
        names = {t.name for t in fig.data}
        assert {"MA20", "MA50"} <= names

    def test_no_volume_column_no_bar(self):
        df = _ohlcv().drop(columns=["volume"])
        fig = candlestick_figure(df, "AAA", show_bollinger=False)
        assert all(t.type != "bar" for t in fig.data)

    def test_dark_theme_applied(self):
        fig = candlestick_figure(_ohlcv(), "AAA")
        assert fig.layout.paper_bgcolor == DARK_LAYOUT["paper_bgcolor"]
        assert fig.layout.xaxis.rangeslider.visible is False

    def test_up_down_colors(self):
        fig = candlestick_figure(_ohlcv(), "AAA")
        candle = fig.data[0]
        assert candle.increasing.line.color == UP_COLOR
        assert candle.decreasing.line.color == DOWN_COLOR


class TestRsiMacd:
    def test_traces(self):
        fig = rsi_macd_figure(_ohlcv())
        kinds = [t.type for t in fig.data]
        assert kinds.count("scatter") == 3  # RSI + MACD + Signal
        assert kinds.count("bar") == 1  # Hist
        # RSI 超买/超卖参考线（hline 落进 layout.shapes）
        assert len(fig.layout.shapes) >= 2

    def test_short_series_does_not_crash(self):
        fig = rsi_macd_figure(_ohlcv(10))  # 全 warmup：NaN 曲线也要能画
        assert len(fig.data) == 4


class TestPnlBar:
    def test_colors_by_sign_and_sorting(self):
        fig = pnl_bar_figure(
            [
                {"symbol": "WIN", "unrealized_pnl": 500.0},
                {"symbol": "LOSE", "unrealized_pnl": -300.0},
            ]
        )
        bar = fig.data[0]
        assert list(bar.y) == ["LOSE", "WIN"]  # 按盈亏升序
        assert list(bar.marker.color) == [DOWN_COLOR, UP_COLOR]

    def test_nan_pnl_rendered_na(self):
        fig = pnl_bar_figure([{"symbol": "X", "unrealized_pnl": float("nan")}])
        assert fig.data[0].text[0] == "n/a"

    def test_empty_positions(self):
        fig = pnl_bar_figure([])
        assert len(fig.data[0].y) == 0
