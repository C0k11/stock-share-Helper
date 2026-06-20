"""quantai.backtest.compare 测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from quantai.backtest.compare import buy_and_hold, compare_fill_timings, format_comparison_markdown


def test_buy_and_hold_holds_full_position(prices: pd.DataFrame) -> None:
    bh = buy_and_hold(prices)
    assert bh.fill_timing == "next_open"
    # 满仓持有到底：换手只有首日建仓那一次。
    assert bh.total_turnover == pytest.approx(1.0)


def test_markdown_with_benchmark_has_bh_and_gap_columns(prices: pd.DataFrame) -> None:
    weight = pd.Series(1.0, index=prices.index)
    results = compare_fill_timings(prices, weight)
    md = format_comparison_markdown(results, benchmark=buy_and_hold(prices), title="TEST")
    assert "Buy&Hold SPY" in md
    assert "新版 − B&H" in md
    assert "Sharpe 口径" in md  # 脚注：写清 rf 与算法
    assert "rf=2.0%" in md
    assert "pp" in md  # 差距列用百分点


def test_compare_returns_both_modes(prices: pd.DataFrame) -> None:
    cc = prices["close"].pct_change(fill_method=None)
    weight = (cc.rolling(3).mean() > 0).astype(float)
    results = compare_fill_timings(prices, weight, cost_per_turnover=0.0005)
    assert set(results) == {"close", "next_open"}
    assert results["close"].fill_timing == "close"
    assert results["next_open"].fill_timing == "next_open"


def test_markdown_table_has_all_metric_rows(prices: pd.DataFrame) -> None:
    weight = pd.Series(1.0, index=prices.index)
    md = format_comparison_markdown(compare_fill_timings(prices, weight), title="TEST")
    for label in ("总收益", "年化(CAGR)", "年化波动", "Sharpe", "最大回撤", "胜率"):
        assert label in md
    assert "TEST" in md
