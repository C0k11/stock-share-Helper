"""quantai.backtest.compare 测试。"""

from __future__ import annotations

import pandas as pd

from quantai.backtest.compare import compare_fill_timings, format_comparison_markdown


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
