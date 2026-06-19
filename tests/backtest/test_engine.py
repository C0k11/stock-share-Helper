"""quantai.backtest.engine 测试 —— 重点：证明 lookahead 会把收益做虚高。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantai.backtest.engine import run_backtest


def test_lookahead_close_mode_inflates_returns(prices: pd.DataFrame) -> None:
    """偷看信号(只在上涨日做多)在 close 模式下虚高，在 next_open 模式下不再。"""
    cc = prices["close"].pct_change(fill_method=None)
    peeking_weight = (cc > 0).astype(float)  # 只有 close[t] 出来才知道 -> 偷看未来

    old = run_backtest(prices, peeking_weight, fill_timing="close", cost_per_turnover=0.0)
    new = run_backtest(prices, peeking_weight, fill_timing="next_open", cost_per_turnover=0.0)

    # 偷看信号在旧模式下吃下所有上涨日 -> 巨幅虚高
    assert old.metrics.total_return > 1.0
    # 修复后远低于旧版（lookahead 红利消失）
    assert old.metrics.total_return > new.metrics.total_return * 3


def test_buy_and_hold_modes_agree(prices: pd.DataFrame) -> None:
    """恒定满仓(无换手)时，两种成交时点的日收益应一致（预热 2 天除外）。"""
    w = pd.Series(1.0, index=prices.index)
    old = run_backtest(prices, w, fill_timing="close", cost_per_turnover=0.0)
    new = run_backtest(prices, w, fill_timing="next_open", cost_per_turnover=0.0)
    np.testing.assert_allclose(old.returns.iloc[2:], new.returns.iloc[2:], atol=1e-12)


def test_costs_reduce_total_return(prices: pd.DataFrame) -> None:
    cc = prices["close"].pct_change(fill_method=None)
    alternating = (cc.rolling(2).mean() > 0).astype(float)  # 会频繁换手
    free = run_backtest(prices, alternating, fill_timing="next_open", cost_per_turnover=0.0)
    costly = run_backtest(prices, alternating, fill_timing="next_open", cost_per_turnover=0.001)
    assert costly.metrics.total_return < free.metrics.total_return
    assert costly.total_turnover > 0


def test_invalid_fill_timing_raises(prices: pd.DataFrame) -> None:
    w = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError):
        run_backtest(prices, w, fill_timing="midnight")


def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.bdate_range("2021-01-01", periods=2))
    with pytest.raises(ValueError):
        run_backtest(df, pd.Series(1.0, index=df.index))
