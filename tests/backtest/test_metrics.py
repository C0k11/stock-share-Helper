"""quantai.backtest.metrics 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.backtest.metrics import compute_metrics


def _equity(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_monotonic_equity_has_zero_drawdown_and_full_win_rate() -> None:
    eq = _equity(list(100000 * 1.001 ** np.arange(253)))
    m = compute_metrics(eq)
    assert m.max_drawdown == 0.0
    assert m.win_rate == 1.0
    assert m.total_return == np.float64(1.001**252 - 1)


def test_max_drawdown_value() -> None:
    eq = _equity([100, 120, 90, 100])  # 峰 120 -> 谷 90 = -25%
    assert compute_metrics(eq).max_drawdown == np.float64(-0.25)


def test_win_rate_half() -> None:
    eq = _equity([100, 101, 100, 101, 100])  # 涨跌涨跌 -> 50%
    assert compute_metrics(eq).win_rate == 0.5
