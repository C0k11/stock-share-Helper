"""quantai.execution.costs 测试。"""

from __future__ import annotations

import pytest

from quantai.config.schema import AppConfig
from quantai.execution.costs import TransactionCosts


def test_min_commission_floor_dominates_small_trade() -> None:
    tc = TransactionCosts(commission_rate=0.0005, min_commission=1.0, slippage_bps=0.0)
    # 100 美元 * 0.0005 = 0.05 < 1.0 -> 取最低佣金 1.0
    assert tc.cost(100.0) == pytest.approx(1.0)


def test_cost_includes_commission_and_slippage() -> None:
    tc = TransactionCosts(commission_rate=0.0005, min_commission=1.0, slippage_bps=5.0)
    # 100000 * 0.0005 = 50 (>1) ; 滑点 100000 * 5/10000 = 50 -> 100
    assert tc.cost(100_000.0) == pytest.approx(100.0)


def test_round_trip_is_double() -> None:
    tc = TransactionCosts()
    assert tc.round_trip(100_000.0) == pytest.approx(tc.cost(100_000.0) * 2)


def test_from_config_reads_backtest_costs() -> None:
    cfg = AppConfig()
    tc = TransactionCosts.from_config(cfg.backtest.costs)
    assert tc.commission_rate == cfg.backtest.costs.commission_rate
    assert tc.slippage_bps == cfg.backtest.costs.slippage_bps
