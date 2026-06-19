"""quantai.execution.simulator 测试。"""

from __future__ import annotations

import pytest

from quantai.execution.simulator import ExecutionSimulator


def test_close_mode_buy_adds_slippage() -> None:
    sim = ExecutionSimulator(mode="close", slippage_bps=10.0)
    res = sim.execute_buy(close=100.0)
    assert res.filled is True
    assert res.price == pytest.approx(100.0 * 1.001)  # +10bps
    assert res.note == "moc_fill"


def test_close_mode_sell_subtracts_slippage() -> None:
    sim = ExecutionSimulator(mode="close", slippage_bps=10.0)
    res = sim.execute_sell(close=100.0)
    assert res.price == pytest.approx(100.0 * 0.999)


def test_passive_buy_misses_when_low_above_limit() -> None:
    sim = ExecutionSimulator(mode="passive", limit_threshold_bps=50.0)
    # 限价 = open*(1-0.005)=99.5；low=100 未触及 -> 错过
    res = sim.execute_buy(close=101.0, open=100.0, high=102.0, low=100.0)
    assert res.filled is False
    assert res.note == "missed_limit"
    assert sim.stats["missed_orders"] == 1.0


def test_passive_buy_fills_when_low_touches_limit() -> None:
    sim = ExecutionSimulator(mode="passive", limit_threshold_bps=50.0)
    res = sim.execute_buy(close=101.0, open=100.0, high=102.0, low=99.0)
    assert res.filled is True
    assert res.price == pytest.approx(99.5)


def test_execution_cost_rate_signs() -> None:
    # buy 高于 close 是正成本；sell 低于 close 也是正成本(取反)
    assert ExecutionSimulator.execution_cost_rate(side="buy", close=100.0, exec_price=101.0) == pytest.approx(0.01)
    assert ExecutionSimulator.execution_cost_rate(side="sell", close=100.0, exec_price=99.0) == pytest.approx(0.01)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        ExecutionSimulator(mode="weird").execute_buy(close=100.0)
