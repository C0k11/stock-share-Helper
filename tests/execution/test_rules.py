"""quantai.execution.rules 测试。"""

from __future__ import annotations

from quantai.execution.rules import Action, TradingRules


def test_stop_loss_triggers_clear() -> None:
    rules = TradingRules(stop_loss_pct=0.08)
    # 入场 100，现价 90 -> -10% < -8% -> 清仓
    action = rules.determine_action(
        current_position=0.5, target_position=0.5, current_price=90.0, entry_price=100.0
    )
    assert action == Action.CLEAR


def test_take_profit_triggers_reduce() -> None:
    rules = TradingRules(take_profit_pct=0.25)
    action = rules.determine_action(
        current_position=0.5, target_position=0.5, current_price=130.0, entry_price=100.0
    )
    assert action == Action.REDUCE


def test_buy_from_flat_to_target() -> None:
    rules = TradingRules(rebalance_threshold=0.05)
    action = rules.determine_action(current_position=0.0, target_position=0.5, current_price=100.0)
    assert action == Action.BUY


def test_hold_when_below_rebalance_threshold() -> None:
    rules = TradingRules(rebalance_threshold=0.05)
    action = rules.determine_action(current_position=0.50, target_position=0.52, current_price=100.0)
    assert action == Action.HOLD


def test_generate_signal_sets_stops_on_buy() -> None:
    rules = TradingRules(stop_loss_pct=0.08, take_profit_pct=0.25)
    sig = rules.generate_trade_signal("SPY", 0.0, 0.5, 100.0)
    assert sig.action == Action.BUY
    assert sig.stop_loss == 92.0
    assert sig.take_profit == 125.0
    assert sig.entry_price == 100.0
