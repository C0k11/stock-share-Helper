"""quantai.execution.state 测试。"""

from __future__ import annotations

from quantai.execution.state import TickerExecutionState


def test_min_hold_days_blocks_early_reverse() -> None:
    st = TickerExecutionState(min_hold_days=3)
    st.update_signal(1)            # 建多
    assert st.current_position == 1
    # 最短持有期内即便反向也不动
    assert st.update_signal(-1) == 1
    assert st.update_signal(-1) == 1


def test_reverse_needs_confirm_days() -> None:
    st = TickerExecutionState(min_hold_days=0, reverse_confirm_days=2)
    st.update_signal(1)
    assert st.update_signal(-1) == 1   # 第 1 次反向，未确认
    assert st.update_signal(-1) == -1  # 第 2 次确认 -> 翻空


def test_force_flat_resets() -> None:
    st = TickerExecutionState()
    st.update_signal(1)
    assert st.update_signal(0, force_flat=True) == 0
    assert st.current_position == 0


def test_keep_policy_holds_on_zero_signal() -> None:
    st = TickerExecutionState(hold_policy="keep", min_hold_days=0)
    st.update_signal(1)
    assert st.update_signal(0) == 1   # keep -> 维持多头


def test_exit_policy_flattens_on_zero_signal() -> None:
    st = TickerExecutionState(hold_policy="exit", min_hold_days=0)
    st.update_signal(1)
    assert st.update_signal(0) == 0


def test_dict_roundtrip() -> None:
    st = TickerExecutionState(min_hold_days=2, reverse_confirm_days=3)
    st.update_signal(1)
    restored = TickerExecutionState.from_dict(st.to_dict())
    assert restored.current_position == st.current_position
    assert restored.min_hold_days == 2
    assert restored.reverse_confirm_days == 3
