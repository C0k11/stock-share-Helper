"""quantai.risk.gate 测试。"""

from __future__ import annotations

from quantai.risk.gate import RiskGate


def test_drawdown_breach_forces_clear() -> None:
    gate = RiskGate(max_drawdown_limit_pct=-8.0)
    action, pos, trace = gate.adjudicate({"drawdown_20d_pct": -9.0}, None, "BUY", 0.5)
    assert action == "CLEAR" and pos == 0.0
    assert any("FORCE CLEAR" in t for t in trace)


def test_position_capped_to_limit() -> None:
    gate = RiskGate(max_pos_limit=0.5)
    action, pos, _ = gate.adjudicate({"drawdown_20d_pct": 0.0}, None, "BUY", 0.9)
    assert pos == 0.5


def test_panic_event_forces_reduce() -> None:
    gate = RiskGate()
    news = [{"event_type": "war_breakout", "impact_equity": -0.5}]
    action, pos, _ = gate.adjudicate({"drawdown_20d_pct": 0.0}, news, "BUY", 0.5)
    assert action == "REDUCE" and pos <= 0.1


def test_high_vol_scales_position_down() -> None:
    gate = RiskGate(vol_reduce_trigger_ann_pct=30.0, max_pos_limit=1.0)
    _, pos, trace = gate.adjudicate({"volatility_ann_pct": 50.0}, None, "BUY", 0.5)
    assert pos < 0.5
    assert any("High Vol" in t for t in trace)


def test_clean_proposal_approved() -> None:
    gate = RiskGate()
    action, pos, trace = gate.adjudicate({"drawdown_20d_pct": 0.0, "volatility_ann_pct": 10.0}, None, "BUY", 0.3)
    assert action == "BUY" and pos == 0.3
    assert any("Approved" in t for t in trace)
