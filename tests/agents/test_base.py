"""quantai.agents.base 纯数据契约测试。"""

from __future__ import annotations

from quantai.agents.base import (
    Account,
    Action,
    AgentContext,
    ExpertDecision,
    FinalDecision,
    Position,
    Regime,
)


def test_action_normalize_valid_and_invalid():
    assert Action.normalize("buy") == "BUY"
    assert Action.normalize("  Sell ") == "SELL"
    assert Action.normalize("clear") == "CLEAR"
    assert Action.normalize("garbage") == "HOLD"
    assert Action.normalize(None) == "HOLD"
    assert Action.normalize("garbage", default="SELL") == "SELL"


def test_position_side():
    assert Position(shares=10).side == "LONG"
    assert Position(shares=-3).side == "SHORT"
    assert Position(shares=0).side == "FLAT"


def test_context_accessors_defensive():
    ctx = AgentContext(
        ticker="NVDA",
        features={
            "technical": {"close": 100.0, "return_5d": 2.0},
            "signal": {"news_score": 0.5},
            "volatility_ann_pct": 45.0,
        },
    )
    assert ctx.technical["close"] == 100.0
    assert ctx.signal["news_score"] == 0.5
    assert ctx.volatility_ann_pct == 45.0


def test_context_accessors_missing_keys_return_empty():
    ctx = AgentContext(ticker="AAPL")
    assert ctx.technical == {}
    assert ctx.signal == {}
    assert ctx.volatility_ann_pct == 0.0


def test_context_volatility_bad_value():
    ctx = AgentContext(ticker="X", features={"volatility_ann_pct": "oops"})
    assert ctx.volatility_ann_pct == 0.0


def test_expert_decision_to_dict():
    d = ExpertDecision(decision="BUY", analysis="momentum", expert="scalper")
    out = d.to_dict()
    assert out == {
        "decision": "BUY",
        "analysis": "momentum",
        "expert": "scalper",
        "meta": {},
    }


def test_final_decision_to_dict_defaults():
    f = FinalDecision()
    out = f.to_dict()
    assert out["action"] == "HOLD"
    assert out["approved"] is False
    assert out["macro_label"] == "NEUTRAL"
    assert out["chart_score"] == 0


def test_regime_constants():
    assert Regime.AGGRESSIVE == "aggressive"
    assert Regime.DEFENSIVE == "defensive"
    assert Regime.CASH_PRESERVATION == "cash_preservation"


def test_account_defaults():
    a = Account()
    assert a.cash == 0.0 and a.equity == 0.0 and a.leverage == 0.0
