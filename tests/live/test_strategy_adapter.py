"""quantai.live.strategy_adapter.AgentStrategy 测试（用假 orchestrator + 真 PaperBroker）。"""

from __future__ import annotations

from quantai.agents.base import FinalDecision
from quantai.execution.broker import PaperBroker, Position
from quantai.live.strategy_adapter import AgentStrategy


class FakeOrch:
    def __init__(self, decision: FinalDecision):
        self.decision = decision
        self.calls = []

    def decide(self, ctx, llm=None):
        self.calls.append(ctx)
        return self.decision


def _feed(strat, *, ticker="NVDA", n=35, start=100.0, step=1.0):
    out = None
    for i in range(n):
        px = start + i * step
        out = strat.on_bar(
            {"ticker": ticker, "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1000}
        )
    return out


def test_warmup_returns_none():
    orch = FakeOrch(FinalDecision(action="BUY", approved=True))
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), min_history=30)
    out = _feed(strat, n=10)
    assert out is None
    assert orch.calls == []  # 预热期不调大脑


def test_approved_buy_returns_signal():
    orch = FakeOrch(
        FinalDecision(action="BUY", approved=True, expert="scalper", reason="bull", chart_score=1, regime="aggressive")
    )
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), min_history=30)
    sig = _feed(strat, n=35)
    assert sig is not None
    assert sig["ticker"] == "NVDA" and sig["action"] == "BUY"
    assert sig["shares"] > 0 and sig["price"] > 0
    assert sig["expert"] == "scalper" and sig["regime"] == "aggressive"


def test_not_approved_returns_none():
    orch = FakeOrch(FinalDecision(action="HOLD", approved=False))
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), min_history=30)
    assert _feed(strat, n=35) is None


def test_sell_when_flat_blocked_without_short():
    orch = FakeOrch(FinalDecision(action="SELL", approved=True))
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), allow_short=False, min_history=30)
    assert _feed(strat, n=35) is None


def test_sell_reduces_long_capped_to_held():
    broker = PaperBroker(cash=100000.0)
    broker.positions["NVDA"] = Position("NVDA", 10.0, 100.0)
    orch = FakeOrch(FinalDecision(action="SELL", approved=True))
    strat = AgentStrategy(orch, broker=broker, allow_short=False, min_history=30)
    sig = _feed(strat, n=35)
    assert sig is not None and sig["action"] == "SELL"
    assert sig["shares"] == 10  # 不超卖现有多仓


def test_sell_opens_short_when_allowed():
    orch = FakeOrch(FinalDecision(action="SELL", approved=True))
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), allow_short=True, min_history=30)
    sig = _feed(strat, n=35)
    assert sig is not None and sig["action"] == "SELL" and sig["shares"] > 0


def test_macro_injected_into_context():
    orch = FakeOrch(FinalDecision(action="HOLD", approved=False))
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), min_history=30)
    strat.update_macro(vix=20.0, tnx=4.0)
    _feed(strat, n=35)
    assert orch.calls
    assert orch.calls[-1].features.get("macro") == {"vix": 20.0, "tnx": 4.0}


def test_features_shape():
    orch = FakeOrch(FinalDecision(action="HOLD", approved=False))
    strat = AgentStrategy(orch, broker=PaperBroker(cash=100000.0), min_history=30)
    _feed(strat, n=35)
    feats = orch.calls[-1].features
    assert "technical" in feats and isinstance(feats["technical"], dict)
    assert "volatility_ann_pct" in feats
    assert feats["technical"]  # 非空


def test_position_and_account_from_broker():
    broker = PaperBroker(cash=80000.0)
    broker.positions["NVDA"] = Position("NVDA", 5.0, 100.0)
    orch = FakeOrch(FinalDecision(action="HOLD", approved=False))
    strat = AgentStrategy(orch, broker=broker, min_history=30)
    _feed(strat, n=35)
    ctx = orch.calls[-1]
    assert ctx.position.shares == 5.0
    assert ctx.account.cash == 80000.0
    assert ctx.account.equity > 0


def test_size_scales_with_equity():
    orch = FakeOrch(FinalDecision(action="BUY", approved=True))
    rich = AgentStrategy(orch, broker=PaperBroker(cash=1000000.0), min_history=30)
    poor = AgentStrategy(orch, broker=PaperBroker(cash=20000.0), min_history=30)
    s_rich = _feed(rich, n=35)
    s_poor = _feed(poor, n=35)
    assert s_rich["shares"] > s_poor["shares"]


def test_no_broker_uses_default_equity():
    orch = FakeOrch(FinalDecision(action="BUY", approved=True))
    strat = AgentStrategy(orch, broker=None, min_history=30, default_equity=50000.0)
    sig = _feed(strat, n=35)
    assert sig is not None and sig["shares"] > 0
