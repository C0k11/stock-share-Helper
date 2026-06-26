"""quantai.live.runner.LiveRunner 测试（同步 drain，不依赖线程时序）。"""

from __future__ import annotations

from quantai.agents.base import FinalDecision
from quantai.live.events import EventType
from quantai.live.runner import LiveRunner


class AlwaysBuy:
    def decide(self, ctx, llm=None):
        return FinalDecision(action="BUY", approved=True, expert="scalper", reason="t", regime="aggressive")


def _drain(engine, limit=2000):
    n = 0
    while not engine.events.empty() and n < limit:
        engine._handle_event(engine.events.get_nowait())
        n += 1


def test_builds_with_defaults_and_snapshot():
    runner = LiveRunner(["NVDA"], source="simulated", seed=1, cash=100000.0)
    snap = runner.snapshot()
    assert snap["cash"] == 100000.0
    assert snap["equity"] == 100000.0
    assert snap["positions"] == {}
    assert snap["orders"] == 0
    assert snap["feed_source"] == "simulated"


def test_default_orchestrator_built():
    runner = LiveRunner(["NVDA"], source="simulated", seed=1)
    # 零配置默认大脑可用
    from quantai.agents.orchestrator import AgentOrchestrator

    assert isinstance(runner.orchestrator, AgentOrchestrator)


def test_end_to_end_buy_fills_position():
    runner = LiveRunner(["NVDA"], orchestrator=AlwaysBuy(), source="simulated", seed=1, cash=100000.0)
    # 喂 40 根上行 bar，逐个同步处理整条事件级联
    for i in range(40):
        px = 100.0 + i
        runner.engine.push(
            EventType.MARKET_DATA,
            {"ticker": "NVDA", "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1000},
        )
        _drain(runner.engine)
    assert "NVDA" in runner.broker.positions
    assert runner.broker.positions["NVDA"].shares > 0
    assert runner.broker.orders  # 真的下过单
    snap = runner.snapshot()
    assert snap["positions"]["NVDA"]["shares"] > 0


def test_fill_events_routed_through_engine():
    runner = LiveRunner(["NVDA"], orchestrator=AlwaysBuy(), source="simulated", seed=2, cash=100000.0)
    fills = []
    # 包一层监听 FILL（通过把 portfolio 接上）
    class P:
        def on_fill(self, f):
            fills.append(f)

    runner.engine.portfolio = P()
    for i in range(40):
        px = 100.0 + i
        runner.engine.push(
            EventType.MARKET_DATA,
            {"ticker": "NVDA", "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1000},
        )
        _drain(runner.engine)
    assert fills and all(f["ticker"] == "NVDA" for f in fills)
