"""quantai.live.runner.LiveRunner 测试（同步 drain，不依赖线程时序）。"""

from __future__ import annotations

from quantai.agents.base import FinalDecision
from quantai.live.events import EventType
from quantai.live.runner import LiveRunner


class AlwaysBuy:
    def decide(self, ctx, llm=None):
        return FinalDecision(action="BUY", approved=True, expert="scalper", reason="t", regime="aggressive")


class BuyThenSell:
    def __init__(self, switch_after: int):
        self.n = 0
        self.switch = switch_after

    def decide(self, ctx, llm=None):
        self.n += 1
        action = "BUY" if self.n <= self.switch else "SELL"
        return FinalDecision(action=action, approved=True, expert="scalper", reason="t", regime="aggressive")


class FakeRecorder:
    def __init__(self):
        self.records = []
        self.outcomes = []

    def record(self, *, agent_id, context, action, outcome=None, feedback=""):
        rid = f"r{len(self.records)}"
        self.records.append({"id": rid, "agent_id": agent_id, "action": action})
        return rid

    def log_outcome(self, *, ref_id, outcome, comment=""):
        self.outcomes.append({"ref_id": ref_id, "outcome": outcome})


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


def test_flywheel_records_decisions_and_backfills_outcomes():
    rec = FakeRecorder()
    runner = LiveRunner(
        ["NVDA"], orchestrator=BuyThenSell(switch_after=3), source="simulated", seed=1, cash=100000.0, recorder=rec
    )
    for i in range(40):
        px = 100.0 + i  # 上行：买入累积多头，后续卖出回吐 -> 触发 log_outcome
        runner.engine.push(
            EventType.MARKET_DATA,
            {"ticker": "NVDA", "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1000},
        )
        _drain(runner.engine)
    # 决策被记录（带 ref_id），平仓盈亏回填到这些 ref_id
    assert rec.records, "决策应被 recorder.record 记录"
    assert rec.outcomes, "平仓应通过 broker.log_outcome 回填盈亏"
    rec_ids = {r["id"] for r in rec.records}
    assert all(o["ref_id"] in rec_ids for o in rec.outcomes)
