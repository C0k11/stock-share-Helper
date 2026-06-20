"""quantai.agents.orchestrator.AgentOrchestrator 端到端管线测试（无 torch/无真模型）。"""

from __future__ import annotations

from quantai.agents.base import AgentContext
from quantai.agents.debate import System2Debate
from quantai.agents.gatekeeper import Gatekeeper
from quantai.agents.orchestrator import AgentOrchestrator
from quantai.agents.overlays import ChartistOverlay, MacroGovernor
from quantai.agents.planner import Planner
from quantai.agents.router import HeuristicRouter
from quantai.config import AppConfig


class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.is_loaded = True
        self.calls: list[dict] = []

    def chat(self, messages, *, adapter=None, **kw):
        self.calls.append({"adapter": adapter})
        return self.responses.pop(0) if self.responses else ""


def _orch(*, system2_enabled=False, all_agents=False, require_model=False, committee_policy="conservative"):
    return AgentOrchestrator(
        planner=Planner(policy="rule"),
        gatekeeper=Gatekeeper(model_path="", require_model=require_model, vol_trigger_ann_pct=120.0),
        router=HeuristicRouter(vol_threshold=60.0),
        system2=System2Debate(enabled=system2_enabled, buy_only=True),
        chartist=ChartistOverlay(enabled=False),
        macro=MacroGovernor(enabled=False),
        all_agents_mode=all_agents,
        committee_policy=committee_policy,
    )


def _ctx(ret_5d=5.0, price_vs_ma20=3.0, vol=20.0):
    return AgentContext(
        ticker="NVDA",
        features={
            "technical": {
                "return_5d": ret_5d,
                "price_vs_ma20": price_vs_ma20,
                "volatility_20d": vol,
                "close": 100.0,
            },
            "volatility_ann_pct": vol,
        },
    )


def test_planner_cash_preservation_blocks():
    out = _orch().decide(_ctx(vol=130.0))
    assert out.action == "HOLD"
    assert out.approved is False
    assert out.reason == "planner_cash_preservation"
    assert out.regime == "cash_preservation"


def test_gatekeeper_rejects_without_model():
    # require_model=True + 无模型 -> 拒绝
    out = _orch(require_model=True).decide(_ctx())
    assert out.action == "HOLD"
    assert out.reason == "gatekeeper_rejected"


def test_happy_path_buy_heuristic_no_debate():
    out = _orch(system2_enabled=False).decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0))
    assert out.action == "BUY"
    assert out.approved is True
    assert out.expert == "scalper"


def test_neutral_features_hold():
    out = _orch(system2_enabled=False).decide(_ctx(ret_5d=0.5, price_vs_ma20=0.0))
    assert out.action == "HOLD"
    assert out.approved is False


def test_router_routes_analyst_on_high_vol():
    # vol=75: 规则 regime 仍是 aggressive(<80)，但路由 >=60 -> analyst
    out = _orch(system2_enabled=False).decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0, vol=75.0))
    assert out.expert == "analyst"
    assert out.action == "BUY"


def test_system2_no_llm_passes_through():
    out = _orch(system2_enabled=True).decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=None)
    assert out.action == "BUY"
    assert out.approved is True
    assert out.reason == "system2_no_llm"


def test_system2_veto_blocks_trade():
    # 注入 llm 时，专家先消费一次（出 BUY），再 critic、再 judge
    llm = SeqLLM(
        [
            '{"decision": "BUY", "analysis": "up"}',
            '{"accept": false, "suggested_decision": "HOLD"}',
            '{"final_decision": "HOLD", "rationale": "too risky"}',
        ]
    )
    out = _orch(system2_enabled=True).decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=llm)
    assert out.action == "HOLD"
    assert out.approved is False
    assert out.reason == "too risky"
    assert len(llm.calls) == 3  # expert + critic + judge


def test_system2_override_to_sell():
    llm = SeqLLM(
        [
            '{"decision": "BUY", "analysis": "up"}',
            '{"accept": false, "suggested_decision": "SELL"}',
            '{"final_decision": "SELL", "rationale": "reversal"}',
        ]
    )
    out = _orch(system2_enabled=True).decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=llm)
    assert out.action == "SELL"
    assert out.approved is True


def test_committee_conservative_agree():
    out = _orch(all_agents=True, committee_policy="conservative").decide(
        _ctx(ret_5d=5.0, price_vs_ma20=3.0)
    )
    assert out.action == "BUY"
    assert out.expert == "committee"


def test_committee_conservative_hold_on_neutral():
    out = _orch(all_agents=True, committee_policy="conservative").decide(
        _ctx(ret_5d=0.5, price_vs_ma20=0.0)
    )
    assert out.action == "HOLD"


def test_expert_uses_injected_llm():
    llm = SeqLLM(['{"decision": "SELL", "analysis": "downtrend"}'])
    out = _orch(system2_enabled=False).decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=llm)
    assert out.action == "SELL"  # 模型覆盖启发式
    assert out.approved is True


def test_trace_populated():
    out = _orch(system2_enabled=False).decide(_ctx())
    assert out.trace["regime"] == "aggressive"
    assert "router" in out.trace
    assert out.trace["expert_action"] == "BUY"


def test_from_config_builds_and_runs():
    cfg = AppConfig()
    orch = AgentOrchestrator.from_config(cfg)
    out = orch.decide(_ctx())
    # 不依赖磁盘上是否存在 RL 模型权重：只断言管线跑通、产出合法决策。
    assert out.action in {"HOLD", "BUY", "SELL"}
    assert out.regime == "aggressive"
    assert isinstance(out.to_dict(), dict)


def test_to_dict_shape():
    out = _orch(system2_enabled=False).decide(_ctx())
    d = out.to_dict()
    assert set(d) >= {"action", "approved", "reason", "expert", "regime", "chart_score", "macro_label"}
