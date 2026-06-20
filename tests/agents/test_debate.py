"""quantai.agents.debate.System2Debate 测试（假 llm 注入，纯逻辑+编排）。"""

from __future__ import annotations

from quantai.agents.base import AgentContext, Position
from quantai.agents.debate import System2Debate
from quantai.config import AppConfig


class SeqLLM:
    """按顺序吐出预设响应（第 1 次 = critic，第 2 次 = judge）。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, adapter=None, **kw):
        self.calls.append({"messages": messages, "adapter": adapter})
        return self.responses.pop(0) if self.responses else ""


def _ctx():
    return AgentContext(
        ticker="NVDA",
        features={"technical": {"close": 100.0, "return_5d": 2.0}, "signal": {"composite": 1}},
        position=Position(shares=5.0, avg_price=90.0),
        asof="2024-01-02",
    )


# --- should_run --- #
def test_should_run_buy_only():
    d = System2Debate(enabled=True, buy_only=True)
    assert d.should_run("BUY") is True
    assert d.should_run("SELL") is True
    assert d.should_run("HOLD") is False


def test_should_run_disabled():
    assert System2Debate(enabled=False).should_run("BUY") is False


def test_should_run_not_buy_only_runs_all():
    d = System2Debate(enabled=True, buy_only=False)
    assert d.should_run("HOLD") is True


# --- aggregate --- #
def test_aggregate_hold_and_clear_veto():
    d = System2Debate()
    assert d.aggregate("HOLD", "BUY") == (False, "HOLD", "system2_hold")
    assert d.aggregate("CLEAR", "BUY")[0] is False


def test_aggregate_buy_sell_approved():
    d = System2Debate()
    assert d.aggregate("SELL", "BUY", "flip") == (True, "SELL", "flip")


def test_aggregate_unknown_keeps_proposal():
    d = System2Debate()
    assert d.aggregate("???", "BUY", "r") == (True, "BUY", "r")


# --- run --- #
def test_run_skipped_when_not_triggered():
    d = System2Debate(enabled=True, buy_only=True)
    assert d.run(_ctx(), proposed_action="HOLD") == (True, "HOLD", "system2_skipped")


def test_run_no_llm_passes_through():
    d = System2Debate(enabled=True)
    assert d.run(_ctx(), proposed_action="BUY", llm=None) == (True, "BUY", "system2_no_llm")


def test_run_full_critic_judge_approve():
    llm = SeqLLM(
        [
            '{"accept": true, "suggested_decision": "BUY", "pro": "x", "con": "y", "reasons": ["a"]}',
            '{"final_decision": "BUY", "rationale": "trend intact"}',
        ]
    )
    approved, action, reason = System2Debate().run(
        _ctx(), proposed_action="BUY", proposed_analysis="momentum", llm=llm
    )
    assert approved is True
    assert action == "BUY"
    assert reason == "trend intact"
    assert llm.calls[0]["adapter"] == "system2"
    assert len(llm.calls) == 2


def test_run_judge_vetoes_to_hold():
    llm = SeqLLM(
        [
            '{"accept": false, "suggested_decision": "HOLD"}',
            '{"final_decision": "HOLD", "rationale": "too risky"}',
        ]
    )
    approved, action, reason = System2Debate().run(_ctx(), proposed_action="BUY", llm=llm)
    assert approved is False
    assert action == "HOLD"


def test_run_critic_parse_fail_strict_holds():
    llm = SeqLLM(["i refuse to output json"])
    approved, action, reason = System2Debate(lenient=False).run(
        _ctx(), proposed_action="BUY", llm=llm
    )
    assert approved is False
    assert action == "HOLD"
    assert reason == "critic_parse_failed"


def test_run_critic_parse_fail_lenient_passes():
    llm = SeqLLM(["garbage"])
    approved, action, reason = System2Debate(lenient=True).run(
        _ctx(), proposed_action="BUY", llm=llm
    )
    assert approved is True
    assert action == "BUY"
    assert reason == "critic_parse_failed"


def test_run_judge_parse_fail_strict_holds():
    llm = SeqLLM(['{"accept": true, "suggested_decision": "BUY"}', "no json from judge"])
    approved, action, reason = System2Debate(lenient=False).run(
        _ctx(), proposed_action="BUY", llm=llm
    )
    assert approved is False
    assert action == "HOLD"
    assert reason == "judge_parse_failed"


# --- prompts --- #
def test_build_critic_messages_contains_context():
    msgs = System2Debate().build_critic_messages(
        _ctx(), proposed_expert="scalper", proposed_action="BUY", proposed_analysis="up"
    )
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "Ticker: NVDA" in user
    assert "Proposed decision: BUY" in user
    assert "Current position shares: 5 (LONG)" in user


def test_build_judge_messages_includes_critic_json():
    d = System2Debate()
    msgs = d.build_judge_messages(
        _ctx(), proposed_action="BUY", proposed_analysis="up", critic_json={"accept": True}
    )
    assert "Critic JSON:" in msgs[1]["content"]
    assert "Proposal JSON:" in msgs[1]["content"]


def test_from_config():
    cfg = AppConfig()
    d = System2Debate.from_config(cfg.agents.system2)
    assert d.enabled == cfg.agents.system2.enabled
    assert d.buy_only == cfg.agents.system2.buy_only
    assert d.lenient == cfg.agents.system2.lenient
    assert d.adapter == "system2"
