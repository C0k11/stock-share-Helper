"""quantai.agents.experts 测试（假 LLM 注入；验证 F-008 启发式去随机）。"""

from __future__ import annotations

from quantai.agents.base import Account, AgentContext, Position
from quantai.agents.experts import (
    AnalystExpert,
    NewsExpert,
    ScalperExpert,
    make_expert,
)
from quantai.agents.experts.base import LLMExpert


class FakeLLM:
    def __init__(self, response="", is_loaded=True, raise_exc=False):
        self.response = response
        self.is_loaded = is_loaded
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def chat(self, messages, *, adapter=None, **kw):
        self.calls.append({"messages": messages, "adapter": adapter})
        if self.raise_exc:
            raise RuntimeError("boom")
        return self.response


def _ctx(ret_5d=0.0, price_vs_ma20=0.0, vol=20.0, **kw):
    return AgentContext(
        ticker="NVDA",
        features={
            "technical": {
                "return_5d": ret_5d,
                "price_vs_ma20": price_vs_ma20,
                "volatility_20d": vol,
                "close": 100.0,
            }
        },
        **kw,
    )


# --- 启发式（确定性） --- #
def test_heuristic_buy():
    d = ScalperExpert().heuristic_decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0))
    assert d.decision == "BUY"
    assert d.expert == "scalper"
    assert d.meta["source"] == "heuristic"


def test_heuristic_sell():
    d = ScalperExpert().heuristic_decide(_ctx(ret_5d=-5.0, price_vs_ma20=-3.0))
    assert d.decision == "SELL"


def test_heuristic_hold():
    d = ScalperExpert().heuristic_decide(_ctx(ret_5d=0.5, price_vs_ma20=0.0))
    assert d.decision == "HOLD"


def test_heuristic_is_deterministic_no_random():
    # 同一输入多次必须一致（旧 _heuristic_infer 有 30% 概率随机改判）
    e = ScalperExpert()
    ctx = _ctx(ret_5d=5.0, price_vs_ma20=3.0)
    assert {e.heuristic_decide(ctx).decision for _ in range(50)} == {"BUY"}


# --- 模型路径（注入假 LLM） --- #
def test_decide_no_llm_uses_heuristic():
    d = ScalperExpert().decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=None)
    assert d.decision == "BUY"
    assert d.meta["source"] == "heuristic"


def test_decide_with_llm_parses_json():
    llm = FakeLLM(response='{"decision": "SELL", "analysis": "overbought"}')
    d = AnalystExpert().decide(_ctx(), llm=llm)
    assert d.decision == "SELL"
    assert d.analysis == "overbought"
    assert d.meta["source"] == "model"
    assert d.meta["adapter"] == "analyst"
    assert llm.calls[0]["adapter"] == "analyst"  # 路由到正确 adapter


def test_decide_with_llm_garbage_returns_hold():
    llm = FakeLLM(response="i have no idea what json is")
    d = ScalperExpert().decide(_ctx(), llm=llm)
    assert d.decision == "HOLD"


def test_decide_llm_not_loaded_uses_heuristic():
    llm = FakeLLM(response='{"decision": "BUY"}', is_loaded=False)
    d = ScalperExpert().decide(_ctx(ret_5d=-5.0, price_vs_ma20=-3.0), llm=llm)
    assert d.decision == "SELL"  # 走启发式
    assert d.meta["source"] == "heuristic"
    assert llm.calls == []  # 没调用模型


def test_decide_llm_exception_falls_back_to_heuristic():
    llm = FakeLLM(raise_exc=True)
    d = ScalperExpert().decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=llm)
    assert d.decision == "BUY"
    assert d.meta["source"] == "heuristic"


def test_decide_llm_empty_response_falls_back():
    llm = FakeLLM(response="   ")
    d = ScalperExpert().decide(_ctx(ret_5d=5.0, price_vs_ma20=3.0), llm=llm)
    assert d.meta["source"] == "heuristic"


# --- prompt 构建 --- #
def test_build_user_prompt_contains_context():
    ctx = AgentContext(
        ticker="TSLA",
        features={"technical": {"close": 250.0, "return_5d": 1.2, "volatility_20d": 40.0}},
        position=Position(shares=12.0, avg_price=200.0),
        account=Account(cash=5000.0, equity=8000.0),
        allow_short=True,
    )
    p = ScalperExpert().build_user_prompt(ctx)
    assert "Ticker: TSLA" in p
    assert "allow_short: true" in p
    assert "shares: 12.0000" in p
    assert "Close: 250.00" in p


def test_build_messages_has_system_and_user():
    msgs = ScalperExpert().build_messages(_ctx())
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "JSON" in msgs[0]["content"]


# --- 工厂 / adapters --- #
def test_make_expert_maps_names():
    assert isinstance(make_expert("scalper"), ScalperExpert)
    assert isinstance(make_expert("analyst"), AnalystExpert)
    assert isinstance(make_expert("news"), NewsExpert)


def test_make_expert_unknown_falls_back_scalper():
    assert isinstance(make_expert("wizard"), ScalperExpert)
    assert isinstance(make_expert(""), ScalperExpert)


def test_expert_adapters():
    assert ScalperExpert().adapter == "scalper"
    assert AnalystExpert().adapter == "analyst"
    assert NewsExpert().adapter == "news"


def test_custom_adapter_override():
    e = make_expert("analyst", adapter="analyst_v4")
    assert e.adapter == "analyst_v4"
    assert isinstance(e, AnalystExpert)


def test_base_expert_default_adapter():
    assert LLMExpert().adapter == "scalper"
