"""期权教学场景单测：简报组装（引擎数字齐全）、四任务产出、0DTE 条件、id 确定性。"""

from __future__ import annotations

from quantai.distill.options_scenarios import (
    OPT_TASKS,
    OPTIONS_SYSTEM_PROMPT,
    build_option_brief,
    build_option_scenarios,
)


def _chain(days=30, expiry="2026-07-31"):
    return {
        "symbol": "AAA", "expiry": expiry, "days_to_expiry": days,
        "calls": [
            {"strike": 155, "bid": 6.0, "ask": 6.6, "volume": 300, "impliedVolatility": 0.53},
            {"strike": 160, "bid": 4.2, "ask": 4.8, "volume": 180, "impliedVolatility": 0.54},
        ],
        "puts": [
            {"strike": 140, "bid": 4.6, "ask": 5.0, "volume": 120, "impliedVolatility": 0.58},
            {"strike": 150, "bid": 8.8, "ask": 9.4, "volume": 200, "impliedVolatility": 0.55},
        ],
    }


def _vals():
    return {"rsi_14": 62.0, "ret_20d_pct": 8.5, "realized_vol_20_ann": 0.55, "in_uptrend": True}


class TestBrief:
    def test_contains_engine_numbers(self):
        brief = build_option_brief("AAA", 152.0, _chain(), _vals())
        assert "ATM 隐含波动率" in brief and "54%" in brief  # (0.53+0.55)/2
        assert "保护性 put：行权 140" in brief
        assert "备兑 call：行权 160" in brief
        assert "theta" in brief and "delta" in brief  # BS 引擎段
        assert "RSI(14) 62.0" in brief

    def test_nearest_chain_section(self):
        near = _chain(days=2, expiry="2026-07-09")
        brief = build_option_brief("AAA", 152.0, _chain(), _vals(), nearest_chain=near)
        assert "近端到期链（2026-07-09，2 天" in brief

    def test_same_expiry_no_duplicate_section(self):
        brief = build_option_brief("AAA", 152.0, _chain(), _vals(), nearest_chain=_chain())
        assert "近端到期链" not in brief


class TestScenarios:
    def test_four_tasks_with_nearest(self):
        scs = list(build_option_scenarios(
            "AAA", 152.0, _chain(), _vals(),
            nearest_chain=_chain(days=2, expiry="2026-07-09"), as_of="2026-07-07",
        ))
        assert {s.task for s in scs} == {f"opt_{t}" for t in OPT_TASKS}
        assert all(s.messages[0]["content"] == OPTIONS_SYSTEM_PROMPT for s in scs)
        assert len({s.scenario_id for s in scs}) == len(scs)  # 确定性唯一

    def test_zero_dte_skipped_without_near_chain(self):
        scs = list(build_option_scenarios("AAA", 152.0, _chain(), _vals(), as_of="2026-07-07"))
        assert "opt_zero_dte" not in {s.task for s in scs}
        assert len(scs) == 3

    def test_zero_dte_not_faked_with_far_chain(self):
        """近端链 >7 天时不出末日题——不拿月权冒充 0DTE。"""
        scs = list(build_option_scenarios(
            "AAA", 152.0, _chain(), _vals(),
            nearest_chain=_chain(days=20, expiry="2026-07-27"), as_of="2026-07-07",
        ))
        assert "opt_zero_dte" not in {s.task for s in scs}
