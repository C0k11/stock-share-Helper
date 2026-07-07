"""journal 飞轮单测：规则学生、评审解析、日采幂等、结局回填、训练格式导出。全 mock 零调用。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantai.distill import MockDeepSeekClient
from quantai.distill.journal import (
    backfill_outcomes,
    build_review_messages,
    journal_to_dpo_records,
    journal_to_sft_records,
    load_journal,
    parse_review_score,
    rule_student_answer,
    run_daily_journal,
)
from quantai.distill.scenarios import ScenarioBuilder

_ACTIONS = ("加仓", "持有", "观望", "减仓")


def _prices(n=250, seed=0, drift=0.0006) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n))),
        index=pd.bdate_range("2025-01-06", periods=n),
    )
    return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99})


def _vals(**over) -> dict:
    base = {
        "ma_alignment": 1.0, "in_uptrend": True, "macd_hist": 0.5, "ret_20d_pct": 4.2,
        "rsi_14": 55.0, "retrace_from_20d_high": -0.02, "realized_vol_20_ann": 0.25,
        "sma_20": 101.5, "last_close": 103.0,
    }
    base.update(over)
    return base


class TestRuleStudent:
    def test_bullish_vals_give_add_and_cite_numbers(self):
        ans = rule_student_answer("AAA", _vals())
        assert "加仓" in ans and "【结论】" in ans and "【依据】" in ans and "【风险】" in ans
        assert "55.0" in ans and "4.20" in ans  # 引用真实数值
        assert "101.50" in ans  # MA20 失效条件

    def test_bearish_vals_give_reduce(self):
        ans = rule_student_answer(
            "BBB",
            _vals(ma_alignment=0.0, in_uptrend=False, macd_hist=-0.4, ret_20d_pct=-8.0,
                  retrace_from_20d_high=-0.2, realized_vol_20_ann=0.9),
        )
        assert "减仓" in ans

    def test_action_always_one_of_four(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            ans = rule_student_answer(
                "X",
                _vals(ma_alignment=float(rng.uniform(0, 1)), in_uptrend=bool(rng.integers(2)),
                      macd_hist=float(rng.normal()), ret_20d_pct=float(rng.normal(0, 5)),
                      rsi_14=float(rng.uniform(10, 90))),
            )
            assert any(a in ans for a in _ACTIONS)

    def test_invalidation_line_respects_price_vs_ma20(self):
        above = rule_student_answer("A", _vals(last_close=103.0, sma_20=101.5))
        assert "跌破 MA20" in above
        below = rule_student_answer("A", _vals(last_close=95.0, sma_20=101.5))
        assert "站稳 MA20" in below and "跌破 MA20" not in below

    def test_nan_factors_honest_and_no_crash(self):
        ans = rule_student_answer("N", _vals(ma_alignment=float("nan"), macd_hist=float("nan"),
                                             ret_20d_pct=float("nan"), rsi_14=float("nan"),
                                             sma_20=float("nan")))
        assert "不计分" in ans and "n/a" in ans

    def test_deterministic(self):
        assert rule_student_answer("A", _vals()) == rule_student_answer("A", _vals())


class TestReviewParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("评分: 7\n【点评】还行", 7),
            ("评分：10\n【点评】完美", 10),
            ("评分: 0\n差", 0),
            ("我给 8/10 的评价", 8),
            ("评分: 99\n越界", None),
            ("没有分数的散文", None),
            ("", None),
        ],
    )
    def test_parse(self, text, expected):
        assert parse_review_score(text) == expected

    def test_review_messages_contain_brief_and_student(self):
        sc = next(iter(ScenarioBuilder(tasks=["decision"]).build({"AAA": _prices()})))
        msgs = build_review_messages(sc, "【结论】学生答案")
        assert msgs[0]["role"] == "system" and "评审" in msgs[0]["content"]
        assert "RSI" in msgs[1]["content"] and "【系统判断】" in msgs[1]["content"]
        assert "学生答案" in msgs[1]["content"]


class TestRunDailyJournal:
    def _run(self, tmp_path, **kw):
        client = MockDeepSeekClient(canned="评分: 3\n【点评】方向对但没引用数值。\n【正确做法】持有。")
        prices = {"AAA": _prices(), "BBB": _prices(seed=1)}
        return client, run_daily_journal(prices, client, tmp_path, **kw)

    def test_writes_complete_records(self, tmp_path):
        client, summary = self._run(tmp_path)
        assert summary["written"] == 2 and summary["scored"] == 2 and not summary["failures"]
        recs = [json.loads(line) for line in Path(summary["path"]).read_text(encoding="utf-8").splitlines()]
        r = recs[0]
        assert r["task"] == "decision" and r["student_kind"] == "rule_v1"
        assert any(a in r["student_answer"] for a in _ACTIONS)
        assert r["review"]["score"] == 3
        assert r["meta"]["outcome"] is None  # 未回填前诚实为 None
        assert [m["role"] for m in r["messages"]] == ["system", "user"]
        # 每场景恰 2 次教师调用（作答 + 评审）
        assert client.usage_totals["requests"] == 4

    def test_same_data_day_idempotent(self, tmp_path):
        _, s1 = self._run(tmp_path)
        client2, s2 = self._run(tmp_path)
        assert s2["skipped"] is True
        assert client2.usage_totals["requests"] == 0  # 跳过 = 零调用零花费
        recs = [json.loads(line) for line in Path(s1["path"]).read_text(encoding="utf-8").splitlines()]
        assert len(recs) == 2  # 没有追加重复

    def test_247_symbol_does_not_block_or_duplicate_equities(self, tmp_path):
        """BTC-USD 型 7×24 标的：as_of 恒为今天，不得拖股票成重复采集/误跳过。"""
        client = MockDeepSeekClient(canned="评分: 5\nok")
        stock = _prices(250)  # 最后一根 = 周五（bdate_range）
        crypto_early = pd.DataFrame(
            {"close": pd.Series(np.linspace(90, 110, 250),
                                index=pd.date_range("2025-03-01", periods=250, freq="D"))}
        )
        s1 = run_daily_journal({"AAA": stock, "BTC": crypto_early}, client, tmp_path)
        assert s1["written"] == 2

        # 周末再跑：股票 as_of 不变（去重跳过），crypto 多一根新 as_of（只采它）
        crypto_late = pd.DataFrame(
            {"close": pd.Series(np.linspace(90, 110.5, 251),
                                index=pd.date_range("2025-03-01", periods=251, freq="D"))}
        )
        client2 = MockDeepSeekClient(canned="评分: 5\nok")
        s2 = run_daily_journal({"AAA": stock, "BTC": crypto_late}, client2, tmp_path)
        assert s2["written"] == 1  # 只有 crypto 的新快照
        assert client2.usage_totals["requests"] == 2  # 1 场景 × 2 调用，股票零重复花费
        all_ids = [r["scenario_id"] for _, recs in load_journal(tmp_path) for r in recs]
        assert len(all_ids) == len(set(all_ids)) == 3  # 全局无重复

    def test_force_reruns(self, tmp_path):
        self._run(tmp_path)
        client, s2 = self._run(tmp_path, force=True)
        assert s2["skipped"] is False and client.usage_totals["requests"] == 4

    def test_cap_limits_scenarios(self, tmp_path):
        _, summary = self._run(tmp_path, cap=1)
        assert summary["written"] == 1

    def test_empty_teacher_answer_recorded_as_failure(self, tmp_path):
        client = MockDeepSeekClient(canned="")
        summary = run_daily_journal({"AAA": _prices()}, client, tmp_path)
        assert summary["written"] == 0 and len(summary["failures"]) == 1
        assert "empty" in summary["failures"][0]["error"]

    def test_short_history_no_scenarios(self, tmp_path):
        client = MockDeepSeekClient()
        summary = run_daily_journal({"AAA": _prices(30)}, client, tmp_path)
        assert summary["written"] == 0 and "no scenarios" in summary.get("reason", "")


class TestBackfillOutcomes:
    def test_fills_maturing_records_incrementally(self, tmp_path):
        full = _prices(250)
        cut = full.index[200]
        client = MockDeepSeekClient(canned="评分: 5\nok")
        run_daily_journal({"AAA": full.loc[:cut]}, client, tmp_path)

        # 只喂 +6 根：fwd_5d 可算，fwd_20d 还不成熟
        part = full.loc[: full.index[206]]
        stat = backfill_outcomes(tmp_path, {"AAA": part})
        assert stat["records_updated"] == 1
        rec = load_journal(tmp_path)[0][1][0]
        base = float(full.loc[cut, "close"])
        expect_5d = float(full["close"].iloc[205] / base - 1)
        assert rec["meta"]["outcome"]["fwd_5d"] == pytest.approx(expect_5d, rel=1e-12)
        assert rec["meta"]["outcome"]["fwd_20d"] is None

        # 全量数据：补齐 fwd_20d，且 fwd_5d 不变
        stat2 = backfill_outcomes(tmp_path, {"AAA": full})
        assert stat2["records_updated"] == 1
        rec = load_journal(tmp_path)[0][1][0]
        assert rec["meta"]["outcome"]["fwd_5d"] == pytest.approx(expect_5d, rel=1e-12)
        assert rec["meta"]["outcome"]["fwd_20d"] == pytest.approx(
            float(full["close"].iloc[220] / base - 1), rel=1e-12
        )

        # 已填齐后幂等
        assert backfill_outcomes(tmp_path, {"AAA": full})["records_updated"] == 0

    def test_missing_symbol_prices_skipped(self, tmp_path):
        client = MockDeepSeekClient(canned="评分: 5\nok")
        run_daily_journal({"AAA": _prices()}, client, tmp_path)
        stat = backfill_outcomes(tmp_path, {})
        assert stat["records_updated"] == 0


class TestExports:
    def _records(self, tmp_path, canned="评分: 3\n【点评】弱。\n【正确做法】观望。"):
        client = MockDeepSeekClient(canned=canned)
        run_daily_journal({"AAA": _prices(), "BBB": _prices(seed=1)}, client, tmp_path)
        return [r for _, recs in load_journal(tmp_path) for r in recs]

    def test_sft_contract(self, tmp_path):
        recs = journal_to_sft_records(self._records(tmp_path))
        assert len(recs) == 2
        conv = recs[0]["conversations"]
        assert [m["role"] for m in conv] == ["system", "user", "assistant"]
        assert conv[-1]["content"].strip()
        assert recs[0]["meta"]["review_score"] == 3

    def test_dpo_only_low_scores(self, tmp_path):
        low = self._records(tmp_path)
        pairs = journal_to_dpo_records(low, max_student_score=6)
        assert len(pairs) == 2
        assert set(pairs[0]) >= {"prompt", "chosen", "rejected"}
        assert pairs[0]["rejected"] != pairs[0]["chosen"]
        assert pairs[0]["meta"]["rejected_kind"] == "rule_v1"
        # 高分学生不硬造偏好对
        assert journal_to_dpo_records(low, max_student_score=2) == []

    def test_dpo_skips_unscored(self, tmp_path):
        recs = self._records(tmp_path, canned="没有分数的评审散文")
        assert journal_to_dpo_records(recs) == []
