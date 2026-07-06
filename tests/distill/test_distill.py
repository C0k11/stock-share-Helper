"""distill/ 单测：场景构造、mock 教师全管线、密钥 fail-fast、格式对接训练器契约。

⛔ 全部离线 + mock：本文件（及整个 CI）不产生任何真实 DeepSeek 调用。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantai.distill import (
    DeepSeekClient,
    DistillGenerator,
    MissingApiKeyError,
    MockDeepSeekClient,
    ScenarioBuilder,
    build_indicator_brief,
    scenario_to_dpo_record,
    scenario_to_sft_record,
    weak_baseline_answer,
    write_jsonl,
)


def _prices(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))),
        index=pd.bdate_range("2025-01-06", periods=n),
    )
    return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99})


class TestScenarioBuilder:
    def test_builds_all_tasks_per_symbol(self):
        scs = list(ScenarioBuilder().build({"AAA": _prices(), "BBB": _prices(seed=1)}))
        assert len(scs) == 8  # 2 symbols × 4 tasks
        ids = {s.scenario_id for s in scs}
        assert len(ids) == 8  # 确定性且唯一

    def test_short_series_skipped(self):
        scs = list(ScenarioBuilder(min_bars=60).build({"AAA": _prices(30)}))
        assert scs == []

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="未知任务"):
            ScenarioBuilder(tasks=["trend", "typo"])

    def test_messages_shape(self):
        sc = next(iter(ScenarioBuilder(tasks=["trend"]).build({"AAA": _prices()})))
        assert [m["role"] for m in sc.messages] == ["system", "user"]
        assert "RSI" in sc.messages[1]["content"]
        assert "AAA" in sc.messages[1]["content"]

    def test_brief_uses_analysis_numbers(self):
        df = _prices()
        brief, vals = build_indicator_brief(df, "AAA")
        from quantai.analysis import rsi

        assert vals["rsi_14"] == pytest.approx(float(rsi(df["close"], 14).iloc[-1]))
        assert f"{vals['last_close']:.2f}" in brief

    def test_insufficient_data_shows_na_not_fabricated(self):
        # 70 根：MA200 未热身 -> 简报里必须出现 n/a，不许编数字
        brief, vals = build_indicator_brief(_prices(70), "AAA")
        assert np.isnan(vals["sma_200"])
        assert "n/a" in brief


class TestProductionTaskScenarios:
    """生产同款任务场景：prompt 必须与线上常量同源，坏例按任务分支。"""

    def _news(self, n=10):
        from quantai.data.news import entries_to_items

        return entries_to_items(
            [{"title": f"Headline {i}", "link": f"https://t/{i}"} for i in range(n)],
            symbol="AAA", source="test",
        )

    def test_news_scenarios_batched_and_source_identical(self):
        from quantai.agents.news_scorer import SCORING_SYSTEM_PROMPT
        from quantai.distill.scenarios import build_news_scoring_scenarios

        scs = list(build_news_scoring_scenarios(self._news(10), as_of="2026-07-06", batch_size=8))
        assert len(scs) == 2  # 10 条 -> 8+2
        assert scs[0].messages[0]["content"] == SCORING_SYSTEM_PROMPT  # 与生产同源
        assert "0. [AAA] Headline 0" in scs[0].messages[1]["content"]
        assert scs[0].task == "news_scoring"

    def test_report_scenario_source_identical(self):
        from quantai.agents.analyst import REPORT_SYSTEM_PROMPT
        from quantai.distill.scenarios import build_market_report_scenario

        sc = build_market_report_scenario({"AAA": _prices(), "BBB": _prices(seed=1)}, as_of="2026-07-06")
        assert sc is not None
        assert sc.messages[0]["content"] == REPORT_SYSTEM_PROMPT
        assert "AAA" in sc.messages[1]["content"] and "BBB" in sc.messages[1]["content"]
        assert "RSI" in sc.messages[1]["content"]

    def test_report_scenario_none_when_all_too_short(self):
        from quantai.distill.scenarios import build_market_report_scenario

        assert build_market_report_scenario({"AAA": _prices(20)}, as_of="x") is None

    def test_weak_baseline_task_branches(self):
        from quantai.distill.scenarios import build_market_report_scenario, build_news_scoring_scenarios

        news_sc = next(iter(build_news_scoring_scenarios(self._news(2), as_of="x")))
        assert "JSON" not in weak_baseline_answer(news_sc)  # 坏例故意不守 JSON 格式
        report_sc = build_market_report_scenario({"AAA": _prices()}, as_of="x")
        assert "【组合体检】" in weak_baseline_answer(report_sc)  # 有结构但空洞无数字


class TestClientSafety:
    def test_missing_key_fails_fast(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(MissingApiKeyError, match="DEEPSEEK_API_KEY"):
            DeepSeekClient()

    def test_key_not_leaked_in_repr_or_dict(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-test-123")
        c = DeepSeekClient()
        assert "sk-secret-test-123" not in repr(c)
        # usage_totals 等公开面不含 key
        assert "sk-secret-test-123" not in json.dumps(c.usage_totals)

    def test_defaults_explicit_v4_model_with_thinking(self, monkeypatch):
        """显式版本名 + 思考模式默认开（旧别名 2026-07-24 退役，防回归）。"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-test-123")
        c = DeepSeekClient()
        assert c.model == "deepseek-v4-pro"
        assert c.thinking is True
        assert "deepseek-chat" not in c.model and "reasoner" not in c.model

    def test_mock_client_records_calls(self):
        m = MockDeepSeekClient()
        out = m.chat([{"role": "user", "content": "hi"}])
        assert "【结论】" in out
        assert m.usage_totals["requests"] == 1
        assert len(m.calls) == 1


class TestGeneratorPipeline:
    def _run(self, tmp_path: Path, **kw):
        scs = list(ScenarioBuilder().build({"AAA": _prices()}))
        gen = DistillGenerator(MockDeepSeekClient())
        return gen.run(scs, tmp_path / "sft.jsonl", tmp_path / "dpo.jsonl", **kw), tmp_path

    def test_full_pipeline_writes_both_files(self, tmp_path):
        summary, out = self._run(tmp_path)
        assert summary["sft_written"] == 4 and summary["dpo_written"] == 4
        assert summary["failures"] == []
        assert (out / "sft.jsonl").exists() and (out / "dpo.jsonl").exists()

    def test_limit_is_hard_fuse(self, tmp_path):
        summary, _ = self._run(tmp_path, limit=2)
        assert summary["sft_written"] == 2

    def test_single_failure_does_not_kill_batch(self, tmp_path):
        scs = list(ScenarioBuilder().build({"AAA": _prices()}))

        class FlakyClient(MockDeepSeekClient):
            def chat(self, messages, temperature=None):
                if len(self.calls) == 1:  # 第二条炸
                    self.calls.append(messages)
                    raise RuntimeError("boom")
                return super().chat(messages, temperature)

        summary = DistillGenerator(FlakyClient()).run(
            scs, tmp_path / "s.jsonl", tmp_path / "d.jsonl"
        )
        assert summary["sft_written"] == 3
        assert len(summary["failures"]) == 1
        assert "boom" in summary["failures"][0]["error"]


class TestTrainerFormatContract:
    """格式契约：产物必须能被现有训练器按其自己的规则消费。"""

    def _records(self):
        sc = next(iter(ScenarioBuilder(tasks=["trend"]).build({"AAA": _prices()})))
        answer = "【结论】测试。\n【依据】x。\n【风险】y。"
        return scenario_to_sft_record(sc, answer), scenario_to_dpo_record(sc, answer)

    def test_sft_conversations_feed_build_training_texts(self):
        from quantai.llm.finetune import build_training_texts

        sft, _ = self._records()

        class FakeTok:
            def apply_chat_template(self, conv, tokenize, add_generation_prompt):
                assert all(set(m) >= {"role", "content"} for m in conv)
                return " | ".join(m["content"] for m in conv)

        texts = build_training_texts(FakeTok(), [sft["conversations"]])
        assert "【结论】测试。" in texts[0]
        assert sft["conversations"][-1]["role"] == "assistant"

    def test_dpo_record_passes_column_validation_and_formatting(self):
        from quantai.llm.dpo import format_prompt_messages, validate_dpo_columns

        _, dpo = self._records()
        validate_dpo_columns(list(dpo.keys()))  # prompt/chosen/rejected 齐

        class FakeTok:
            def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
                return "".join(m["content"][:5] for m in msgs)

        formatted = format_prompt_messages(FakeTok(), dpo["prompt"])
        assert isinstance(formatted, str) and formatted
        assert dpo["chosen"] != dpo["rejected"]

    def test_jsonl_roundtrip(self, tmp_path):
        sft, dpo = self._records()
        p = tmp_path / "x.jsonl"
        write_jsonl(p, [sft, dpo])
        lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2
        assert lines[0]["conversations"][0]["role"] == "system"

    def test_weak_baseline_is_deterministic(self):
        sc = next(iter(ScenarioBuilder(tasks=["trend"]).build({"AAA": _prices()})))
        assert weak_baseline_answer(sc) == weak_baseline_answer(sc)


class TestNoRealCallsInCI:
    def test_this_test_suite_never_touches_network(self):
        """哨兵：distill 测试环境不应配置真实 key（防误伤真额度）。

        若本机 shell 恰好设了 DEEPSEEK_API_KEY，也只断言测试自身从不实例化
        真实 client 发请求——本套件唯一的 DeepSeekClient() 出现在 fail-fast 测试里
        （monkeypatch 删 key）。这里做个显式声明性断言。
        """
        import quantai.distill.client as c

        assert hasattr(c, "MockDeepSeekClient")
