"""news_scorer 单测：prompt 构造、脏 JSON 抢救、越界裁剪、未打分诚实缺失、聚合。"""

from __future__ import annotations

from quantai.agents.news_scorer import (
    aggregate_symbol_sentiment,
    build_scoring_prompt,
    score_news,
)
from quantai.data.news import entries_to_items


def _items():
    return entries_to_items(
        [
            {"title": "AAA beats earnings, raises guidance", "link": "https://t/1", "summary": "strong quarter"},
            {"title": "AAA faces SEC probe", "link": "https://t/2"},
            {"title": "Market mixed on Fed day", "link": "https://t/3"},
        ],
        symbol="AAA",
        source="test",
    )


class _LLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, user, system=None):
        self.calls.append((user, system))
        return self.reply


class TestScoring:
    def test_prompt_numbers_all_items(self):
        p = build_scoring_prompt(_items())
        assert "0. [AAA] AAA beats earnings" in p
        assert "2. [AAA] Market mixed" in p

    def test_clean_parse(self):
        llm = _LLM('[{"id":0,"sentiment":0.8,"label":"bullish"},'
                   '{"id":1,"sentiment":-0.6,"label":"bearish"},'
                   '{"id":2,"sentiment":0.0,"label":"neutral"}]')
        out = score_news(_items(), llm)
        assert [r["sentiment"] for r in out] == [0.8, -0.6, 0.0]
        assert out[0]["label"] == "bullish"
        assert len(llm.calls) == 1  # 整批一次调用

    def test_markdown_fenced_json_repaired(self):
        llm = _LLM('思考过程...\n```json\n[{"id":0,"sentiment":0.5,"label":"bullish"}]\n```')
        out = score_news(_items(), llm)
        assert out[0]["sentiment"] == 0.5

    def test_out_of_range_clamped(self):
        llm = _LLM('[{"id":0,"sentiment":3.7,"label":"bullish"}]')
        out = score_news(_items(), llm)
        assert out[0]["sentiment"] == 1.0

    def test_missing_entries_honestly_unscored(self):
        llm = _LLM('[{"id":0,"sentiment":0.4,"label":"bullish"}]')  # 只回了 1/3
        out = score_news(_items(), llm)
        assert out[1]["sentiment"] is None and out[1]["label"] == "unscored"
        assert out[2]["sentiment"] is None

    def test_garbage_reply_all_unscored_not_zero(self):
        out = score_news(_items(), _LLM("抱歉我不能……"))
        assert all(r["sentiment"] is None for r in out)  # 不许拿 0 冒充中性

    def test_empty_input_zero_calls(self):
        llm = _LLM("[]")
        assert score_news([], llm) == []
        assert llm.calls == []

    def test_out_of_range_and_duplicate_ids_rejected(self):
        """LLM 幻觉 id（越界/重复）不得把分挂到错误头条上：越界丢弃、重复保首个。"""
        llm = _LLM('[{"id":0,"sentiment":0.9,"label":"bullish"},'
                   '{"id":0,"sentiment":-0.9,"label":"bearish"},'
                   '{"id":99,"sentiment":1.0,"label":"bullish"}]')
        out = score_news(_items(), llm)
        assert out[0]["sentiment"] == 0.9 and out[0]["label"] == "bullish"
        assert out[1]["sentiment"] is None and out[2]["sentiment"] is None

    def test_invalid_label_derived_from_sentiment(self):
        """label 非法（如 'positive'）时按 sentiment 推导，不再一律填 neutral。"""
        llm = _LLM('[{"id":0,"sentiment":0.9,"label":"positive"},'
                   '{"id":1,"sentiment":-0.8,"label":"负面"},'
                   '{"id":2,"sentiment":0.05,"label":"whatever"}]')
        out = score_news(_items(), llm)
        assert [r["label"] for r in out] == ["bullish", "bearish", "neutral"]


class TestAggregate:
    def test_symbol_mean_skips_unscored(self):
        items = _items()
        scored = [
            {"item": items[0], "sentiment": 0.8, "label": "bullish"},
            {"item": items[1], "sentiment": -0.4, "label": "bearish"},
            {"item": items[2], "sentiment": None, "label": "unscored"},
        ]
        agg = aggregate_symbol_sentiment(scored)
        assert agg["AAA"] == 0.2  # (0.8-0.4)/2，未打分条不参与
