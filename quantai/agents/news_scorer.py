"""新闻量化：LLM 对头条批量打情绪分（-1 空 ~ +1 多），一次调用打一批。

诚实口径：
- 输入只有**标题级**信息（RSS 标题+摘要，无全文）——打的是"标题情绪"，
  不是深度事件分析；prompt 里明说，让模型对模糊标题给低 |score|。
- LLM 输出经 `repair_and_parse_json` 抢救；解析失败/缺条目的头条**不给分**
  （score=None），绝不用中性 0 冒充"已量化"。
- LLM 注入（`generate(user, system)` 鸭子接口），Mock 可测零 GPU。
"""

from __future__ import annotations

from typing import Optional

from quantai.llm.json_utils import repair_and_parse_json

_SYSTEM = (
    "你是金融新闻情绪标注器。对每条头条给 sentiment ∈ [-1, 1]"
    "（-1 极度利空，0 中性，+1 极度利多，仅凭标题无法判断时给接近 0 的值）"
    "和 label ∈ {bullish, bearish, neutral}。"
    "只输出 JSON 数组，元素形如 {\"id\": 0, \"sentiment\": 0.5, \"label\": \"bullish\"}，"
    "不要输出任何其它文字。"
)


def build_scoring_prompt(items: list) -> str:
    """头条列表 → 编号打分任务（一次调用打整批）。"""
    lines = ["对下列新闻头条打分："]
    for i, it in enumerate(items):
        sym = f"[{it.symbol}] " if getattr(it, "symbol", None) else ""
        summary = (getattr(it, "summary", "") or "")[:200]
        lines.append(f"{i}. {sym}{it.title}" + (f" —— {summary}" if summary else ""))
    return "\n".join(lines)


def score_news(items: list, llm) -> list[dict]:
    """头条 → [{item, sentiment, label}]。解析不出的条目 sentiment=None（诚实缺失）。

    返回长度恒等于输入长度、顺序一致；空输入直接返回 []（零 LLM 调用）。
    """
    if not items:
        return []
    raw = llm.generate(build_scoring_prompt(items), system=_SYSTEM)
    parsed = repair_and_parse_json(raw)
    by_id: dict[int, dict] = {}
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("id"))
                s = entry.get("sentiment")
                s = max(-1.0, min(1.0, float(s))) if s is not None else None
            except (TypeError, ValueError):
                continue
            label = str(entry.get("label", "")).lower()
            if label not in ("bullish", "bearish", "neutral"):
                label = ""
            by_id[idx] = {"sentiment": s, "label": label}
    out = []
    for i, it in enumerate(items):
        hit = by_id.get(i, {})
        out.append(
            {
                "item": it,
                "sentiment": hit.get("sentiment"),
                "label": hit.get("label") or ("unscored" if hit.get("sentiment") is None else "neutral"),
            }
        )
    return out


def aggregate_symbol_sentiment(scored: list[dict]) -> dict[str, Optional[float]]:
    """按 symbol 聚合平均情绪（只算打出分的条目；一条都没打出 → None）。"""
    acc: dict[str, list[float]] = {}
    for row in scored:
        sym = getattr(row["item"], "symbol", None)
        if sym and row["sentiment"] is not None:
            acc.setdefault(sym, []).append(row["sentiment"])
    return {s: (sum(v) / len(v) if v else None) for s, v in acc.items()}
