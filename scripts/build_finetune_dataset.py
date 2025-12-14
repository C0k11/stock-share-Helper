#!/usr/bin/env python
"""构建微调数据集（弱标注）

流程：
1) 从RSS/NewsAPI抓取新闻（DataFetcher.fetch_news）
2) 使用 RuleBasedNewsParser 生成弱标注
3) 输出 HuggingFace JSON 数据集格式（conversations）到 data/finetune/train.json

用法：
  .\\venv311\\Scripts\\python.exe scripts\\build_finetune_dataset.py --limit 50
  .\\venv311\\Scripts\\python.exe scripts\\build_finetune_dataset.py --keywords fed inflation rates --limit 100

说明：
- 输出文件在 data/ 下，默认不会被git提交
- 生成的是“可训练模板”，建议你人工抽查/修正后再进行7B正式训练
"""

import sys
from pathlib import Path
import argparse
import hashlib
import random
from loguru import logger


# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Build weak-labeled finetune dataset")
    parser.add_argument("--out", default="data/finetune/train.json", help="输出json路径")
    parser.add_argument("--limit", type=int, default=50, help="最大新闻条数")
    parser.add_argument("--keywords", nargs="+", default=None, help="关键词过滤")
    parser.add_argument("--sources", nargs="+", default=None, help="RSS源URL列表（可选，默认内置）")
    parser.add_argument("--add-explain", action="store_true", help="同时生成解释类样本（弱标注）")
    parser.add_argument("--append", action="store_true", help="如果输出文件已存在，则追加新样本而不是覆盖")
    parser.add_argument("--dedup", action="store_true", help="合并时按输入去重（建议与--append一起使用）")
    parser.add_argument("--min-title-len", type=int, default=8, help="最短标题长度，过短则丢弃")
    parser.add_argument("--min-content-len", type=int, default=30, help="最短内容长度，过短则丢弃")
    parser.add_argument("--split-val", action="store_true", help="同时输出验证集 val.json（从合并后的样本中切分）")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例（默认 0.05）")
    parser.add_argument("--seed", type=int, default=42, help="切分随机种子")

    args = parser.parse_args()

    def conversation_key(item: dict) -> str:
        convs = item.get("conversations") or []
        user_text = ""
        for msg in convs:
            if msg.get("role") == "user":
                user_text = (msg.get("content") or "").strip()
                break
        norm = " ".join(user_text.split())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    def load_existing(out_path: Path):
        if not out_path.exists():
            return []
        try:
            import json

            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            logger.warning(f"Failed to load existing dataset for append: {e}")
            return []

    from src.data.fetcher import DataFetcher
    from src.llm.news_parser import RuleBasedNewsParser
    from src.llm.finetune.dataset import FineTuneDataset

    fetcher = DataFetcher()

    logger.info("Fetching news...")
    news_list = fetcher.fetch_news(keywords=args.keywords, sources=args.sources, limit=args.limit)
    logger.info(f"Fetched news items: {len(news_list)}")

    if not news_list:
        logger.warning("No news fetched. If you expect data, check network access or set NEWSAPI_KEY.")
        return

    parser_rb = RuleBasedNewsParser()
    ds = FineTuneDataset(data_path=str(Path(args.out).parent))

    kept = 0

    def compute_confidence(item: dict) -> float:
        # Heuristic confidence: source weight + content length
        w = float(item.get("weight", 1.0) or 1.0)
        title = (item.get("title") or "")
        content = (item.get("content") or "")
        length_score = min(1.0, (len(title) + len(content)) / 600.0)
        return max(0.0, min(1.0, 0.35 + 0.20 * w + 0.45 * length_score))

    for n in news_list:
        title = (n.get("title") or "").strip()
        content = (n.get("content") or "").strip()
        if not title and not content:
            continue

        if len(title) < args.min_title_len:
            continue
        if len(content) < args.min_content_len:
            continue

        parsed = parser_rb.parse(title=title, content=content)

        meta = {
            "source": n.get("source") or "",
            "source_id": n.get("source_id") or "",
            "category": n.get("category") or "",
            "weight": float(n.get("weight", 1.0) or 1.0),
            "published_at": n.get("published_at") or "",
            "url": n.get("url") or "",
        }
        meta["confidence"] = compute_confidence({**n, **meta})

        ds.add_news_sample(
            title=title,
            content=content,
            event_type=parsed.event_type,
            sentiment=parsed.sentiment,
            impact_equity=int(parsed.impact_equity),
            impact_bond=int(parsed.impact_bond),
            impact_gold=int(parsed.impact_gold),
            summary=parsed.summary,
            meta=meta,
        )

        if args.add_explain:
            # 弱标注解释：根据影响方向生成一个可训练的“解释/建议”模板
            actions = []
            if int(parsed.impact_equity) < 0:
                actions.append("降低股票(如SPY/QQQ)仓位")
            elif int(parsed.impact_equity) > 0:
                actions.append("增加股票(如SPY/QQQ)仓位")

            if int(parsed.impact_bond) > 0:
                actions.append("增加债券(如TLT/IEF)仓位")
            elif int(parsed.impact_bond) < 0:
                actions.append("降低债券(如TLT/IEF)仓位")

            if int(parsed.impact_gold) > 0:
                actions.append("增加黄金(如GLD)仓位")
            elif int(parsed.impact_gold) < 0:
                actions.append("降低黄金(如GLD)仓位")

            if not actions:
                actions.append("保持组合不变，等待更多确认信号")

            context = (
                f"新闻标题：{title}\n"
                f"新闻摘要：{parsed.summary}\n"
                f"事件类型：{parsed.event_type}\n"
                f"情绪：{parsed.sentiment}\n"
                f"影响：equity={int(parsed.impact_equity)}, bond={int(parsed.impact_bond)}, gold={int(parsed.impact_gold)}\n"
                "当前组合：SPY 40%, TLT 40%, GLD 20%。风险档位：balanced。\n"
                "请解释应如何调整仓位，并给出风险提示（简洁、可执行）。"
            )

            explanation = (
                f"根据该新闻事件（{parsed.event_type}）及其情绪（{parsed.sentiment}），"
                f"对资产的影响倾向为：股票{int(parsed.impact_equity)}、债券{int(parsed.impact_bond)}、黄金{int(parsed.impact_gold)}。"
                f"因此建议：{'; '.join(actions)}。"
                "同时注意控制回撤与波动，若后续价格走势与预期不一致，应及时减仓/止损并等待趋势确认。"
            )
            ds.add_explanation_sample(context=context, explanation=explanation, meta=meta)

        kept += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_data = ds.to_conversation_format()

    if args.append and out_path.exists():
        existing = load_existing(out_path)
        logger.info(f"Appending to existing dataset: {out_path} (existing={len(existing)}, new={len(new_data)})")
        merged = existing + new_data
        if args.dedup:
            seen = set()
            deduped = []
            for item in merged:
                key = conversation_key(item)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            merged = deduped
            logger.info(f"Dedup applied: total={len(merged)}")
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        final_data = merged
    else:
        ds.save(filename=out_path.name)
        final_data = new_data

    if args.split_val:
        random.seed(args.seed)
        data = list(final_data)
        random.shuffle(data)
        val_n = int(len(data) * float(args.val_ratio))
        val_n = max(1, val_n) if len(data) >= 2 else 0
        val_data = data[:val_n]
        train_data = data[val_n:]
        import json

        val_path = out_path.parent / "val.json"
        train_path = out_path.parent / out_path.name
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Split dataset: train={len(train_data)} val={len(val_data)}")

    stats = ds.get_statistics()
    logger.info(f"Dataset written: {out_path}")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
