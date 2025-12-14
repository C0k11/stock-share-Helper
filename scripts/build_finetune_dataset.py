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

    args = parser.parse_args()

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
    for n in news_list:
        title = (n.get("title") or "").strip()
        content = (n.get("content") or "").strip()
        if not title and not content:
            continue

        parsed = parser_rb.parse(title=title, content=content)

        ds.add_news_sample(
            title=title,
            content=content,
            event_type=parsed.event_type,
            sentiment=parsed.sentiment,
            impact_equity=int(parsed.impact_equity),
            impact_bond=int(parsed.impact_bond),
            impact_gold=int(parsed.impact_gold),
            summary=parsed.summary,
        )
        kept += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save(filename=out_path.name)

    stats = ds.get_statistics()
    logger.info(f"Dataset written: {out_path}")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
