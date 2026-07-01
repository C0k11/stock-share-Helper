"""薄 CLI：分析真实持仓（quantai.portfolio）。

示例：
    # 用默认 portfolio.local.yaml + SPY 基准（联网抓 yfinance 日线）
    python scripts/analyze.py

    # 指定文件 / 基准 / 历史长度
    python scripts/analyze.py --portfolio my.local.yaml --benchmark QQQ --years 3

    # 机器可读输出（接仪表盘/仓库 ETL）
    python scripts/analyze.py --json

这是个**薄入口**：加载持仓 -> 抓价格 -> `PortfolioAnalyzer` -> 打印报告。
计算全部在 `quantai/analysis` + `quantai/portfolio`（纯函数、可离线测试）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantai.config import load_config
from quantai.data.prices import PriceFetcher
from quantai.portfolio import PortfolioAnalyzer, format_snapshot_text, load_portfolio


def main(argv: list[str] | None = None) -> int:
    cfg = load_config().portfolio
    p = argparse.ArgumentParser(description="QuantAI real-portfolio analyzer")
    p.add_argument("--portfolio", default=cfg.file, help=f"持仓文件（默认 {cfg.file}）")
    p.add_argument("--benchmark", default=cfg.benchmark, help=f"基准（默认 {cfg.benchmark}）")
    p.add_argument("--years", type=int, default=cfg.history_years, help="历史回看年数")
    p.add_argument("--json", action="store_true", help="输出 JSON（供仪表盘/ETL）")
    args = p.parse_args(argv)

    portfolio = load_portfolio(args.portfolio)
    if not portfolio.positions:
        print("持仓文件为空（只有现金），没有可分析的标的。", file=sys.stderr)
        return 1

    start = (datetime.now() - timedelta(days=args.years * 365)).strftime("%Y-%m-%d")
    fetcher = PriceFetcher()
    prices = fetcher.fetch_prices(portfolio.symbols, start)
    bench = fetcher.fetch_prices([args.benchmark], start).get(args.benchmark)
    if bench is None or bench.empty:
        print(f"基准 {args.benchmark} 抓不到数据（网络/代码问题），中止。", file=sys.stderr)
        return 2

    snap = PortfolioAnalyzer(prices, bench, benchmark=args.benchmark).analyze(portfolio)
    if args.json:
        print(json.dumps(snap.as_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print(format_snapshot_text(snap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
