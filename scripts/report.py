"""薄 CLI：每日分析简报（真实持仓 + 自选股 + 新闻 + 仓库 → LLM 财经分析）。

示例：
    # 只出数据简报（无 GPU、秒级）
    python scripts/report.py

    # 简报 + 本地 LLM 财经分析（加载 Qwen，需 GPU，首次加载约 1-2 分钟）
    python scripts/report.py --llm

产物：data/reports/report_<date>.md（简报 + 可选 LLM 分析），同时打印到终端。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantai.config import load_config


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="QuantAI daily analyst report")
    p.add_argument("--portfolio", default=cfg.portfolio.file)
    p.add_argument("--benchmark", default=cfg.portfolio.benchmark)
    p.add_argument("--llm", action="store_true", help="加载本地 LLM 生成财经分析（GPU）")
    p.add_argument("--out-dir", default="data/reports")
    args = p.parse_args(argv)

    from quantai.agents.analyst import build_brief, generate_commentary
    from quantai.analysis import realized_volatility, rsi
    from quantai.data.news import NewsFetcher
    from quantai.data.prices import PriceFetcher
    from quantai.data.watchlist import load_watchlist
    from quantai.portfolio import PortfolioAnalyzer, load_portfolio

    portfolio = load_portfolio(args.portfolio)
    watchlist = load_watchlist(cfg.portfolio.watchlist_file)
    start = (datetime.now() - timedelta(days=cfg.portfolio.history_years * 365)).strftime("%Y-%m-%d")
    symbols = list(dict.fromkeys(portfolio.symbols + watchlist + [args.benchmark]))

    print(f"[report] fetching {len(symbols)} symbols ...")
    fetcher = PriceFetcher()
    prices = fetcher.fetch_prices(symbols, start)
    bench = prices.get(args.benchmark)
    if bench is None or bench.empty:
        print(f"基准 {args.benchmark} 抓不到数据，中止。", file=sys.stderr)
        return 2
    snap = PortfolioAnalyzer(prices, bench, benchmark=args.benchmark).analyze(portfolio)

    # 自选股行情/指标行
    watch_rows = []
    for sym in watchlist:
        df = prices.get(sym)
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = df["close"].astype(float).dropna()
        if close.empty:
            continue
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else float("nan")
        watch_rows.append(
            {
                "symbol": sym,
                "last": last,
                "day_pct": (last / prev - 1) if prev == prev else float("nan"),
                "rsi": float(rsi(close, 14).iloc[-1]) if len(close) > 14 else float("nan"),
                "vol": float(realized_volatility(close, 20).iloc[-1]) if len(close) > 20 else float("nan"),
            }
        )

    # 新闻：持仓 + 自选股前 10（控制请求量）
    news_fetcher = NewsFetcher(extra_feeds=cfg.data.news_feeds)
    news = {s: news_fetcher.fetch_symbol_news(s, limit=3) for s in (portfolio.symbols + watchlist[:10])}

    # 仓库连接（存在才接）
    con = None
    db = Path("data/warehouse/quantai.duckdb")
    if db.exists():
        from quantai.warehouse import connect

        con = connect(db)

    as_of = datetime.now().strftime("%Y-%m-%d")
    brief = build_brief(snap, watch_rows, news, warehouse_con=con, as_of=as_of)
    if con is not None:
        con.close()

    report = brief
    if args.llm:
        print("[report] loading local LLM (GPU) ...")
        from quantai.llm.inference import LocalLLM

        llm = LocalLLM.from_config(cfg.llm)
        # 报告场景放开生成上限：config 默认 gen_max_time_sec=12s 是桌面对话的
        # 防卡死值，长报告会被腰斩（实测只出了第一段）；离线报告不赶时间。
        llm.gen_max_time_sec = 240.0
        llm.max_new_tokens = 2500  # 1500 实测在【今日关注】段中途截断
        commentary = generate_commentary(brief, llm)
        report = f"{brief}\n\n---\n\n# LLM 财经分析（本地 {cfg.llm.model_name}）\n\n{commentary}"

    out = Path(args.out_dir) / f"report_{as_of}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
