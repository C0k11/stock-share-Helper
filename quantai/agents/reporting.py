"""报告编排：日报 / 盘中快报的完整流程（CLI 与 streamlit 驻留模型共用）。

从 `scripts/report.py` 抽出——同一份流程两处跑：
- CLI：`python scripts/report.py [--intraday] [--llm]`（每次冷加载模型）
- streamlit AI 分析页：`st.cache_resource` 驻留 LocalLLM，秒级复用

联网/GPU 都在这里发生；纯逻辑（简报组装/打分/统计）在 analyst / news_scorer，
已单测。本模块只做接线，保持薄。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from quantai.agents.analyst import (
    build_brief,
    build_intraday_brief,
    generate_commentary,
    generate_intraday_commentary,
    intraday_stats,
)
from quantai.agents.news_scorer import aggregate_symbol_sentiment, score_news
from quantai.config import load_config

_WAREHOUSE_DB = Path("data/warehouse/quantai.duckdb")


def load_report_llm():
    """加载本地 LLM 并放开报告场景的生成上限（GPU；调用方自行缓存/驻留）。"""
    from quantai.llm.inference import LocalLLM

    llm = LocalLLM.from_config(load_config().llm)
    llm.gen_max_time_sec = 240.0
    llm.max_new_tokens = 2500
    return llm


def _persist_scores(scored: list[dict], model: str) -> int:
    """情绪分入仓库（库不存在/被占用时如实跳过，不炸报告主流程）。"""
    if not scored or not _WAREHOUSE_DB.exists():
        return 0
    try:
        from quantai.warehouse import connect, load_news, load_news_scores

        con = connect(_WAREHOUSE_DB)
        try:
            load_news(con, [r["item"] for r in scored])  # 头条可能还没入库（盘中新抓的）
            return load_news_scores(con, scored, model=model)
        finally:
            con.close()
    except Exception:  # noqa: BLE001 - 单写者锁冲突等：跳过持久化，报告照出
        return 0


def make_daily_report(
    llm=None, out_dir: str = "data/reports", log: Callable[[str], None] = print
) -> tuple[str, Path]:
    """日报（收盘口径）：持仓+自选股+新闻+仓库摘要 →（可选 LLM）→ 落盘。"""
    from quantai.analysis import realized_volatility, rsi
    from quantai.data.news import NewsFetcher
    from quantai.data.prices import PriceFetcher
    from quantai.data.watchlist import load_watchlist
    from quantai.portfolio import PortfolioAnalyzer, load_portfolio

    cfg = load_config()
    portfolio = load_portfolio(cfg.portfolio.file)
    watchlist = load_watchlist(cfg.portfolio.watchlist_file)
    start = (datetime.now() - timedelta(days=cfg.portfolio.history_years * 365)).strftime("%Y-%m-%d")
    symbols = list(dict.fromkeys(portfolio.symbols + watchlist + [cfg.portfolio.benchmark]))

    log(f"[report] fetching {len(symbols)} symbols ...")
    prices = PriceFetcher().fetch_prices(symbols, start)
    bench = prices.get(cfg.portfolio.benchmark)
    if bench is None or bench.empty:
        raise RuntimeError(f"基准 {cfg.portfolio.benchmark} 抓不到数据")
    snap = PortfolioAnalyzer(prices, bench, benchmark=cfg.portfolio.benchmark).analyze(portfolio)

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

    news_fetcher = NewsFetcher(extra_feeds=cfg.data.news_feeds)
    news = {s: news_fetcher.fetch_symbol_news(s, limit=3) for s in (portfolio.symbols + watchlist[:10])}

    # Polymarket 事件概率（公开 API；tag 列表 config 驱动，空列表=关闭）
    event_rows: list[dict] = []
    if cfg.data.event_tags:
        from quantai.data.events import EventsFetcher

        odds = EventsFetcher().fetch_all(cfg.data.event_tags, limit_per_tag=15)
        event_rows = [o.as_dict() for o in odds]

    as_of = datetime.now().strftime("%Y-%m-%d")
    con = None
    if _WAREHOUSE_DB.exists():
        from quantai.warehouse import connect, load_event_odds

        con = connect(_WAREHOUSE_DB)
        if event_rows:
            try:
                from quantai.data.events import EventOdd

                load_event_odds(con, [EventOdd(**r) for r in event_rows], as_of=as_of)
            except Exception:  # noqa: BLE001 - 单写者冲突等：跳过持久化不炸报告
                pass
    brief = build_brief(snap, watch_rows, news, warehouse_con=con, as_of=as_of, event_rows=event_rows)
    if con is not None:
        con.close()

    report = brief
    if llm is not None:
        log("[report] generating LLM commentary ...")
        model_name = getattr(llm, "model_name", "local")
        # 顺带把持仓+近期新闻打分入库（情绪时间线数据源）
        flat_news = [it for items in news.values() for it in items]
        scored = score_news(flat_news, llm)
        n = _persist_scores(scored, model=str(model_name))
        log(f"[report] news scored & persisted: {n}")
        commentary = generate_commentary(brief, llm)
        report = f"{brief}\n\n---\n\n# LLM 财经分析（本地 {model_name}）\n\n{commentary}"

    out = Path(out_dir) / f"report_{as_of}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return report, out


def make_intraday_report(
    llm=None,
    symbols_cap: int = 8,
    out_dir: str = "data/reports",
    log: Callable[[str], None] = print,
) -> tuple[str, Path]:
    """盘中快报：当日 1 分钟会话统计 + 卖压代理 →（可选 LLM 打分+快评）→ 落盘。"""
    import yfinance as yf

    from quantai.data.news import NewsFetcher
    from quantai.data.watchlist import load_watchlist
    from quantai.portfolio import load_portfolio

    cfg = load_config()
    portfolio = load_portfolio(cfg.portfolio.file)
    watch = load_watchlist(cfg.portfolio.watchlist_file)
    symbols = list(dict.fromkeys(portfolio.symbols + watch[:symbols_cap]))
    log(f"[intraday] {len(symbols)} symbols, fetching 1m session bars ...")

    rows = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            m1 = t.history(period="1d", interval="1m")
            d5 = t.history(period="10d", interval="1d")
            if m1.empty or len(d5) < 2:
                continue
            m1.columns = [c.lower() for c in m1.columns]
            prev_close = float(d5["Close"].iloc[-2])
            avg_vol = float(d5["Volume"].iloc[:-1].mean())
            stats = intraday_stats(m1, prev_close, avg_vol)
            if stats:
                rows.append({"symbol": sym, **stats})
        except Exception as exc:  # noqa: BLE001 - 单标的失败不炸整批
            log(f"  skip {sym}: {exc}")

    scored, agg = None, None
    model_name = getattr(llm, "model_name", "local") if llm is not None else ""
    if llm is not None:
        news = NewsFetcher().fetch_all(symbols, limit_per_symbol=3)
        scored = score_news(news, llm)
        agg = aggregate_symbol_sentiment(scored)
        n = _persist_scores(scored, model=str(model_name))
        log(f"[intraday] news scored & persisted: {n}")

    as_of = datetime.now().strftime("%Y-%m-%d %H:%M")
    brief = build_intraday_brief(rows, scored, agg, as_of=as_of)
    report = brief
    if llm is not None:
        report = (
            f"{brief}\n\n---\n\n# LLM 盘中快评（本地 {model_name}）\n\n"
            + generate_intraday_commentary(brief, llm)
        )

    out = Path(out_dir) / f"intraday_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return report, out