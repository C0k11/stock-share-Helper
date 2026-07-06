"""薄 CLI：数据仓库编排（ETL 装载 → dbt build → Tableau 导出）。

示例：
    # 端到端：抓真实行情 + 加载持仓/信号/回测 + dbt build + 导出 Tableau 数据源
    python scripts/warehouse.py --full --portfolio portfolio.example.yaml

    # 只重跑 SQL 转换层（raw 不动）
    python scripts/warehouse.py --dbt

    # 只导出 marts 层给 Tableau（CSV，见 tableau/DASHBOARD_SPEC.md）
    python scripts/warehouse.py --export

计算路径：yfinance/持仓/信号/回测 →(pandas ETL)→ raw →(dbt SQL)→ staging → marts
→(export)→ tableau/exports/*.csv 或 Tableau 直连 DuckDB（连接方式见 SPEC）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_REPO = Path(__file__).resolve().parent.parent
_DBT_DIR = _REPO / "warehouse"
_DEFAULT_DB = _REPO / "data" / "warehouse" / "quantai.duckdb"
_EXPORT_DIR = _REPO / "tableau" / "exports"

_MARTS = (
    "dim_date", "dim_symbol", "fact_prices", "fact_positions",
    "fact_trades", "fact_signals", "fact_backtest_results", "fact_backtest_equity",
    "fact_news",
)


def _load_raw(db_path: Path, portfolio_file: str, years: int, benchmark: str) -> None:
    from quantai.backtest import run_backtest
    from quantai.config import load_config
    from quantai.data.news import NewsFetcher
    from quantai.data.prices import PriceFetcher
    from quantai.portfolio import load_portfolio
    from quantai.signals.generator import SignalGenerator
    from quantai.warehouse import (
        connect, load_backtest, load_news, load_positions, load_prices,
        load_signals, load_trading_days,
    )
    from quantai.warehouse.etl import init_raw_tables
    import pandas as pd

    from quantai.data.watchlist import load_watchlist

    portfolio = load_portfolio(portfolio_file)
    watchlist = load_watchlist(load_config().portfolio.watchlist_file)
    symbols = list(dict.fromkeys(portfolio.symbols + watchlist + [benchmark]))
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    print(f"[etl] fetching {symbols} since {start} ...")
    prices = PriceFetcher().fetch_prices(symbols, start)
    con = connect(db_path)
    try:
        init_raw_tables(con)  # 全部 raw 表建齐（含还没有数据的，如 trades）——dbt 才能全模型编译
        n = load_prices(con, prices)
        print(f"[etl] raw.prices          +{n}")
        n = load_trading_days(con, start, end)
        print(f"[etl] raw.trading_days    +{n}")
        as_of = max(str(df.index[-1].date()) for df in prices.values()) if prices else end
        n = load_positions(con, portfolio, as_of=as_of)
        print(f"[etl] raw.positions       +{n} (as_of {as_of})")
        gen = SignalGenerator()
        for sym, df in prices.items():
            load_signals(con, sym, gen.generate(df))
        print(f"[etl] raw.signals         {len(prices)} symbols")
        news = NewsFetcher(extra_feeds=load_config().data.news_feeds).fetch_all(symbols)
        n = load_news(con, news)
        print(f"[etl] raw.news            +{n} new items ({len(news)} fetched)")
        if benchmark in prices:
            df = prices[benchmark]
            weight = pd.Series(0.6, index=df.index)  # demo：60% 恒权重基准回测
            result = run_backtest(df, weight)
            load_backtest(con, f"const60_{benchmark}", result, strategy="const_weight_0.6", symbol=benchmark)
            print(f"[etl] raw.backtest_*      run=const60_{benchmark} (sharpe {result.metrics.sharpe:.2f})")
    finally:
        con.close()


def _run_dbt(db_path: Path) -> None:
    env = {**os.environ, "QUANTAI_DB_PATH": str(db_path)}
    dbt_exe = Path(sys.executable).parent / ("dbt.exe" if os.name == "nt" else "dbt")
    cmd = [str(dbt_exe), "build", "--project-dir", str(_DBT_DIR), "--profiles-dir", str(_DBT_DIR)]
    print(f"[dbt] {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise SystemExit(f"dbt build 失败（exit {res.returncode}）")


def _export(db_path: Path) -> None:
    from quantai.warehouse import connect

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    try:
        for t in _MARTS:
            out = _EXPORT_DIR / f"{t}.csv"
            path_sql = out.as_posix().replace("'", "''")  # 路径含单引号时转义（SQL 字面量）
            con.execute(
                f"COPY (SELECT * FROM marts.{t}) TO '{path_sql}' (HEADER, DELIMITER ',')"
            )
            n = con.execute(f"SELECT count(*) FROM marts.{t}").fetchone()[0]
            print(f"[export] {out.name:<28} {n:>8} rows")
    finally:
        con.close()
    print(f"[export] done -> {_EXPORT_DIR}（Tableau: Connect > Text file，或直连 DuckDB，见 tableau/DASHBOARD_SPEC.md）")


def main(argv: list[str] | None = None) -> int:
    from quantai.config import load_config

    # 默认值统一从 PortfolioConfig 取（与 scripts/analyze.py 同源）——否则用户在
    # config/env 覆盖 portfolio.file/benchmark 后，两个 CLI 会静默分析不同的组合。
    cfg = load_config().portfolio
    p = argparse.ArgumentParser(description="QuantAI warehouse orchestrator")
    p.add_argument("--db", default=str(_DEFAULT_DB), help="DuckDB 文件路径")
    p.add_argument("--portfolio", default=cfg.file, help=f"持仓文件（默认 {cfg.file}）")
    p.add_argument("--benchmark", default=cfg.benchmark)
    p.add_argument("--years", type=int, default=cfg.history_years)
    p.add_argument("--load", action="store_true", help="pandas ETL 装载 raw 层")
    p.add_argument("--dbt", action="store_true", help="跑 dbt build（staging+marts+tests）")
    p.add_argument("--export", action="store_true", help="导出 marts -> tableau/exports/*.csv")
    p.add_argument("--full", action="store_true", help="load + dbt + export 一条龙")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    if args.full:
        args.load = args.dbt = args.export = True
    if not (args.load or args.dbt or args.export):
        p.print_help()
        return 1
    if args.load:
        _load_raw(db_path, args.portfolio, args.years, args.benchmark)
    if args.dbt:
        _run_dbt(db_path)
    if args.export:
        _export(db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
