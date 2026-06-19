"""生成「旧(close, lookahead) vs 新(next_open, 修复)」回测对比表。

策略：SPY 日频，长/平仓 —— SignalGenerator 综合信号 > 0 时满仓，否则空仓。
两种成交时点喂同一信号、同一成本，差异即 lookahead 影响（诚实口径的对比数字）。

用法：
    python scripts/lookahead_compare.py --symbol SPY --start 2010-01-01
    python scripts/lookahead_compare.py --cache data/cache/spy_lookahead.parquet  # 复用本地数据，离线可跑
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 让脚本可独立运行：把仓库根加入 sys.path（pytest 走 pyproject 的 pythonpath，脚本没有）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantai.backtest import compare_fill_timings, format_comparison_markdown
from quantai.config import load_config
from quantai.signals import SignalGenerator

REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "backtest_old_vs_new.md"


def _load_prices(symbol: str, start: str, cache: Path | None) -> pd.DataFrame:
    if cache and cache.exists():
        print(f"[data] 读取本地缓存 {cache}")
        return pd.read_parquet(cache)
    from quantai.data import PriceFetcher  # 延迟导入，离线缓存路径不需要 yfinance

    print(f"[data] 通过 yfinance 抓取 {symbol} 自 {start} ...")
    prices = PriceFetcher().fetch_prices([symbol], start_date=start)[symbol]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache)
        print(f"[data] 已缓存到 {cache}")
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description="OLD vs NEW fill-timing backtest comparison")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--cache", type=Path, default=None, help="价格 parquet 缓存路径")
    args = parser.parse_args()

    cfg = load_config()
    cost_per_turnover = cfg.backtest.costs.commission_rate + cfg.backtest.costs.slippage_bps / 1e4

    prices = _load_prices(args.symbol, args.start, args.cache)

    signals = SignalGenerator(
        ma_short=cfg.strategy.trend.ma_short,
        ma_long=cfg.strategy.trend.ma_long,
        momentum_lookback=cfg.strategy.trend.momentum_lookback,
    ).generate(prices)
    weight = (signals["composite_signal"] > 0).astype(float)  # 长/平仓

    results = compare_fill_timings(prices, weight, cost_per_turnover=cost_per_turnover)
    table = format_comparison_markdown(
        results, title=f"{args.symbol}（综合信号长/平仓，含成本 {cost_per_turnover*1e4:.0f}bps/换手）"
    )

    print("\n" + table)
    REPORT_PATH.write_text(
        "# 回测：lookahead 修复前后对比（次日开盘成交）\n\n"
        "> 由 `scripts/lookahead_compare.py` 生成。两列用**同一信号、同一成本**，唯一差别是成交时点。\n\n"
        + table
        + "\n说明：旧版用 close[t] 决策又用 close[t] 成交（同根 K 线），系统性高估；"
        "新版 close[t] 决策、open[t+1] 成交，消除该未来函数。\n",
        encoding="utf-8",
    )
    print(f"[report] 已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()
