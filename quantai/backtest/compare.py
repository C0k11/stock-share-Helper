"""OLD(close, lookahead) vs NEW(next_open, 修复) 对比工具。

把同一份价格 + 同一个目标仓位，分别用两种成交时点回测，并排出对比表——
用来量化"修掉 lookahead 后历史业绩降了多少"（诚实口径的对比数字）。
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestResult, run_backtest


def compare_fill_timings(
    prices: pd.DataFrame,
    weight: pd.Series,
    *,
    cost_per_turnover: float = 0.0,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.02,
) -> dict[str, BacktestResult]:
    """对同一策略跑 close(旧) 与 next_open(新) 两种成交时点。"""
    kwargs = dict(
        cost_per_turnover=cost_per_turnover,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
    )
    return {
        "close": run_backtest(prices, weight, fill_timing="close", **kwargs),
        "next_open": run_backtest(prices, weight, fill_timing="next_open", **kwargs),
    }


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def format_comparison_markdown(results: dict[str, BacktestResult], *, title: str = "") -> str:
    """把对比结果排成 Markdown 表（含每个指标的变化）。"""
    old = results["close"].metrics
    new = results["next_open"].metrics

    header = f"### {title} · 旧(close, 有 lookahead) vs 新(next_open, 修复)\n\n" if title else ""
    lines = [
        header + f"期间 {new.start[:10]} ~ {new.end[:10]}，{new.trading_days} 个交易日\n",
        "| 指标 | 旧版(虚高) | 新版(诚实) | 变化 |",
        "| :-- | --: | --: | --: |",
        f"| 总收益 | {_pct(old.total_return)} | {_pct(new.total_return)} | {_pct(new.total_return - old.total_return)} |",
        f"| 年化(CAGR) | {_pct(old.cagr)} | {_pct(new.cagr)} | {_pct(new.cagr - old.cagr)} |",
        f"| 年化波动 | {_pct(old.annual_volatility)} | {_pct(new.annual_volatility)} | {_pct(new.annual_volatility - old.annual_volatility)} |",
        f"| Sharpe | {old.sharpe:.2f} | {new.sharpe:.2f} | {new.sharpe - old.sharpe:+.2f} |",
        f"| 最大回撤 | {_pct(old.max_drawdown)} | {_pct(new.max_drawdown)} | {_pct(new.max_drawdown - old.max_drawdown)} |",
        f"| 胜率(正收益日) | {_pct(old.win_rate)} | {_pct(new.win_rate)} | {_pct(new.win_rate - old.win_rate)} |",
    ]
    return "\n".join(lines) + "\n"
