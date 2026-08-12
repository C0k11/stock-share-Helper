"""OLD(close, lookahead) vs NEW(next_open, 修复) 对比工具。

把同一份价格 + 同一个目标仓位，分别用两种成交时点回测，并排出对比表——
用来量化"修掉 lookahead 后历史业绩降了多少"（给出诚实口径的对比数字）。
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestResult, FillTiming, run_backtest


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


def buy_and_hold(
    prices: pd.DataFrame,
    *,
    cost_per_turnover: float = 0.0,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.02,
    fill_timing: FillTiming = "next_open",
) -> BacktestResult:
    """基准：首日满仓买入并一直持有（“直接拿着大盘不动”）。

    跑同一个引擎 / 同一份成本模型，权重恒为 1（只在首日产生一次建仓换手），
    唯一区别是策略会调仓、它不调——因此是与策略 apples-to-apples 的市场对标。
    """
    weight = pd.Series(1.0, index=prices.index)
    return run_backtest(
        prices,
        weight,
        fill_timing=fill_timing,
        cost_per_turnover=cost_per_turnover,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _pp(x: float) -> str:
    """差值按百分点(pp)带符号展示，区别于绝对水平的 `_pct`。"""
    return f"{x * 100:+.2f}pp"


_SHARPE_NOTE = (
    "> Sharpe 口径：无风险利率 rf=2.0%/年，(日均收益×252 - rf) ÷ (日收益标准差×sqrt(252))，算术均值年化。\n"
    "> Buy&Hold = 首日满仓 SPY 持有到底（同一引擎/成本，next_open 成交）；「新版 - B&H」为策略相对大盘的诚实差距"
    "(pp；最大回撤为负数，差值正值=策略回撤更浅、风险更低)。\n"
)


def format_comparison_markdown(
    results: dict[str, BacktestResult],
    *,
    benchmark: BacktestResult | None = None,
    title: str = "",
) -> str:
    """把对比结果排成 Markdown 表。

    benchmark 为 None：旧 vs 新 + 变化列（3 列对比）。
    benchmark 给定：再并入「Buy&Hold SPY」列与「新版 - B&H」诚实差距列（市场对标）。
    """
    old = results["close"].metrics
    new = results["next_open"].metrics

    header = f"### {title} · 旧(close, 有 lookahead) vs 新(next_open, 修复)\n\n" if title else ""
    period = f"期间 {new.start[:10]} ~ {new.end[:10]}，{new.trading_days} 个交易日\n"

    if benchmark is None:
        lines = [
            header + period,
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

    bh = benchmark.metrics
    lines = [
        header + period,
        "| 指标 | 旧版(close,虚高) | 新版(next_open,诚实) | Buy&Hold SPY | 新版 - B&H |",
        "| :-- | --: | --: | --: | --: |",
        f"| 总收益 | {_pct(old.total_return)} | {_pct(new.total_return)} | {_pct(bh.total_return)} | {_pp(new.total_return - bh.total_return)} |",
        f"| 年化(CAGR) | {_pct(old.cagr)} | {_pct(new.cagr)} | {_pct(bh.cagr)} | {_pp(new.cagr - bh.cagr)} |",
        f"| 年化波动 | {_pct(old.annual_volatility)} | {_pct(new.annual_volatility)} | {_pct(bh.annual_volatility)} | {_pp(new.annual_volatility - bh.annual_volatility)} |",
        f"| Sharpe | {old.sharpe:.2f} | {new.sharpe:.2f} | {bh.sharpe:.2f} | {new.sharpe - bh.sharpe:+.2f} |",
        f"| 最大回撤 | {_pct(old.max_drawdown)} | {_pct(new.max_drawdown)} | {_pct(bh.max_drawdown)} | {_pp(new.max_drawdown - bh.max_drawdown)} |",
        f"| 胜率(正收益日) | {_pct(old.win_rate)} | {_pct(new.win_rate)} | {_pct(bh.win_rate)} | {_pp(new.win_rate - bh.win_rate)} |",
    ]
    return "\n".join(lines) + "\n\n" + _SHARPE_NOTE
