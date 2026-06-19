"""quantai.backtest —— 向量化回测（lookahead 修复在此）。

fill_timing="next_open"（默认）= 修复版；"close" = 旧版(虚高)，仅用于复现对比。

用法：
    from quantai.backtest import run_backtest, compute_metrics
"""

from .compare import compare_fill_timings, format_comparison_markdown
from .engine import BacktestResult, run_backtest
from .metrics import PerformanceReport, compute_metrics

__all__ = [
    "run_backtest",
    "BacktestResult",
    "compute_metrics",
    "PerformanceReport",
    "compare_fill_timings",
    "format_comparison_markdown",
]
