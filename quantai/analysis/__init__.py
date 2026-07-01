"""quantai.analysis — 技术指标 + 金融统计引擎（纯 pandas/numpy 函数层）。

三个子模块：
- `trend`：MACD（金叉/死叉）、均线体系、ROC/动量、RSI（超卖）、随机指标、回踩检测。
- `volatility`：realized/annualized 波动率、Bollinger、ATR、rolling 相关性。
- `stats`：收益分布、回撤画像、Sharpe（复用 backtest.metrics）/Sortino、beta。

全层纯函数、因果窗口（无 lookahead，截断不变性测试保证）、边界诚实（数据不足
给 NaN 不报错、未定义不编数字）。文档见 `docs/modules/analysis.md`。
"""

from quantai.analysis.stats import (
    DrawdownStats,
    ReturnDistribution,
    beta,
    drawdown_curve,
    drawdown_stats,
    max_drawdown,
    return_distribution,
    returns_from_prices,
    rolling_beta,
    sharpe_ratio,
    sortino_ratio,
)
from quantai.analysis.trend import (
    ema,
    macd,
    macd_cross,
    momentum,
    moving_average_system,
    pullback,
    roc,
    rsi,
    rsi_signals,
    sma,
    stochastic,
)
from quantai.analysis.volatility import (
    atr,
    bollinger,
    realized_volatility,
    rolling_correlation,
)

__all__ = [
    # trend
    "sma", "ema", "moving_average_system", "macd", "macd_cross",
    "roc", "momentum", "rsi", "rsi_signals", "stochastic", "pullback",
    # volatility
    "realized_volatility", "bollinger", "atr", "rolling_correlation",
    # stats
    "returns_from_prices", "return_distribution", "ReturnDistribution",
    "drawdown_curve", "drawdown_stats", "DrawdownStats",
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "beta", "rolling_beta",
]
