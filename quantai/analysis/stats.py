"""金融统计：收益分布、回撤、Sharpe/Sortino、beta vs 基准。

Sharpe / 最大回撤直接**复用** `quantai.backtest.metrics`（同一口径：rf=2%/年、
(日均×252 − rf)/(日 std×√252)、算术年化），不重复实现——分析层与回测层数字必须一致。

同 `trend.py` 的约定：纯函数、边界诚实。未定义的统计量（样本不足/分母为 0）
返回 NaN 而非编数字，各函数 docstring 标明口径。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

# 复用回测层口径（分析层与回测层的 Sharpe/回撤数字必须一致，不搞两套）。
from quantai.backtest.metrics import (  # noqa: F401  (re-export)
    TRADING_DAYS,
    max_drawdown,
    sharpe_ratio,
)


def returns_from_prices(close: pd.Series) -> pd.Series:
    """价格 → 简单日收益。r_t = close_t/close_{t-1} − 1，首位 NaN 丢弃。"""
    return close.pct_change(fill_method=None).dropna()


# --------------------------------------------------------------------------- #
# 收益分布
# --------------------------------------------------------------------------- #
@dataclass
class ReturnDistribution:
    """日收益分布画像（全部基于简单日收益）。

    - ann_return = mean_daily × 252（算术年化，与 Sharpe 分子同口径）
    - ann_volatility = std_daily × √252
    - skewness / excess_kurtosis：pandas 口径（偏度 g1；峰度为超额峰度，正态=0）
    - var_95 = 日收益 5% 分位（历史模拟法 VaR，负数表示亏损）
    - cvar_95 = 低于 var_95 的日收益均值（Expected Shortfall）
    - positive_share = 正收益日占比（>0 的天数 / 全部天数）
    """

    n_days: int
    mean_daily: float
    std_daily: float
    ann_return: float
    ann_volatility: float
    skewness: float
    excess_kurtosis: float
    var_95: float
    cvar_95: float
    best_day: float
    worst_day: float
    positive_share: float

    def as_dict(self) -> dict:
        return asdict(self)


def return_distribution(returns: pd.Series) -> ReturnDistribution:
    """日收益分布统计。见 :class:`ReturnDistribution` 各字段口径。

    边界：空序列 → n_days=0，其余全 NaN；skew/kurt 分别需要 >=3 / >=4 个样本，
    不足时 pandas 给 NaN（保留，不编数字）。NaN 值先剔除再统计。
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        nan = float("nan")
        return ReturnDistribution(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    var_95 = float(r.quantile(0.05))
    tail = r[r <= var_95]
    return ReturnDistribution(
        n_days=n,
        mean_daily=float(r.mean()),
        std_daily=float(r.std()),
        ann_return=float(r.mean() * TRADING_DAYS),
        ann_volatility=float(r.std() * np.sqrt(TRADING_DAYS)),
        skewness=float(r.skew()),
        excess_kurtosis=float(r.kurt()),
        var_95=var_95,
        cvar_95=float(tail.mean()) if len(tail) else float("nan"),
        best_day=float(r.max()),
        worst_day=float(r.min()),
        positive_share=float((r > 0).sum() / n),
    )


# --------------------------------------------------------------------------- #
# 回撤
# --------------------------------------------------------------------------- #
def drawdown_curve(equity: pd.Series) -> pd.Series:
    """回撤曲线。dd_t = equity_t / max(equity[..t]) − 1（<= 0）。"""
    peak = equity.expanding().max()
    return (equity / peak - 1.0).rename("drawdown")


@dataclass
class DrawdownStats:
    """最大回撤画像。

    - max_drawdown：最深回撤（负数，与 `backtest.metrics.max_drawdown` 同口径）。
    - peak_date / trough_date：最深回撤对应的前高日 / 谷底日。
    - longest_underwater_days：最长水下期（相邻两次创新高之间的**交易日数**；
      末段未收复的水下期也计入）。
    """

    max_drawdown: float
    peak_date: str
    trough_date: str
    longest_underwater_days: int

    def as_dict(self) -> dict:
        return asdict(self)


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    """最大回撤 + 发生区间 + 最长水下期。空序列 → NaN/空串/0。"""
    eq = equity.dropna()
    if len(eq) == 0:
        return DrawdownStats(float("nan"), "", "", 0)
    dd = drawdown_curve(eq)
    trough = dd.idxmin()
    peak_region = eq.loc[:trough]
    peak = peak_region.idxmax()
    at_high = dd == 0
    # 每个位置距最近一次创新高的天数；水下期 = 连续未创新高段的长度。
    groups = at_high.cumsum()
    underwater = (~at_high).groupby(groups).cumsum()
    return DrawdownStats(
        max_drawdown=float(dd.min()),
        peak_date=str(peak),
        trough_date=str(trough),
        longest_underwater_days=int(underwater.max()) if len(underwater) else 0,
    )


# --------------------------------------------------------------------------- #
# 风险调整收益
# --------------------------------------------------------------------------- #
def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Sortino 比率（只惩罚下行波动）。

    excess_t = r_t − rf/252（日超额收益，MAR = 无风险利率）；
    downside_dev = √(mean(min(excess, 0)²)) × √252（下行偏差：整段样本上、
    只取负超额的均方根——**分母对全样本取均值**，不是只对负样本，这是
    Sortino 的标准口径，否则会低估下行风险）；
    Sortino = (mean(excess) × 252) / downside_dev。

    边界：无下行样本（一路超越 rf）→ 分母 0，数学未定义 → 返回 NaN（不像
    `sharpe_ratio` 返回 0——那是旧口径兼容；这里诚实标 NaN，测试固定该行为）。
    空序列 → NaN。rf 与 `sharpe_ratio` 默认一致（2%/年）。
    """
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))
    if downside_dev == 0 or np.isnan(downside_dev):
        return float("nan")
    return float(excess.mean() * periods_per_year / downside_dev)


# --------------------------------------------------------------------------- #
# beta
# --------------------------------------------------------------------------- #
def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """CAPM beta。β = Cov(r_a, r_b) / Var(r_b)（样本协方差/方差，ddof=1）。

    索引按交集对齐后剔除任一侧 NaN 的行。边界：有效样本 < 2 或基准方差为 0
    → 未定义，返回 NaN。
    """
    a, b = asset_returns.align(benchmark_returns, join="inner")
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    var_b = float(b.var())
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    return float(a.cov(b) / var_b)


def rolling_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60
) -> pd.Series:
    """滚动 beta。β_t = Cov(r_a, r_b)[t-w+1..t] / Var(r_b)[t-w+1..t]。

    索引交集对齐；基准窗口方差为 0 → 该点 NaN。前 window-1 个位置 NaN。
    """
    if window < 2:
        raise ValueError(f"window 需 >= 2，收到 {window}")
    a, b = asset_returns.align(benchmark_returns, join="inner")
    cov = a.rolling(window).cov(b)
    var_b = b.rolling(window).var()
    return (cov / var_b.where(var_b > 0)).rename(f"rolling_beta_{window}")
