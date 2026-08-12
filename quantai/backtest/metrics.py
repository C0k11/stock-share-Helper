"""绩效指标：总收益 / 年化(CAGR) / 年化波动 / Sharpe / 最大回撤 / 胜率。

从旧 `src/backtest/metrics.py` 精简迁移（保留口径），加类型与 dataclass 报告。
口径与旧版一致：CAGR 用 days/252 折算年数；Sharpe = (mean*252 - rf)/(std*sqrt(252))；
胜率 = 正收益日占比（全样本日）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    start: str
    end: str
    trading_days: int

    def as_dict(self) -> dict:
        return asdict(self)


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annual_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    vol = returns.std() * np.sqrt(periods_per_year)
    if vol == 0 or np.isnan(vol):
        return 0.0
    excess = returns.mean() * periods_per_year - risk_free_rate
    return float(excess / vol)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    return float(((equity - peak) / peak).min())


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_metrics(
    equity: pd.Series,
    *,
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceReport:
    """从净值曲线算出整套绩效指标。"""
    returns = equity.pct_change(fill_method=None).dropna()
    return PerformanceReport(
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year),
        annual_volatility=annual_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate(returns),
        start=str(equity.index[0]),
        end=str(equity.index[-1]),
        trading_days=len(equity),
    )
