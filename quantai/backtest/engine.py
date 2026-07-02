"""向量化单资产回测引擎，按 `fill_timing` 切换成交时点。

核心（lookahead 修复就在这里）：
- 输入 `weight[t]`：在 **close[t]** 用 ≤t 的信息**决策**出的目标仓位。
- fill_timing="close"（旧 / 有 lookahead）：让 `weight[t]` 直接吃下 close[t-1]→close[t] 的收益，
  等于"看到 close[t] 才定的仓位，却又用它赚了截至 close[t] 的钱" —— 历史业绩**虚高**的根源。
- fill_timing="next_open"（新 / 修复）：close[t] 的决策在 **open[t+1]** 成交。当日收益拆成
  隔夜(close[t-1]→open[t]) 持上上次决策 + 日内(open[t]→close[t]) 持昨日决策，全部只用过去仓位。

两种模式喂的是**同一个 weight、同一份成本**，因此对比表里的差异 = 纯 lookahead 影响（+成本时点的次要项）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .metrics import PerformanceReport, compute_metrics

FillTiming = Literal["close", "next_open"]


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    metrics: PerformanceReport
    fill_timing: str
    total_turnover: float


def _turnover(weight: pd.Series) -> pd.Series:
    """每次决策的换手 |Δweight|，首日记为 |weight[0]|（建仓）。"""
    turn = weight.diff().abs()
    turn.iloc[0] = abs(float(weight.iloc[0]))
    return turn.fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    weight: pd.Series,
    *,
    fill_timing: FillTiming = "next_open",
    cost_per_turnover: float = 0.0,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.02,
) -> BacktestResult:
    """对单资产价格 + 目标仓位序列做向量化回测。

    Args:
        prices: 含 `open`/`close` 列、按交易日索引的 DataFrame。
        weight: 与 prices 对齐的目标仓位（close[t] 决策值），如 0/1 或 [-1,1]。
        fill_timing: "close"=旧(lookahead) / "next_open"=新(修复，默认)。
        cost_per_turnover: 每单位换手的成本率（≈ 佣金率 + 滑点率）。
    """
    if fill_timing not in ("close", "next_open"):
        raise ValueError(f"fill_timing must be 'close' or 'next_open', got {fill_timing!r}")
    if not {"open", "close"}.issubset(prices.columns):
        raise ValueError("prices 需含 'open' 和 'close' 列")

    close = prices["close"]
    open_ = prices["open"]
    weight = weight.reindex(close.index).astype(float).fillna(0.0)
    turnover = _turnover(weight)

    if fill_timing == "close":
        # 旧：同根 K 线决策 + 成交（lookahead）
        cc = close.pct_change(fill_method=None)
        gross = weight * cc
        cost = turnover * cost_per_turnover
    else:
        # 新：close[t] 决策 -> open[t+1] 成交（无 lookahead）
        # 当日按 open 边界拆两段，持仓不同；收益**复利**而非相加：
        #   隔夜 close[t-1]->open[t] 持上上次决策 w[t-2]；日内 open[t]->close[t] 持昨日决策 w[t-1]。
        overnight = open_ / close.shift(1) - 1            # close[t-1] -> open[t]
        intraday = close / open_ - 1                       # open[t]   -> close[t]
        gross = (1 + weight.shift(2) * overnight) * (1 + weight.shift(1) * intraday) - 1
        # 换手在成交日(open[t+1])扣费 -> 顺延一天
        cost = turnover.shift(1).fillna(0.0) * cost_per_turnover

    strat = gross.fillna(0.0) - cost.fillna(0.0)
    equity = (1.0 + strat).cumprod() * initial_capital

    return BacktestResult(
        equity=equity,
        returns=strat,
        metrics=compute_metrics(equity, risk_free_rate=risk_free_rate),
        fill_timing=fill_timing,
        total_turnover=float(turnover.sum()),
    )
