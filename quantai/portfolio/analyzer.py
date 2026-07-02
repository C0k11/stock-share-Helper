"""组合分析器：真实持仓 × 价格历史 → 盈亏 / 暴露 / 风险 / 技术指标快照。

**可测性**：`PortfolioAnalyzer` 只吃注入的价格数据（`dict[symbol, DataFrame]`），
不碰网络；联网抓数在 `scripts/analyze.py` / api 层用 `PriceFetcher` 完成后注入。

**诚实口径**（重要，docstring 逐条对应字段）：
- 我们只有「当前持仓 + 每股成本」，**没有完整交易流水**，因此：
  - 未实现盈亏 = (最新价 − 加权平均成本) × 股数 —— 这是准确的。
  - 组合风险统计（波动/Sharpe/Sortino/回撤/beta）基于**当前持仓固定不变**的
    合成历史（industry 标准的 holdings-based analysis），字段名带 `current_holdings_`
    前缀明示假设；它不是你的真实历史 PnL 曲线。
- 缺价的标的**不静默丢弃**：列入 `missing_prices` 并从持仓与统计中**整体剔除**
  （不产生持仓行、不进任何总计；报告顶部醒目列出）。close 列存在但**全 NaN**
  同样按缺价处理（yfinance 退市/坏抓取的常见形状，审计确认后收紧）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from quantai.analysis import (
    beta,
    drawdown_curve,
    macd,
    pullback,
    realized_volatility,
    returns_from_prices,
    rsi,
    sharpe_ratio,
    sortino_ratio,
)
from quantai.portfolio.loader import Portfolio


@dataclass
class PositionSnapshot:
    """单标的快照（多 lot 已聚合：股数求和、成本加权平均、建仓日取最早）。"""

    symbol: str
    shares: float
    avg_cost: float
    open_date: str
    last_price: float
    market_value: float
    cost_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float  # 占组合总值（含现金）比重
    day_change_pct: float  # 最新收盘 vs 前收盘
    # 技术指标快照（analysis/ 引擎，最新值）
    rsi_14: float
    macd_histogram: float
    in_uptrend: bool
    is_pullback: bool
    realized_vol_20: float  # 年化
    beta_vs_benchmark: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    """组合级快照。风险统计基于「当前持仓固定」合成历史（见模块 docstring）。"""

    as_of: str
    benchmark: str
    cash: float
    total_market_value: float  # 持仓市值合计（不含现金）
    total_value: float  # 市值 + 现金
    total_cost: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    day_change_pct: float  # 持仓部分的单日变动（市值加权）
    top_weight: float  # 最大单一持仓权重（集中度）
    herfindahl: float  # HHI = Σ w_i²（持仓权重，不含现金；越高越集中）
    current_holdings_ann_vol: float
    current_holdings_sharpe: float
    current_holdings_sortino: float
    current_holdings_max_drawdown: float
    portfolio_beta: float  # Σ w_i·β_i（现金权重摊薄）
    positions: list[PositionSnapshot] = field(default_factory=list)
    missing_prices: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = [p.as_dict() if isinstance(p, PositionSnapshot) else p for p in self.positions]
        return d


def _aggregate_lots(portfolio: Portfolio) -> list[tuple[str, float, float, str]]:
    """多 lot 聚合 → (symbol, total_shares, 加权平均成本, 最早建仓日)。

    加权平均成本 = Σ(shares×cost)/Σshares；多空 lot 对冲后 shares 为 0 的标的
    直接剔除（净持仓为零，无暴露）。
    """
    agg: dict[str, dict] = {}
    for p in portfolio.positions:
        slot = agg.setdefault(
            p.symbol, {"shares": 0.0, "cost": 0.0, "open_date": p.open_date}
        )
        slot["shares"] += p.shares
        slot["cost"] += p.shares * p.cost_basis
        slot["open_date"] = min(slot["open_date"], p.open_date)
    out = []
    for sym, s in agg.items():
        if s["shares"] == 0:
            continue
        out.append((sym, s["shares"], s["cost"] / s["shares"], str(s["open_date"])))
    return out


class PortfolioAnalyzer:
    """真实组合 × 注入的价格历史 → 快照。

    Args:
        prices: {symbol: OHLC DataFrame}，需含 `close` 列（`high` 缺失时用 close 兜底，
            只影响 pullback 的滚动高点口径）。索引为交易日（升序）。
        benchmark_prices: 基准（如 SPY）的 OHLC DataFrame。
        benchmark: 基准名（进快照，报告用）。
    """

    def __init__(
        self,
        prices: dict[str, pd.DataFrame],
        benchmark_prices: pd.DataFrame,
        benchmark: str = "SPY",
    ) -> None:
        self.prices = prices
        self.benchmark_prices = benchmark_prices
        self.benchmark = benchmark

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
        """入口卫生：tz-aware 索引剥 tz（yfinance 默认带 America/New_York，与合成
        tz-naive 数据混用会让 pandas align 直接 TypeError）；重复日期保留最后一条
        （yfinance 已知毛刺：盘中条+结算条同日重复）。"""
        out = df
        if getattr(out.index, "tz", None) is not None:
            out = out.tz_localize(None)
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep="last")]
        return out

    def analyze(self, portfolio: Portfolio) -> PortfolioSnapshot:
        lots = _aggregate_lots(portfolio)
        bench_df = self.benchmark_prices
        if len(bench_df) and "close" in bench_df.columns:
            bench_close = self._clean_frame(bench_df)["close"].astype(float)
        else:
            # 缺 close 列与空 df 同等对待：无基准（beta 全 NaN），不 KeyError 崩掉
            bench_close = pd.Series(dtype=float)
        bench_returns = returns_from_prices(bench_close) if len(bench_close) else pd.Series(dtype=float)

        snapshots: list[PositionSnapshot] = []
        missing: list[str] = []
        value_series: dict[str, pd.Series] = {}  # symbol -> shares×close 历史
        prev_value_total = 0.0  # Σ shares×前收（组合日变动的正确分母）
        delta_value_total = 0.0  # Σ shares×(last−prev)
        day_change_ok = True

        for sym, shares, avg_cost, open_date in lots:
            df = self.prices.get(sym)
            if df is None or len(df) == 0 or "close" not in df.columns:
                missing.append(sym)
                continue
            df = self._clean_frame(df)
            close = df["close"].astype(float)
            valid_close = close.dropna()
            if valid_close.empty:
                # close 列全 NaN（退市/坏抓取的常见形状）：等同缺价，进 missing，
                # 绝不让 NaN 毒化总计（诚实契约，有测试）。
                missing.append(sym)
                continue
            high = df["high"].astype(float) if "high" in df.columns else close
            last = float(valid_close.iloc[-1])
            prev = float(valid_close.iloc[-2]) if len(valid_close) >= 2 else float("nan")
            if prev == prev:
                prev_value_total += shares * prev
                delta_value_total += shares * (last - prev)
            else:
                day_change_ok = False
            mv = shares * last
            cost_value = shares * avg_cost
            pnl = mv - cost_value
            pb = pullback(close, high).iloc[-1]
            snapshots.append(
                PositionSnapshot(
                    symbol=sym,
                    shares=shares,
                    avg_cost=avg_cost,
                    open_date=open_date,
                    last_price=last,
                    market_value=mv,
                    cost_value=cost_value,
                    unrealized_pnl=pnl,
                    # 空头 lot cost_value<0：分母取绝对值保证「涨=亏」符号正确
                    unrealized_pnl_pct=pnl / abs(cost_value) if cost_value else float("nan"),
                    weight=float("nan"),  # 二次遍历回填（需总值）
                    day_change_pct=(last / prev - 1) if prev == prev else float("nan"),
                    rsi_14=float(rsi(close, 14).iloc[-1]),
                    macd_histogram=float(macd(close)["macd_histogram"].iloc[-1]),
                    in_uptrend=bool(pb["in_uptrend"]),
                    is_pullback=bool(pb["pullback"]),
                    realized_vol_20=float(realized_volatility(close, 20).iloc[-1]),
                    beta_vs_benchmark=beta(returns_from_prices(close), bench_returns),
                )
            )
            value_series[sym] = close * shares

        total_mv = float(sum(s.market_value for s in snapshots))
        total_value = total_mv + portfolio.cash
        total_cost = float(sum(s.cost_value for s in snapshots))
        total_pnl = total_mv - total_cost

        # 权重回填（分母 = 总值含现金，带符号=暴露方向）；
        # 集中度改用 **gross** 口径（|mv|/Σ|mv|）——净分母在含空头时权重会 >1、
        # HHI 无界、近似美元中性组合直接爆表（审计确认后修正）。
        for s in snapshots:
            s.weight = s.market_value / total_value if total_value else float("nan")
        gross_mv = float(sum(abs(s.market_value) for s in snapshots))
        inner_w = [abs(s.market_value) / gross_mv for s in snapshots] if gross_mv else []
        # 组合日变动 = Σ shares·(last−prev) / |Σ shares·prev|（**前收市值**加权。
        # 旧实现用当日市值加权：+5%/−5% 等权组合会报 +0.25% 的系统性正偏、
        # 净空头账本符号还会翻转——审计确认后重写）。分母含空头取 abs
        # 保证「亏=负」；净暴露≈0 时数学未定义 → NaN。
        day_change = (
            delta_value_total / abs(prev_value_total)
            if day_change_ok and snapshots and abs(prev_value_total) > 1e-12
            else float("nan")
        )

        # 当前持仓固定的合成历史（holdings-based；交易日索引取各标的交集）。
        # 合成净值 <= 0（深度净空头 + 低现金）时 pct_change 收益无意义
        # （符号翻转、Sharpe 垃圾值）→ 统计全 NaN（诚实拒算）。
        vol = sharpe = sortino = mdd = float("nan")
        if value_series:
            values = pd.concat(value_series.values(), axis=1, join="inner").sum(axis=1)
            values = values + portfolio.cash
            if len(values) >= 3 and bool((values > 0).all()):
                rets = returns_from_prices(values)
                vol = float(rets.std() * np.sqrt(252))
                sharpe = float(sharpe_ratio(rets))
                sortino = float(sortino_ratio(rets))
                mdd = float(drawdown_curve(values).min())

        # 组合 beta = Σ w_i·β_i（w 含现金摊薄；缺 beta 的标的按 NaN 传播诚实处理）
        betas = [s.beta_vs_benchmark * s.weight for s in snapshots]
        portfolio_beta = float(sum(betas)) if betas and not any(np.isnan(b) for b in betas) else float("nan")

        as_of = ""
        if value_series:
            as_of = str(max(vs.index[-1] for vs in value_series.values()))

        return PortfolioSnapshot(
            as_of=as_of,
            benchmark=self.benchmark,
            cash=portfolio.cash,
            total_market_value=total_mv,
            total_value=total_value,
            total_cost=total_cost,
            total_unrealized_pnl=total_pnl,
            total_unrealized_pnl_pct=total_pnl / abs(total_cost) if total_cost else float("nan"),
            day_change_pct=day_change,
            top_weight=max(inner_w) if inner_w else float("nan"),
            herfindahl=float(sum(w * w for w in inner_w)) if inner_w else float("nan"),
            current_holdings_ann_vol=vol,
            current_holdings_sharpe=sharpe,
            current_holdings_sortino=sortino,
            current_holdings_max_drawdown=mdd,
            portfolio_beta=portfolio_beta,
            positions=snapshots,
            missing_prices=missing,
        )


def format_snapshot_text(snap: PortfolioSnapshot) -> str:
    """快照 → 终端可读报告（纯文本，CLI 用）。"""

    def pct(x: float) -> str:
        return f"{x * 100:+.2f}%" if x == x else "n/a"

    def num(x: float) -> str:
        return f"{x:,.2f}" if x == x else "n/a"

    lines = [
        f"=== Portfolio Snapshot @ {snap.as_of or 'n/a'} (benchmark {snap.benchmark}) ===",
        f"Total value : ${num(snap.total_value)}  (cash ${num(snap.cash)})",
        f"Unrealized  : ${num(snap.total_unrealized_pnl)} ({pct(snap.total_unrealized_pnl_pct)})"
        f"   Day: {pct(snap.day_change_pct)}",
        f"Risk (current holdings held fixed): vol {pct(snap.current_holdings_ann_vol)}"
        f" | Sharpe {num(snap.current_holdings_sharpe)} | Sortino {num(snap.current_holdings_sortino)}"
        f" | MDD {pct(snap.current_holdings_max_drawdown)} | beta {num(snap.portfolio_beta)}",
        f"Concentration: top {pct(snap.top_weight)} | HHI {num(snap.herfindahl)}",
    ]
    if snap.missing_prices:
        lines.append(f"!! MISSING PRICES (excluded from stats): {', '.join(snap.missing_prices)}")
    lines.append("")
    header = (
        f"{'SYM':<6}{'SHARES':>9}{'AVGCOST':>10}{'LAST':>10}{'MKT VAL':>12}"
        f"{'PNL':>12}{'PNL%':>9}{'W%':>7}{'RSI':>6}{'TREND':>7}{'PB':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    def cell(x: float, width: int, spec: str = ",.2f") -> str:
        # NaN 统一渲染 n/a（与 pct()/num() 同一约定；旧实现直接打 'nan'）
        return f"{x:>{width}{spec}}" if x == x else f"{'n/a':>{width}}"

    for s in sorted(snap.positions, key=lambda x: -abs(x.market_value)):
        lines.append(
            f"{s.symbol:<6}{cell(s.shares, 9, '.2f')}{cell(s.avg_cost, 10, '.2f')}"
            f"{cell(s.last_price, 10, '.2f')}{cell(s.market_value, 12)}"
            f"{cell(s.unrealized_pnl, 12)}{pct(s.unrealized_pnl_pct):>9}"
            f"{cell(s.weight * 100 if s.weight == s.weight else float('nan'), 7, '.1f')}"
            f"{cell(s.rsi_14, 6, '.1f')}{'UP' if s.in_uptrend else '--':>7}{'Y' if s.is_pullback else '':>4}"
        )
    return "\n".join(lines)
