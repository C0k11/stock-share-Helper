"""LLM 分析师：把系统的全部真实状态组装成 brief，交给 LLM 出财经分析。

「LLM 能看到什么」= brief 里有什么（全部来自系统真实数据，逐节标注来源）：
1. 真实持仓快照（PortfolioAnalyzer：盈亏/权重/风险统计/组合 beta）
2. 每标的技术指标（analysis/ 引擎：RSI/MACD/均线/波动/回踩）
3. 最新新闻头条（quantai.data.news，含自选股）
4. 数据仓库 SQL 摘要（DuckDB marts：近 5 日涨跌幅榜、距 52 周高点、回测结果）

设计：`build_brief` 纯组装（全部输入注入，离线可测）；`generate_commentary`
吃鸭子类型 `llm.generate(user, system)`（`LocalLLM` 兼容，测试用假 LLM）。
诚实约束进 system prompt：只引用 brief 里的数字、不确定要说、不给投资建议式断言。
"""

from __future__ import annotations

from typing import Optional

_SYSTEM_PROMPT = (
    "你是一名严谨的量化分析师助手。基于下面的系统数据简报做客观分析：\n"
    "1) 只引用简报中出现的数字，不编造任何数据；2) 数据不足处明说不足；\n"
    "3) 输出【组合体检】【个股要点】【风险提示】【今日关注】四段；\n"
    "4) 语气克制，这是分析参考不是投资建议。"
)


def _fmt(x, nd: int = 2) -> str:
    try:
        v = float(x)
        return f"{v:.{nd}f}" if v == v else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _portfolio_section(snap) -> str:
    lines = [
        "## 一、真实持仓快照（PortfolioAnalyzer）",
        f"- 总资产 ${_fmt(snap.total_value)}（现金 ${_fmt(snap.cash)}）"
        f"，未实现盈亏 ${_fmt(snap.total_unrealized_pnl)}"
        f"（{_fmt(snap.total_unrealized_pnl_pct * 100, 2)}%），当日 {_fmt(snap.day_change_pct * 100, 2)}%",
        f"- 组合 beta {_fmt(snap.portfolio_beta)}，年化波动 {_fmt(snap.current_holdings_ann_vol * 100, 1)}%"
        f"，最大回撤 {_fmt(snap.current_holdings_max_drawdown * 100, 1)}%（holdings-based 假设）",
        f"- 集中度：最大持仓权重 {_fmt(snap.top_weight * 100, 1)}%，HHI {_fmt(snap.herfindahl)}",
    ]
    for s in snap.positions:
        lines.append(
            f"- {s.symbol}: {_fmt(s.shares, 0)} 股 @ 均价 {_fmt(s.avg_cost)}，现价 {_fmt(s.last_price)}"
            f"，盈亏 {_fmt(s.unrealized_pnl)}（{_fmt(s.unrealized_pnl_pct * 100, 1)}%）"
            f"，RSI {_fmt(s.rsi_14, 1)}，趋势{'向上' if s.in_uptrend else '不明/向下'}"
            f"{'，回踩形态' if s.is_pullback else ''}"
        )
    if snap.missing_prices:
        lines.append(f"- ⚠ 缺行情已剔除：{', '.join(snap.missing_prices)}")
    return "\n".join(lines)


def _watchlist_section(rows: list[dict]) -> str:
    if not rows:
        return "## 二、自选股\n（无自选股数据）"
    lines = ["## 二、自选股行情/指标（最新收盘）"]
    for r in rows:
        lines.append(
            f"- {r['symbol']}: {_fmt(r.get('last'))}（日 {_fmt(r.get('day_pct', float('nan')) * 100, 2)}%）"
            f"，RSI {_fmt(r.get('rsi'), 1)}，20D 年化波动 {_fmt(r.get('vol', float('nan')) * 100, 1)}%"
        )
    return "\n".join(lines)


def _news_section(news_by_symbol: dict[str, list]) -> str:
    lines = ["## 三、最新新闻头条（RSS）"]
    empty = True
    for sym, items in news_by_symbol.items():
        for it in items[:3]:
            ts = it.published.strftime("%m-%d") if getattr(it, "published", None) else "?"
            lines.append(f"- [{sym}] ({ts}) {it.title}")
            empty = False
    if empty:
        lines.append("（无新闻数据）")
    return "\n".join(lines)


def _warehouse_section(con) -> str:
    """DuckDB marts 摘要（连接注入；仓库还没建则如实说明）。"""
    lines = ["## 四、数据仓库摘要（DuckDB marts）"]
    if con is None:
        lines.append("（仓库未初始化，跑 scripts/warehouse.py --full 后可用）")
        return "\n".join(lines)
    try:
        movers = con.execute(
            """SELECT symbol, round(100 * (max_by(close, date) / min_by(close, date) - 1), 2) AS pct_5d
               FROM (SELECT symbol, date, close,
                            row_number() OVER (PARTITION BY symbol ORDER BY date DESC) rn
                     FROM marts.fact_prices)
               WHERE rn <= 5 GROUP BY symbol ORDER BY pct_5d DESC"""
        ).fetchall()
        lines.append(
            "- 近5日涨跌榜: "
            + ", ".join(f"{s} {p:+.2f}%" for s, p in movers[:8] if p is not None)
        )
        high52 = con.execute(
            """SELECT symbol, round(100 * pct_from_52w_high, 1)
               FROM marts.fact_prices
               WHERE (symbol, date) IN (SELECT symbol, max(date) FROM marts.fact_prices GROUP BY symbol)
                 AND pct_from_52w_high IS NOT NULL
               ORDER BY 2"""
        ).fetchall()
        if high52:
            lines.append(
                "- 距52周高点: " + ", ".join(f"{s} {p}%" for s, p in high52[:8])
            )
        bt = con.execute(
            "SELECT run_id, round(sharpe,2), round(100*max_drawdown,1) FROM marts.fact_backtest_results"
        ).fetchall()
        for run_id, sharpe, mdd in bt[:3]:
            lines.append(f"- 回测 {run_id}: Sharpe {sharpe}, MDD {mdd}%")
    except Exception as exc:  # noqa: BLE001 - 表缺失等如实降级
        lines.append(f"（仓库查询失败：{exc}）")
    return "\n".join(lines)


def build_brief(
    snap,
    watchlist_rows: Optional[list[dict]] = None,
    news_by_symbol: Optional[dict[str, list]] = None,
    warehouse_con=None,
    as_of: str = "",
) -> str:
    """全部真实状态 → 一份 LLM 可读的 markdown 简报（纯组装，离线可测）。"""
    parts = [f"# QuantAI 每日数据简报{f'（{as_of}）' if as_of else ''}"]
    parts.append(_portfolio_section(snap))
    parts.append(_watchlist_section(watchlist_rows or []))
    parts.append(_news_section(news_by_symbol or {}))
    parts.append(_warehouse_section(warehouse_con))
    return "\n\n".join(parts)


def generate_commentary(brief: str, llm) -> str:
    """brief → LLM 财经分析。llm 鸭子接口 `generate(user, system=...)`（LocalLLM 兼容）。"""
    return llm.generate(brief, system=_SYSTEM_PROMPT)
