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


# --------------------------------------------------------------------------- #
# 盘中模式（分钟级 bar → 当日会话统计 + 卖压代理指标）
# --------------------------------------------------------------------------- #
def intraday_stats(df_1m, prev_close: float, avg_daily_volume: float) -> dict:
    """当日分钟 bar → 会话统计（纯函数，可测）。

    卖压/买压是**bar 级代理指标**（我们只有 OHLCV，没有逐笔/盘口数据，不冒充 order flow）：
    - `down_volume_share`：收跌 bar 的成交量占比（>0.5 = 抛压主导）。
    - `vs_vwap_pct`：现价相对当日 VWAP（负值 = 均价下方交易，偏弱）。
    - `close_position`：现价在当日高低区间的位置（0=贴地，1=顶部）。
    - `vol_vs_avg`：当日累计量 / 20 日均量（>1 提前放量）。
    """
    import numpy as np

    close = df_1m["close"].astype(float).dropna()
    if close.empty or prev_close != prev_close or prev_close <= 0:
        return {}
    last = float(close.iloc[-1])
    high = float(df_1m["high"].astype(float).max()) if "high" in df_1m.columns else last
    low = float(df_1m["low"].astype(float).min()) if "low" in df_1m.columns else last
    vol = df_1m["volume"].astype(float) if "volume" in df_1m.columns else None
    out = {
        "last": last,
        "chg_pct": last / prev_close - 1,
        "day_high": high,
        "day_low": low,
        "close_position": (last - low) / (high - low) if high > low else float("nan"),
    }
    if vol is not None and float(vol.sum()) > 0:
        typical = (df_1m["high"].astype(float) + df_1m["low"].astype(float) + close) / 3
        vwap = float((typical * vol).sum() / vol.sum())
        bar_chg = close.diff()
        down_vol = float(vol[bar_chg < 0].sum())
        out.update(
            {
                "vwap": vwap,
                "vs_vwap_pct": last / vwap - 1,
                "cum_volume": float(vol.sum()),
                "vol_vs_avg": float(vol.sum()) / avg_daily_volume if avg_daily_volume > 0 else float("nan"),
                "down_volume_share": down_vol / float(vol.sum()),
            }
        )
    return out


_INTRADAY_SYSTEM = (
    "你是一名盘中交易台分析师。基于下面的**当日分钟级会话数据**做盘中快评：\n"
    "1) 只引用简报中的数字，不编造；2) 卖压/买压结论必须落到 down_volume_share、"
    "VWAP 相对位置、区间位置这些给定指标上；3) 输出【大盘/持仓速览】【压力与量能】"
    "【新闻情绪】【风险】四段，短句、克制；4) 这是分析参考不是投资建议。"
)


def build_intraday_brief(
    rows: list[dict],
    scored_news: Optional[list[dict]] = None,
    symbol_sentiment: Optional[dict] = None,
    as_of: str = "",
) -> str:
    """盘中会话统计 + 量化新闻 → 盘中简报（markdown，纯组装可测）。

    rows: [{"symbol", **intraday_stats 输出}]；scored_news: news_scorer.score_news 输出。
    """
    lines = [f"# QuantAI 盘中快报{f'（{as_of}）' if as_of else ''}", "", "## 一、当日会话（分钟级）"]
    if not rows:
        lines.append("（无盘中数据——非交易时段或数据源无返回）")
    for r in rows:
        seg = (
            f"- {r['symbol']}: {_fmt(r.get('last'))}（vs 昨收 {_fmt(r.get('chg_pct', float('nan')) * 100)}%）"
            f"，区间 [{_fmt(r.get('day_low'))}, {_fmt(r.get('day_high'))}]"
            f"，区间位置 {_fmt(r.get('close_position', float('nan')), 2)}"
        )
        if "vwap" in r:
            seg += (
                f"，VWAP {_fmt(r['vwap'])}（现价 {_fmt(r.get('vs_vwap_pct', float('nan')) * 100)}%）"
                f"，量能 {_fmt(r.get('vol_vs_avg', float('nan')), 2)}× 均量"
                f"，跌 bar 量占比 {_fmt(r.get('down_volume_share', float('nan')) * 100, 1)}%"
            )
        lines.append(seg)
    lines.append("")
    lines.append("## 二、新闻情绪（LLM 标题级量化，[-1,1]）")
    if symbol_sentiment:
        agg = ", ".join(
            f"{s} {v:+.2f}" if v is not None else f"{s} 未打分"
            for s, v in symbol_sentiment.items()
        )
        lines.append(f"- 按标的均值: {agg}")
    if scored_news:
        for row in scored_news[:15]:
            it = row["item"]
            score = f"{row['sentiment']:+.2f}" if row["sentiment"] is not None else "n/a"
            lines.append(f"- [{getattr(it, 'symbol', '') or '-'}] ({score} {row['label']}) {it.title}")
    if not scored_news and not symbol_sentiment:
        lines.append("（无量化新闻——未启用 LLM 打分或无新闻）")
    return "\n".join(lines)


def generate_intraday_commentary(brief: str, llm) -> str:
    """盘中简报 → LLM 盘中快评。"""
    return llm.generate(brief, system=_INTRADAY_SYSTEM)
