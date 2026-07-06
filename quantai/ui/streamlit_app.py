"""Streamlit 仪表盘 —— 两页：真实组合分析（Phase C/F）+ 实盘会话（读 v2 API）。

诚实：**只展示真实数据**——组合页读 `portfolio.local.yaml` + yfinance 实时行情；
实盘页读 API 账户快照/成交。没有数据就给空态与指引，不伪造任何面板（C-2 口径）。

跑：
    venv311\\Scripts\\python.exe -m streamlit run quantai/ui/streamlit_app.py
    （组合分析页独立可用；实盘页需先 `python scripts/serve.py` 起 API。）

图表构建在 `quantai.ui.charts`（纯函数、可离线测试）；streamlit 仅做渲染，
故 `import` 本文件无副作用（streamlit 在 `main()` 内懒导入）。
"""
from __future__ import annotations

from typing import Any, Dict


def _fmt_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def _fmt_pct(x: Any) -> str:
    try:
        v = float(x)
        return f"{v * 100:+.2f}%" if v == v else "n/a"
    except Exception:
        return "n/a"


def _render_portfolio_page(st) -> None:  # pragma: no cover - 需 streamlit 运行时
    from datetime import datetime, timedelta
    from pathlib import Path

    from quantai.config import load_config
    from quantai.portfolio import PortfolioAnalyzer, load_portfolio
    from quantai.ui.charts import candlestick_figure, pnl_bar_figure, rsi_macd_figure

    cfg = load_config().portfolio
    st.sidebar.header("组合分析")
    pf_path = st.sidebar.text_input("持仓文件", value=cfg.file)
    benchmark = st.sidebar.text_input("基准", value=cfg.benchmark)
    years = st.sidebar.slider("历史回看（年）", 1, 5, cfg.history_years)

    if not Path(pf_path).exists():
        st.info(
            f"没找到持仓文件 `{pf_path}`。\n\n"
            "把 `portfolio.example.yaml` 复制成 `portfolio.local.yaml` 填入真实持仓"
            "（该文件被 .gitignore 排除，不会进版本库）。"
        )
        return
    portfolio = load_portfolio(pf_path)
    if not portfolio.positions:
        st.info("持仓文件只有现金，没有可分析的标的。")
        return

    @st.cache_data(ttl=900, show_spinner="拉取行情中…")
    def _fetch(symbols: tuple, start: str) -> dict:
        from quantai.data.prices import PriceFetcher

        return PriceFetcher().fetch_prices(list(symbols), start)

    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    symbols = tuple(dict.fromkeys(portfolio.symbols + [benchmark]))
    prices = _fetch(symbols, start)
    bench_df = prices.get(benchmark)
    if bench_df is None or bench_df.empty:
        st.error(f"基准 {benchmark} 抓不到行情（网络/代码问题），无法分析。")
        return

    snap = PortfolioAnalyzer(prices, bench_df, benchmark=benchmark).analyze(portfolio)

    # KPI 行
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总资产", _fmt_money(snap.total_value), _fmt_pct(snap.day_change_pct))
    c2.metric("未实现盈亏", _fmt_money(snap.total_unrealized_pnl), _fmt_pct(snap.total_unrealized_pnl_pct))
    c3.metric("组合 Beta", f"{snap.portfolio_beta:.2f}" if snap.portfolio_beta == snap.portfolio_beta else "n/a")
    c4.metric("年化波动*", _fmt_pct(snap.current_holdings_ann_vol))
    c5.metric("最大回撤*", _fmt_pct(snap.current_holdings_max_drawdown))
    st.caption("*基于「当前持仓固定不变」的合成历史（holdings-based），非真实交易流水。")
    if snap.missing_prices:
        st.warning(f"以下标的抓不到行情，已从统计剔除：{', '.join(snap.missing_prices)}")

    # 持仓表
    st.subheader("持仓")
    st.dataframe(
        [
            {
                "标的": s.symbol,
                "股数": s.shares,
                "均价": round(s.avg_cost, 2),
                "现价": round(s.last_price, 2),
                "市值": round(s.market_value, 2),
                "盈亏": round(s.unrealized_pnl, 2),
                "盈亏%": _fmt_pct(s.unrealized_pnl_pct),
                "权重%": round(s.weight * 100, 1) if s.weight == s.weight else None,
                "RSI": round(s.rsi_14, 1) if s.rsi_14 == s.rsi_14 else None,
                "趋势": "↑" if s.in_uptrend else "—",
                "回踩": "●" if s.is_pullback else "",
            }
            for s in sorted(snap.positions, key=lambda x: -abs(x.market_value))
        ],
        use_container_width=True,
    )
    st.plotly_chart(pnl_bar_figure([s.as_dict() for s in snap.positions]), use_container_width=True)

    # 个股 K 线 + 指标 + 新闻
    st.subheader("个股图表")
    held = [s.symbol for s in snap.positions]
    sym = st.selectbox("标的", held + ([benchmark] if benchmark not in held else []))
    df = prices.get(sym)
    if df is not None and not df.empty:
        st.plotly_chart(candlestick_figure(df, sym), use_container_width=True)
        st.plotly_chart(rsi_macd_figure(df), use_container_width=True)

    @st.cache_data(ttl=900, show_spinner=False)
    def _fetch_news(symbol: str) -> list:
        from quantai.data.news import NewsFetcher

        return [n.as_dict() for n in NewsFetcher().fetch_symbol_news(symbol, limit=10)]

    st.subheader(f"{sym} · 最新新闻")
    news_items = _fetch_news(sym)
    if news_items:
        for n in news_items:
            ts = n["published"].strftime("%m-%d %H:%M") if n["published"] else "时间未知"
            st.markdown(f"- [{n['title']}]({n['link']})  `{ts} UTC`")
    else:
        st.caption("暂无该标的新闻（RSS 源无返回）。")


def main() -> None:  # pragma: no cover - 需 streamlit 运行时
    import streamlit as st

    from quantai.ui.client import QuantAIClient

    st.set_page_config(page_title="QuantAI Dashboard", layout="wide")

    page = st.sidebar.radio("页面", ["组合分析", "实盘会话"], index=0)
    if page == "组合分析":
        st.title("QuantAI · 真实组合分析")
        _render_portfolio_page(st)
        return

    st.title("QuantAI · 实盘仪表盘")
    base_url = st.sidebar.text_input("API 地址", value="http://localhost:8000")
    client = QuantAIClient(base_url)

    # 健康检查
    try:
        client.health()
        st.sidebar.success("API 在线")
    except Exception as exc:
        st.sidebar.error(f"API 不可达：{exc}")
        st.stop()

    # 实盘控制
    st.sidebar.header("实盘会话")
    tickers = st.sidebar.text_input("标的（逗号分隔）", value="NVDA,TSLA")
    source = st.sidebar.selectbox("数据源", ["simulated", "yfinance", "auto"], index=0)
    cash = st.sidebar.number_input("初始现金", value=100000.0, step=10000.0)
    collect = st.sidebar.checkbox("开启数据飞轮采集", value=False)
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("启动"):
        client.start_live(
            [t.strip().upper() for t in tickers.split(",") if t.strip()],
            source=source,
            cash=float(cash),
            interval_sec=0.5,
            collect=collect,
        )
    if col_b.button("停止"):
        client.stop_live()

    # 真实状态
    status: Dict[str, Any] = client.live_status()
    if not status.get("active"):
        st.info("当前无实盘会话。用左侧启动。")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("现金", _fmt_money(status.get("cash")))
    c2.metric("权益", _fmt_money(status.get("equity")))
    c3.metric("订单数", status.get("orders", 0))
    c4.metric("数据源", status.get("feed_source", "-"))

    st.subheader("持仓")
    positions = status.get("positions") or {}
    if positions:
        st.dataframe(
            [{"ticker": k, "shares": v.get("shares"), "avg_price": v.get("avg_price")} for k, v in positions.items()]
        )
    else:
        st.caption("暂无持仓")

    st.subheader("最近成交")
    trades = client.live_trades(limit=50).get("trades", [])
    if trades:
        st.dataframe(trades)
    else:
        st.caption("暂无成交")

    st.subheader("数据飞轮")
    st.json(client.active_adapters())


if __name__ == "__main__":
    main()
