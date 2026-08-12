"""组合分析页：我的钱怎么样了？（组合快照 + 净值曲线，纯平移自 streamlit_app.py）"""
from __future__ import annotations

from quantai.ui.common import fmt_money as _fmt_money
from quantai.ui.common import fmt_pct as _fmt_pct


def render() -> None:  # pragma: no cover - 需 streamlit 运行时
    from datetime import datetime, timedelta
    from pathlib import Path

    import streamlit as st

    from quantai.config import load_config
    from quantai.portfolio import PortfolioAnalyzer, load_portfolio
    from quantai.ui.charts import pnl_bar_figure
    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
    st.title("QuantAI · 真实组合分析" if lang == "zh" else "QuantAI - Portfolio")
    cfg = load_config().portfolio
    st.sidebar.header(tr(lang, "pf.sidebar"))
    pf_path = st.sidebar.text_input(tr(lang, "pf.file"), value=cfg.file)
    benchmark = st.sidebar.text_input(tr(lang, "pf.bench"), value=cfg.benchmark)
    years = st.sidebar.slider(tr(lang, "pf.years"), 1, 5, cfg.history_years)

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
    c1.metric(tr(lang, "pf.total"), _fmt_money(snap.total_value), _fmt_pct(snap.day_change_pct))
    c2.metric(tr(lang, "pf.pnl"), _fmt_money(snap.total_unrealized_pnl), _fmt_pct(snap.total_unrealized_pnl_pct))
    c3.metric(tr(lang, "pf.beta"), f"{snap.portfolio_beta:.2f}" if snap.portfolio_beta == snap.portfolio_beta else "n/a")
    c4.metric(tr(lang, "pf.vol"), _fmt_pct(snap.current_holdings_ann_vol))
    c5.metric(tr(lang, "pf.mdd"), _fmt_pct(snap.current_holdings_max_drawdown))
    st.caption(tr(lang, "pf.note"))
    if snap.missing_prices:
        st.warning(tr(lang, "pf.missing", syms=", ".join(snap.missing_prices)))

    # 持仓表
    st.subheader(tr(lang, "pf.holdings"))
    st.dataframe(
        [
            {
                tr(lang, "pf.h_symbol"): s.symbol,
                tr(lang, "pf.h_shares"): s.shares,
                tr(lang, "pf.h_avg"): round(s.avg_cost, 2),
                tr(lang, "pf.h_last"): round(s.last_price, 2),
                tr(lang, "pf.h_mv"): round(s.market_value, 2),
                tr(lang, "pf.h_pnl"): round(s.unrealized_pnl, 2),
                tr(lang, "pf.h_pnlpct"): _fmt_pct(s.unrealized_pnl_pct),
                tr(lang, "pf.h_weight"): round(s.weight * 100, 1) if s.weight == s.weight else None,
                "RSI": round(s.rsi_14, 1) if s.rsi_14 == s.rsi_14 else None,
                tr(lang, "pf.h_trend"): "^" if s.in_uptrend else "-",
                tr(lang, "pf.h_pb"): "*" if s.is_pullback else "",
            }
            for s in sorted(snap.positions, key=lambda x: -abs(x.market_value))
        ],
        use_container_width=True,
    )
    st.plotly_chart(
        pnl_bar_figure([s.as_dict() for s in snap.positions], title=tr(lang, "pf.pnl_chart")),
        use_container_width=True,
    )

    # 编辑持仓：交易后同步真实仓位（WS 无公开 API，手动两步全站对齐）
    with st.expander(tr(lang, "pf.edit")):
        import yaml as _yaml

        st.caption(tr(lang, "pf.edit_note"))
        e_cash = st.number_input(tr(lang, "pf.e_cash"), value=float(portfolio.cash),
                                 step=10.0, min_value=0.0, key="pf_e_cash")
        edited = []
        for i, pp in enumerate(portfolio.positions):
            c1, c2, c3 = st.columns([1.2, 1, 1])
            c1.markdown(f"**{pp.symbol}**")
            sh = c2.number_input(f"{tr(lang, 'pf.e_shares')} · {pp.symbol}", value=float(pp.shares),
                                 step=1.0, key=f"pf_e_sh_{i}")
            cb = c3.number_input(f"{tr(lang, 'pf.e_cost')} · {pp.symbol}", value=float(pp.cost_basis),
                                 step=0.01, min_value=0.0, key=f"pf_e_cb_{i}")
            edited.append((pp, sh, cb))
        if st.button(tr(lang, "pf.e_save")):
            data = {
                "cash": float(e_cash),
                "positions": [
                    {"symbol": pp.symbol, "shares": float(sh), "cost_basis": float(cb),
                     "open_date": str(pp.open_date)}
                    for pp, sh, cb in edited if sh != 0
                ],
            }
            header = (
                "# 真实持仓（本文件被 .gitignore 排除，永不入库）。\n"
                "# 由仪表盘「编辑持仓」写入；全 USD 口径。\n"
            )
            Path(pf_path).write_text(
                header + _yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            st.success(tr(lang, "pf.e_saved"))
            st.cache_data.clear()
            st.rerun()

    # 宏观长线区（模拟盘同款 line_chart 交互）。两种视图：
    # 净值（默认）=「我的钱随时间的变化」：持仓股数×历史收盘+现金，对照同额投基准；
    # 归一化对比 = 相对表现（涨跌幅口径，与起点价格无关）。
    # 日线蜡烛/RSI/MACD 细看在行情工作台——组合页定位是长线宏观，不重复摆日线。
    import pandas as pd

    held = [s.symbol for s in snap.positions]
    view = st.radio(
        tr(lang, "pf.view"), ["net", "norm"],
        format_func=lambda v: tr(lang, f"pf.view_{v}"), horizontal=True, key="pf_view",
    )
    if view == "net":
        st.subheader(tr(lang, "pf.networth"))
        held_shares: dict[str, float] = {}
        for pp in portfolio.positions:
            held_shares[pp.symbol] = held_shares.get(pp.symbol, 0.0) + pp.shares
        cols = {}
        for s2, sh in held_shares.items():
            df = prices.get(s2)
            if df is not None and not df.empty and "close" in df.columns:
                cols[s2] = df["close"].dropna() * sh
        if cols:
            aligned = pd.concat(cols.values(), axis=1).dropna()
            nw = aligned.sum(axis=1) + float(portfolio.cash)
            chart = {tr(lang, "pf.networth"): nw}
            spy = prices.get(benchmark)
            if spy is not None and not spy.empty and len(nw):
                sc = spy["close"].dropna().reindex(nw.index).dropna()
                if len(sc):
                    chart[tr(lang, "pf.spy_same", bench=benchmark)] = (
                        sc / float(sc.iloc[0]) * float(nw.iloc[0])
                    )
            st.line_chart(pd.DataFrame(chart), use_container_width=True)
            if len(nw):
                st.caption(tr(lang, "pf.net_note", start=str(nw.index[0].date())))
    else:
        st.subheader(tr(lang, "pf.longview"))
        norm = {}
        for s in held + ([benchmark] if benchmark not in held else []):
            df = prices.get(s)
            if df is None or df.empty or "close" not in df.columns:
                continue
            c = df["close"].dropna()
            if len(c) > 1:
                norm[s] = c / float(c.iloc[0]) * 100.0
        if norm:
            st.line_chart(pd.DataFrame(norm), use_container_width=True)
            st.caption(tr(lang, "pf.longview_note", years=years))

    sym = st.selectbox(tr(lang, "pf.h_symbol"), held + ([benchmark] if benchmark not in held else []))

    @st.cache_data(ttl=900, show_spinner=False)
    def _fetch_news(symbol: str) -> list:
        from quantai.data.news import NewsFetcher

        return [n.as_dict() for n in NewsFetcher().fetch_symbol_news(symbol, limit=10)]

    st.subheader(tr(lang, "pf.news_of", sym=sym))
    news_items = _fetch_news(sym)
    if news_items:
        for n in news_items:
            ts = n["published"].strftime("%m-%d %H:%M") if n["published"] else "?"
            st.markdown(f"- [{n['title']}]({n['link']})  `{ts} UTC`")
    else:
        st.caption(tr(lang, "pf.no_news"))
