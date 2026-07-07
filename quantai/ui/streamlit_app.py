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


#: 时间档 -> (yfinance period, interval)。1D/1W 是盘中分钟级，3M 起为日线/周线。
_TIMEFRAMES = {
    "1D": ("1d", "1m"),
    "1W": ("5d", "15m"),
    "1M": ("1mo", "1h"),
    "3M": ("3mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}


def _load_llm_impl():
    """驻留 LLM 单例底层（模块级唯一函数 → st.cache_resource 全站共享一份模型）。"""
    from quantai.agents.reporting import load_report_llm

    return load_report_llm()


def _resident_llm(st):
    """工作台/AI 分析页共用的驻留模型（v2 学生 adapter，若已挂载）。"""
    return st.cache_resource(show_spinner="首次加载本地 LLM（Qwen3-8B + v2 学生，约 40 秒）…")(
        _load_llm_impl
    )()


def _render_workstation_page(st) -> None:  # pragma: no cover - 需 streamlit 运行时
    from pathlib import Path

    from quantai.config import load_config
    from quantai.data.watchlist import load_watchlist
    from quantai.portfolio import load_portfolio
    from quantai.ui.charts import workstation_figure

    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
    cfg = load_config().portfolio
    portfolio = (
        load_portfolio(cfg.file) if Path(cfg.file).exists() else None
    )
    held = portfolio.symbols if portfolio else []
    watch = load_watchlist(cfg.watchlist_file)
    universe = list(dict.fromkeys(held + watch)) or ["SPY"]

    # ---- 数据缓存（盘中分钟级 25s / 日线级 900s，两条缓存互不干扰） ----
    def _fetch_hist(symbol: str, period: str, interval: str):
        import yfinance as yf

        raw = yf.Ticker(symbol).history(period=period, interval=interval)
        raw.columns = [str(c).lower() for c in raw.columns]
        if getattr(raw.index, "tz", None) is not None:
            raw = raw.tz_convert("America/New_York").tz_localize(None)
        return raw

    @st.cache_data(ttl=25, show_spinner=False)
    def _hist_live(symbol: str, period: str, interval: str):
        return _fetch_hist(symbol, period, interval)

    @st.cache_data(ttl=900, show_spinner=False)
    def _hist_slow(symbol: str, period: str, interval: str):
        return _fetch_hist(symbol, period, interval)

    def _hist(symbol: str, period: str, interval: str):
        intra = interval.endswith("m") or interval.endswith("h")
        return (_hist_live if intra else _hist_slow)(symbol, period, interval)

    @st.cache_data(ttl=600, show_spinner=False)
    def _info(symbol: str) -> dict:
        import yfinance as yf

        try:
            return dict(yf.Ticker(symbol).info)
        except Exception:
            return {}

    # ---- 真实仓位横幅（30s 自刷：先直观吞吐用户的钱在哪、赚亏多少） ----
    if portfolio and held:
        @st.fragment(run_every=30)
        def _position_strip():
            from datetime import date as _date

            for p_sym in held:
                m1 = _hist(p_sym, "1d", "1m")
                d5 = _hist(p_sym, "10d", "1d")
                if d5 is None or d5.empty:
                    continue
                dclose = d5["close"].dropna()
                live = m1 is not None and not m1.empty and m1.index[-1].date() == _date.today()
                last = float(m1["close"].dropna().iloc[-1]) if live else float(dclose.iloc[-1])
                # 昨收基准：日线序列已含今日 bar 时取倒数第二根，否则最后一根就是昨收
                if len(dclose) >= 2 and dclose.index[-1].date() == _date.today():
                    prev = float(dclose.iloc[-2])
                elif len(dclose) >= 1 and dclose.index[-1].date() != _date.today() and live:
                    prev = float(dclose.iloc[-1])
                else:
                    prev = float(dclose.iloc[-2]) if len(dclose) >= 2 else float("nan")
                shares = sum(p.shares for p in portfolio.positions if p.symbol == p_sym)
                cost = sum(p.shares * p.cost_basis for p in portfolio.positions if p.symbol == p_sym)
                pnl = shares * last - cost
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(tr(lang, "ws.held_pos", sym=p_sym, shares=f"{shares:,.0f}"),
                          f"${last:,.2f}",
                          f"{(last / prev - 1) * 100:+.2f}% {tr(lang, 'ws.today')}" if prev == prev else None)
                c2.metric(tr(lang, "ws.avg_cost"), f"${cost / shares:,.2f}" if shares else "—")
                c3.metric(tr(lang, "ws.mkt_value"), f"${shares * last:,.2f}")
                c4.metric(tr(lang, "ws.unreal_pnl"), f"${pnl:,.2f}",
                          f"{pnl / abs(cost) * 100:+.2f}%" if cost else None)
                c5.metric(tr(lang, "ws.cash_tfsa"), f"${portfolio.cash:,.2f}")

        _position_strip()
        st.divider()

    # 顶栏：标的选择
    top_l, _sp, _tr = st.columns([2, 2, 1.4])
    sym = top_l.selectbox("标的", universe, label_visibility="collapsed")

    tf_c, rf_c = st.columns([5, 1.3])
    with tf_c:
        tf = st.segmented_control("时间档", list(_TIMEFRAMES), default="1D", label_visibility="collapsed")
    period, interval = _TIMEFRAMES[tf or "1D"]
    intraday = interval.endswith("m") or interval.endswith("h")
    # 自动刷新：st.fragment(run_every) 局部重跑——只刷"价格头+图"，不打扰整页交互
    _REFRESH = {tr(lang, "ws.refresh_off"): None, "15s": 15, "30s": 30, "60s": 60}
    with rf_c:
        auto = st.selectbox(
            tr(lang, "ws.auto_refresh"), list(_REFRESH), index=2,
            key=f"ws_auto_refresh_{lang}", label_visibility="collapsed",
        )

    # 指标开关（放 fragment 外：自动刷新不会把打开的菜单弹回去）
    with st.popover("指标"):
        show_volume = st.checkbox("Volume", True)
        show_vwap = st.checkbox("VWAP（仅盘中档有效）", intraday, disabled=not intraday)
        show_rsi = st.checkbox("RSI(14)", True)
        show_macd = st.checkbox("MACD", True)
        ma_on = st.multiselect("均线（bar 数）", [20, 50, 200], default=[20, 50])
        show_bb = st.checkbox("Bollinger(20,2)", False)
        kind = st.radio("图型", ["line", "candle"], index=0 if intraday else 1, horizontal=True)

    @st.fragment(run_every=_REFRESH[auto])
    def _live_view() -> None:
        df = _hist(sym, period, interval)
        if df is None or df.empty:
            st.error(tr(lang, "ws.no_data", sym=sym, tf=tf))
            return

        close = df["close"].astype(float).dropna()
        last, first = float(close.iloc[-1]), float(close.iloc[0])
        # 1D 档按券商惯例对比**昨收**（而非当日首根 bar，后者是"对比开盘"口径）
        if tf == "1D":
            daily = _hist(sym, "5d", "1d")
            if daily is not None and len(daily) >= 2:
                first = float(daily["close"].dropna().iloc[-2])
        chg, chg_pct = last - first, (last / first - 1) * 100
        o = float(df["open"].dropna().iloc[-1]) if "open" in df.columns else float("nan")
        h = float(df["high"].dropna().max()) if "high" in df.columns else float("nan")
        l = float(df["low"].dropna().min()) if "low" in df.columns else float("nan")
        up = chg >= 0
        color = "#26A69A" if up else "#EF5350"
        # 自定义头部（WS 式：大价格 + 内联涨跌；不用 st.metric——$ 会触发 LaTeX、涨跌不能内联）
        st.markdown(
            f"""<div style="line-height:1.15;margin:2px 0 6px 0">
  <span style="color:#8A8A93;font-size:0.95rem">{sym}</span><br>
  <span style="font-size:2.3rem;font-weight:700">${last:,.2f}</span>
  <span style="color:{color};font-size:1.05rem;font-weight:600;margin-left:8px">
    {chg:+,.2f} ({chg_pct:+.2f}%) <span style="color:#8A8A93;font-weight:400">{tf}</span>
  </span><br>
  <span style="color:#8A8A93;font-size:0.85rem">
    O&nbsp;${o:,.2f}&emsp;H&nbsp;${h:,.2f}&emsp;L&nbsp;${l:,.2f}&emsp;C&nbsp;${last:,.2f}
    &emsp;·&emsp;{interval} bars</span>
</div>""",
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            workstation_figure(
                df, sym, kind=kind, ma_windows=ma_on, show_bollinger=show_bb,
                show_vwap=show_vwap and intraday, show_volume=show_volume,
                show_rsi=show_rsi, show_macd=show_macd,
            ),
            use_container_width=True,
            key="ws_main_chart",
        )
        if _REFRESH[auto]:
            from datetime import datetime as _dt

            st.caption(tr(lang, "ws.auto_caption", auto=auto, ts=f"{_dt.now():%H:%M:%S}"))

    _live_view()

    # ================= 实时作战台（常驻分析主战场） =================
    st.subheader(tr(lang, "ws.board_title"))
    st.caption(tr(lang, "ws.board_caption"))

    def _avg_cost(s2: str):
        sh = sum(p.shares for p in (portfolio.positions if portfolio else []) if p.symbol == s2)
        ct = sum(p.shares * p.cost_basis for p in (portfolio.positions if portfolio else []) if p.symbol == s2)
        return (ct / sh) if sh else None

    # 作战台以真实持仓为核心；自选股按需补充（默认带上前几只，可增删）
    tact_syms = st.multiselect(
        tr(lang, "ws.board_symbols"),
        options=list(dict.fromkeys(held + watch)),
        default=list(dict.fromkeys(held + watch[:4])),
        key="ws_tact_syms",
    )
    tact_syms = list(dict.fromkeys(held + tact_syms))  # 持仓永远在列
    held_cost = {s2: (_avg_cost(s2) or 0.0) for s2 in held}

    from quantai.ui.cockpit import collect_advices

    @st.fragment(run_every=30)
    def _tactics_board():
        from datetime import datetime as _dt

        from quantai.agents.tactician import ALERT_MARKS, advice_row, alerts

        advs = collect_advices(tact_syms, held_cost, _hist, lang=lang)
        if not advs:
            st.caption(tr(lang, "ws.board_empty"))
            return
        for al in alerts(advs, lang=lang)[:4]:
            (st.warning if al.startswith(ALERT_MARKS) else st.info)(al)
        st.dataframe([advice_row(a, lang=lang) for a in advs],
                     use_container_width=True, hide_index=True)
        st.caption(tr(lang, "ws.board_updated", ts=f"{_dt.now():%H:%M:%S}"))

    _tactics_board()

    use_ai = st.toggle(tr(lang, "ws.ai_toggle"), value=True, key="ws_ai_on")
    if use_ai:
        from quantai.agents.reporting import load_report_llm
        from quantai.ui.cockpit import CockpitDaemon

        @st.cache_resource(show_spinner=False)
        def _cockpit_daemon() -> CockpitDaemon:
            # 单例守护线程：模型加载与推理全在后台，UI 永不被 60-90s 的生成挡住；
            # 多标签页/刷新共享同一份（绝不重复加载 10GB 模型）
            return CockpitDaemon(llm_factory=load_report_llm, interval_sec=300).start()

        daemon = _cockpit_daemon()
        daemon.configure(tact_syms, held_cost, lang=lang)

        @st.fragment(run_every=15)
        def _ai_view():
            from datetime import datetime as _dt

            stt = daemon.state
            if stt.get("out"):
                st.markdown(stt["out"])
                st.caption(tr(
                    lang, "ws.ai_status", model=stt.get("model", "?"),
                    round=stt.get("round", 0),
                    ts=f"{_dt.fromtimestamp(stt['ts']):%H:%M:%S}",
                    scored=stt.get("scored", 0), status=stt.get("status", ""),
                ))
            else:
                st.info(tr(lang, "ws.ai_waiting", status=stt.get("status", "…")))
            if stt.get("error"):
                st.warning(tr(lang, "ws.ai_error", err=stt["error"]))

        _ai_view()

    # 下方区块（Market details / Holdings / News）走同一份缓存取数：
    # 自动刷新只跳动上方图表，这里在下一次整页交互时跟上（避免 tabs 被自刷弹回首个页签）
    df = _hist(sym, period, interval)
    if df is None or df.empty:
        return
    close = df["close"].astype(float).dropna()
    last = float(close.iloc[-1])

    # Market details（来自 yfinance info；缺失诚实显示 —）
    info = _info(sym)

    def g(key, money=False, compact=False):
        v = info.get(key)
        if v is None:
            return "—"
        if compact and isinstance(v, (int, float)):
            for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
                if abs(v) >= div:
                    return f"{v / div:,.2f}{unit}"
            return f"{v:,.0f}"
        return f"${v:,.2f}" if money and isinstance(v, (int, float)) else f"{v:,}" if isinstance(v, int) else str(v)

    st.subheader("Market details")
    m1, m2, m3 = st.columns(3)
    m1.metric("Bid", g("bid", True)); m2.metric("Ask", g("ask", True)); m3.metric("Prev close", g("previousClose", True))
    m4, m5, m6 = st.columns(3)
    m4.metric("Volume", g("volume", compact=True)); m5.metric("Avg vol", g("averageVolume", compact=True)); m6.metric("P/E", g("trailingPE"))
    m7, m8, m9 = st.columns(3)
    m7.metric("52W high", g("fiftyTwoWeekHigh", True)); m8.metric("52W low", g("fiftyTwoWeekLow", True)); m9.metric("Exchange", g("exchange"))

    # 底部页签：持仓 / 新闻
    tab_hold, tab_news = st.tabs([tr(lang, "ws.tab_holdings"), tr(lang, "ws.tab_news")])
    with tab_hold:
        pos = next((p for p in (portfolio.positions if portfolio else []) if p.symbol == sym), None)
        if pos:
            agg_shares = sum(p.shares for p in portfolio.positions if p.symbol == sym)
            agg_cost = sum(p.shares * p.cost_basis for p in portfolio.positions if p.symbol == sym)
            avg = agg_cost / agg_shares if agg_shares else float("nan")
            pnl = agg_shares * (last - avg)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(tr(lang, "ws.shares"), f"{agg_shares:,.0f}")
            c2.metric(tr(lang, "ws.avg_price"), f"${avg:,.2f}")
            c3.metric(tr(lang, "ws.mkt_value"), f"${agg_shares * last:,.2f}")
            c4.metric(tr(lang, "ws.unreal_pnl"), f"${pnl:,.2f}",
                      f"{pnl / abs(agg_cost) * 100:+.2f}%" if agg_cost else None)
        else:
            st.caption(tr(lang, "ws.not_held"))
    with tab_news:
        @st.cache_data(ttl=900, show_spinner=False)
        def _news(symbol: str) -> list:
            from quantai.data.news import NewsFetcher

            return [n.as_dict() for n in NewsFetcher().fetch_symbol_news(symbol, limit=8)]

        for n in _news(sym) or []:
            ts = n["published"].strftime("%m-%d %H:%M") if n["published"] else "?"
            st.markdown(f"- [{n['title']}]({n['link']})  `{ts} UTC`")


def _render_watchlist_page(st) -> None:  # pragma: no cover - 需 streamlit 运行时
    from datetime import datetime, timedelta

    from quantai.analysis import realized_volatility, rsi
    from quantai.config import load_config
    from quantai.data.watchlist import add_symbol, load_watchlist, remove_symbol

    wl_file = load_config().portfolio.watchlist_file
    symbols = load_watchlist(wl_file)

    # 添加/删除
    c1, c2 = st.columns([3, 2])
    new_sym = c1.text_input("添加标的（yfinance 代码，加密币如 BTC-USD）", "")
    if c1.button("➕ 添加") and new_sym.strip():
        import yfinance as yf

        sym = new_sym.strip().upper()
        probe = yf.Ticker(sym).history(period="5d")  # 先验证代码有效，无效不入表
        if probe.empty:
            st.error(f"{sym} 在 yfinance 取不到数据，未添加（检查代码拼写）")
        else:
            symbols = add_symbol(sym, wl_file)
            st.success(f"已添加 {sym}")
            st.cache_data.clear()
    rm = c2.multiselect("移除标的", symbols)
    if c2.button("🗑 移除") and rm:
        for s in rm:
            symbols = remove_symbol(s, wl_file)
        st.cache_data.clear()
        st.rerun()

    if not symbols:
        st.info(f"自选股为空。上方添加，或参照 watchlist.example.yaml 编辑 `{wl_file}`。")
        return

    @st.cache_data(ttl=300, show_spinner="拉取自选股行情…")
    def _quotes(syms: tuple, start: str) -> list[dict]:
        from quantai.data.prices import PriceFetcher

        prices = PriceFetcher().fetch_prices(list(syms), start)
        rows = []
        for sym in syms:
            df = prices.get(sym)
            if df is None or df.empty or "close" not in df.columns:
                rows.append({"标的": sym, "现价": None, "日涨跌%": None, "RSI14": None, "20D波动%": None})
                continue
            close = df["close"].astype(float).dropna()
            if close.empty:
                continue
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else float("nan")
            rows.append(
                {
                    "标的": sym,
                    "现价": round(last, 2),
                    "日涨跌%": round((last / prev - 1) * 100, 2) if prev == prev else None,
                    "RSI14": round(float(rsi(close, 14).iloc[-1]), 1) if len(close) > 14 else None,
                    "20D波动%": round(float(realized_volatility(close, 20).iloc[-1]) * 100, 1)
                    if len(close) > 20
                    else None,
                }
            )
        return rows

    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    st.dataframe(_quotes(tuple(symbols), start), use_container_width=True, height=560)
    st.caption("日线级数据（最新收盘，非实时 tick）；新上市/加密标的历史不足时指标为空。")


def _render_analyst_page(st) -> None:  # pragma: no cover - 需 streamlit 运行时
    from pathlib import Path

    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
    st.markdown(tr(lang, "ai.desc"))

    use_llm = st.toggle(tr(lang, "ai.toggle"), value=False)
    llm = _resident_llm(st) if use_llm else None

    from quantai.agents.reporting import make_daily_report

    if st.button(tr(lang, "ai.gen_llm" if use_llm else "ai.gen_plain")):
        with st.spinner(tr(lang, "ai.working")):
            make_daily_report(llm, log=lambda m: None)
        st.rerun()

    # 按修改时间倒序：新报告永远排第一个（旧版按文件名排，刚生成的盘中报告会被埋进列表）
    reports = sorted(Path("data/reports").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        pick = st.selectbox(tr(lang, "ai.history"), [p.name for p in reports])
        st.markdown((Path("data/reports") / pick).read_text(encoding="utf-8"))
    else:
        st.caption(tr(lang, "ai.none"))


def _render_ytd_replay(st) -> None:  # pragma: no cover - 需 streamlit 运行时
    """模拟盘·年初至今回放：真实历史行情 × 规则策略（长/平），一眼看到"如果年初这么跑"。

    诚实口径：策略 = SignalGenerator 组合信号（趋势30%+动量30%+均线交叉20%+突破20%，
    composite>0.1 持有该标的等权份额，否则空仓）；next_open 成交 + 0.1% 换手成本；
    热身数据取 2 年保证 MA200 有效。这是规则层回放，不是 agent 全链路
    （RL 门控无模型时按设计拒单，回放模式绕开它以给出可看的结果）。
    """
    from pathlib import Path

    from quantai.config import load_config
    from quantai.data.watchlist import load_watchlist
    from quantai.portfolio import load_portfolio

    cfg = load_config().portfolio
    held = (
        load_portfolio(cfg.file).symbols if Path(cfg.file).exists() else []
    )
    watch = load_watchlist(cfg.watchlist_file)
    default = ",".join(list(dict.fromkeys(held + watch))[:5]) or "SPY,NVDA,TSLA"

    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
    c1, c2 = st.columns([3, 1])
    tickers_in = c1.text_input(tr(lang, "sim.tickers"), value=default)
    cash = c2.number_input(tr(lang, "sim.cash"), value=100_000.0, step=10_000.0, min_value=1_000.0)
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in tickers_in.split(",") if t.strip()))
    if not tickers:
        st.info(tr(lang, "sim.need_ticker"))
        return

    @st.cache_data(ttl=1800, show_spinner="回放 2026 年初至今（真实行情 × 规则策略）…")
    def _replay(tkrs: tuple, cash_: float):
        import pandas as pd

        from quantai.analysis import drawdown_curve
        from quantai.backtest import run_backtest
        from quantai.data.prices import PriceFetcher
        from quantai.signals.generator import SignalGenerator

        def _norm(df):
            # 统一到纽约墙钟的自然日：BTC-USD 等 crypto 的索引带 UTC 时区，直接和
            # 股票（NY 时区）concat 会错位出"开局只有部分资金入场"的假跳变（审查实锤）
            idx = df.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            out = df.copy()
            out.index = idx.normalize()
            return out[~out.index.duplicated(keep="last")]

        # 热身取自 2024-01-01：确保 2026-01-01 前有足量 bar 喂 MA200
        raw = PriceFetcher().fetch_prices(list(tkrs) + ["SPY"], "2024-01-01")
        prices = {k: _norm(v) for k, v in raw.items() if v is not None and not v.empty}
        gen = SignalGenerator()

        # 先筛出可回放的标的，再按**存活数**分资金——否则被跳过的份额会被当成
        # 100% 亏损计入总收益（审查实锤）
        usable, rows = [], []
        for s in tkrs:
            df = prices.get(s)
            if df is None or "close" not in df.columns:
                rows.append({"标的": s, "状态": "取不到行情，跳过（不占份额）"})
                continue
            pre_ytd = int((df.index < "2026-01-01").sum())
            if pre_ytd < 210:
                rows.append({"标的": s, "状态": f"2026 前仅 {pre_ytd} 根 bar，MA200 热身不足，跳过（不占份额）"})
                continue
            if df.loc["2026-01-01":].empty:
                rows.append({"标的": s, "状态": "无 YTD 数据，跳过（不占份额）"})
                continue
            usable.append(s)
        if not usable:
            return None
        alloc = cash_ / len(usable)

        curves, positions = {}, []
        for s in usable:
            df = prices[s]
            sig = gen.generate(df)
            w = (sig["composite_signal"] > 0.1).astype(float)
            ytd = df.loc["2026-01-01":]
            res = run_backtest(ytd, w.loc[ytd.index], cost_per_turnover=0.001,
                               initial_capital=alloc)
            curves[s] = res.equity
            flips = int((w.loc[ytd.index].diff().fillna(0) != 0).sum())
            in_pos = bool(w.iloc[-1] > 0)
            rows.append({
                "标的": s,
                "YTD 收益": f"{(res.equity.iloc[-1] / alloc - 1) * 100:+.1f}%",
                "最大回撤": f"{res.metrics.max_drawdown * 100:.1f}%",
                "Sharpe": f"{res.metrics.sharpe:.2f}",
                "调仓次数": flips,
                "当前状态": "持有" if in_pos else "空仓",
            })
            if in_pos:
                positions.append({"标的": s, "份额市值": float(res.equity.iloc[-1])})

        # 合并到日期并集：某标的尚未开始的日期用其初始份额（现金态）填充，
        # 绝不 ffill 出"部分资金凭空消失"的开局
        union = None
        for eq in curves.values():
            union = eq.index if union is None else union.union(eq.index)
        total = None
        for eq in curves.values():
            aligned = eq.reindex(union).ffill().fillna(alloc)
            total = aligned if total is None else total + aligned

        spy = prices.get("SPY")
        bench = None
        if spy is not None and not spy.empty:
            sc = spy.loc["2026-01-01":]["close"].dropna()
            if len(sc):
                bench = (sc / sc.iloc[0] * cash_).reindex(union).ffill().fillna(float(cash_))
        dd = float(drawdown_curve(total).min())
        return {"total": total, "bench": bench, "rows": rows, "positions": positions,
                "mdd": dd, "n_usable": len(usable)}

    out = _replay(tickers, float(cash))
    if not out:
        st.error(tr(lang, "sim.no_data"))
        return

    import pandas as pd

    total = out["total"]
    ret = (float(total.iloc[-1]) / float(cash) - 1) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tr(lang, "sim.equity"), f"${float(total.iloc[-1]):,.0f}", f"{ret:+.1f}% YTD")
    m2.metric(tr(lang, "sim.initial"), f"${cash:,.0f}")
    m3.metric(tr(lang, "sim.mdd"), f"{out['mdd'] * 100:.1f}%")
    if out["bench"] is not None:
        m4.metric(tr(lang, "sim.spy_bh"), f"{(float(out['bench'].iloc[-1]) / cash - 1) * 100:+.1f}%")

    chart = pd.DataFrame({tr(lang, "sim.strategy"): total})
    if out["bench"] is not None:
        chart[tr(lang, "sim.spy_bh")] = out["bench"]
    st.line_chart(chart, use_container_width=True)

    st.subheader(tr(lang, "sim.positions"))
    if out["positions"]:
        st.dataframe(out["positions"], use_container_width=True, hide_index=True)
    else:
        st.caption(tr(lang, "sim.all_flat"))
    st.subheader(tr(lang, "sim.per_symbol"))
    st.dataframe(out["rows"], use_container_width=True, hide_index=True)
    st.caption(tr(lang, "sim.footnote"))


def main() -> None:  # pragma: no cover - 需 streamlit 运行时
    import os
    from pathlib import Path

    import streamlit as st

    from quantai.ui.client import QuantAIClient

    st.set_page_config(page_title="QuantAI Dashboard", layout="wide")
    # 全局視覺微调：藏 streamlit 默认头/脚、收紧上边距（券商终端式信息密度）
    st.markdown(
        """<style>
  header[data-testid="stHeader"] {background: transparent; height: 0.8rem;}
  .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px;}
  div[data-testid="stMetricValue"] {font-size: 1.35rem;}
  #MainMenu, footer, div[data-testid="stToolbar"] {visibility: hidden;}
</style>""",
        unsafe_allow_html=True,
    )

    from quantai.ui.i18n import LANGS, tr

    lang_label = st.sidebar.radio(
        "语言 / Language", list(LANGS), index=0, horizontal=True, key="ui_lang_label"
    )
    lang = LANGS[lang_label]
    st.session_state["ui_lang"] = lang

    _PAGE_KEYS = ["workstation", "portfolio", "watchlist", "analyst", "sim"]
    page_key = st.sidebar.radio(
        tr(lang, "nav.pages"), _PAGE_KEYS,
        format_func=lambda k: tr(lang, f"nav.{k}"), index=0, key="ui_page",
    )
    # 旧路由沿用中文页名（组合分析/自选股页内部文案本轮仍为中文，README 已注明）
    page = {"workstation": "行情工作台", "portfolio": "组合分析", "watchlist": "自选股",
            "analyst": "AI 分析", "sim": "模拟盘"}[page_key]

    # Tableau 一键打开（找 tableau/ 下的 .twb/.twbx；没有就开导出目录）
    tb_files = list(Path("tableau").glob("*.twb*"))
    label = (tr(lang, "nav.tableau_open", name=tb_files[0].name) if tb_files
             else tr(lang, "nav.tableau_dir"))
    if st.sidebar.button(label):
        os.startfile(str(tb_files[0]) if tb_files else str(Path("tableau/exports").resolve()))

    if page == "行情工作台":
        _render_workstation_page(st)
        return
    if page == "组合分析":
        st.title("QuantAI · 真实组合分析" if lang == "zh" else "QuantAI · Portfolio")
        _render_portfolio_page(st)
        return
    if page == "自选股":
        st.title("QuantAI · 自选股" if lang == "zh" else "QuantAI · Watchlist")
        _render_watchlist_page(st)
        return
    if page == "AI 分析":
        st.title(tr(lang, "ai.title"))
        _render_analyst_page(st)
        return

    st.title(tr(lang, "sim.title"))
    _MODES = ["ytd", "live"]
    mode = st.radio(
        tr(lang, "sim.mode"), _MODES,
        format_func=lambda m: tr(lang, f"sim.mode_{m}"), horizontal=True,
    )
    if mode == "ytd":
        _render_ytd_replay(st)
        return

    base_url = st.sidebar.text_input("API 地址", value="http://localhost:8000")
    client = QuantAIClient(base_url)

    # 健康检查
    try:
        client.health()
        st.sidebar.success("API 在线")
    except Exception as exc:
        st.sidebar.error(f"API 不可达：{exc}")
        st.info(
            "实盘会话页需要 QuantAI API（纸面交易，零真实资金）。启动方式任选：\n\n"
            "1. 双击 `启动仪表盘.bat`（新版会自动拉起 API）；\n"
            "2. 或另开终端执行 `venv311\\Scripts\\python.exe scripts\\serve.py --host 127.0.0.1`，"
            "然后回来点侧栏任意控件重试。"
        )
        st.stop()

    # 实盘控制
    st.sidebar.header("纸面会话控制")
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

    # 真实状态（5 秒自刷：不用手动刷新页面看会话进展）
    @st.fragment(run_every=5)
    def _live_session_view():
        from datetime import datetime as _dt

        status: Dict[str, Any] = client.live_status()
        if not status.get("active"):
            st.info(
                "当前无纸面会话，用左侧启动。诚实说明：会话由多 agent 决策链驱动，"
                "RL 门控（Gatekeeper）在没有训练好的 RL 模型时**按设计拒绝交易**——"
                "所以纸面会话可能长期空仓观望，这是诚实拒单不是故障。"
                "想立刻看到有仓位、有成交的结果，用「年初至今回放」模式。"
            )
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
            st.caption("暂无持仓（决策链尚未放行任何交易）")

        st.subheader("最近成交")
        trades = client.live_trades(limit=50).get("trades", [])
        if trades:
            st.dataframe(trades)
        else:
            st.caption("暂无成交")

        st.subheader("数据飞轮")
        st.json(client.active_adapters())
        st.caption(f"状态自刷 5s/轮 · 最后更新 {_dt.now():%H:%M:%S}")

    _live_session_view()


if __name__ == "__main__":
    main()
