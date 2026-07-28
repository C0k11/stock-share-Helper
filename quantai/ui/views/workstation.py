"""行情工作台页：现在该不该动仓位？（实时作战，纯平移自 streamlit_app.py）"""
from __future__ import annotations

from quantai.ui.common import TIMEFRAMES as _TIMEFRAMES


def render() -> None:  # pragma: no cover - 需 streamlit 运行时
    from pathlib import Path

    import streamlit as st

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

        from quantai.ui import theme

        up = chg >= 0
        color = theme.UP if up else theme.DOWN
        # 自定义头部（WS 式：大价格 + 内联涨跌；不用 st.metric——$ 会触发 LaTeX、涨跌不能内联）
        st.markdown(
            f"""<div style="line-height:1.15;margin:2px 0 6px 0">
  <span style="color:{theme.TEXT_MUTED};font-size:0.95rem">{sym}</span><br>
  <span style="font-size:2.3rem;font-weight:700">${last:,.2f}</span>
  <span style="color:{color};font-size:1.05rem;font-weight:600;margin-left:8px">
    {chg:+,.2f} ({chg_pct:+.2f}%) <span style="color:{theme.TEXT_MUTED};font-weight:400">{tf}</span>
  </span><br>
  <span style="color:{theme.TEXT_MUTED};font-size:0.85rem">
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
            # 轻交互：无工具栏 + 滚轮缩放（配合图内十字线统一悬浮读数）
            config={"displayModeBar": False, "scrollZoom": True},
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
        default=list(dict.fromkeys(held)) or list(dict.fromkeys(watch[:1])),
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
        _held_sh = {
            s2: sum(p.shares for p in (portfolio.positions if portfolio else []) if p.symbol == s2)
            for s2 in held
        }
        daemon.configure(tact_syms, held_cost, lang=lang, held_shares=_held_sh)

        @st.fragment(run_every=15)
        def _ai_view():
            from datetime import datetime as _dt

            stt = daemon.state
            if stt.get("out"):
                st.markdown(stt["out"])
                if stt.get("out_lang") and stt.get("out_lang") != lang:
                    st.info(tr(lang, "ws.ai_lang_switch"))
                st.caption(tr(
                    lang, "ws.ai_status", model=stt.get("model", "?"),
                    round=stt.get("round", 0),
                    ts=f"{_dt.fromtimestamp(stt['ts']):%H:%M:%S}",
                    scored=stt.get("scored", 0), status=tr(lang, "st." + stt.get("status", "running")),
                ))
            else:
                st.info(tr(lang, "ws.ai_waiting", status=tr(lang, "st." + stt.get("status", "starting"))))
            if stt.get("error"):
                st.warning(tr(lang, "ws.ai_error", err=stt["error"]))

        _ai_view()

    # ================= 对冲台（期权：BS 引擎确定性计算） =================
    st.subheader(tr(lang, "hd.title"))
    st.caption(tr(lang, "hd.caption"))
    hc1, hc2, hc3, hc4 = st.columns([1.6, 1.1, 1.1, 1.1])
    hd_sym = hc1.selectbox(tr(lang, "hd.symbol"), universe, key="hd_sym")
    _real_sh = sum(p.shares for p in (portfolio.positions if portfolio else []) if p.symbol == hd_sym)
    hd_shares = hc2.number_input(
        tr(lang, "hd.shares"), value=float(_real_sh) if _real_sh >= 1 else 100.0,
        min_value=1.0, step=10.0, key="hd_shares",
    )
    hd_floor = hc3.slider(tr(lang, "hd.floor"), 0.80, 0.98, 0.92, 0.01, key="hd_floor")
    hd_target = hc4.slider(tr(lang, "hd.target"), 1.02, 1.20, 1.06, 0.01, key="hd_target")

    @st.cache_data(ttl=300, show_spinner=False)
    def _chain(sym2: str):
        from quantai.data.options_chain import OptionChainFetcher

        return OptionChainFetcher().fetch(sym2)

    @st.fragment(run_every=120)
    def _hedge_desk():
        from quantai.analysis.options import (
            chain_stats, covered_call_plan, protective_put_plan,
        )

        ch = _chain(hd_sym)
        if ch is None:
            st.info(tr(lang, "hd.no_chain", sym=hd_sym))
            return
        m1h = _hist(hd_sym, "1d", "1m")
        dfh = _hist(hd_sym, "10d", "1d")
        spot = None
        if m1h is not None and not m1h.empty:
            spot = float(m1h["close"].dropna().iloc[-1])
        elif dfh is not None and not dfh.empty:
            spot = float(dfh["close"].dropna().iloc[-1])
        if not spot:
            st.info(tr(lang, "hd.no_chain", sym=hd_sym))
            return
        stats = chain_stats(ch["calls"], ch["puts"], spot)
        pc = stats.get("pc_volume_ratio")
        st.caption(tr(
            lang, "hd.stats",
            iv=f"{stats['atm_iv']:.0%}" if stats.get("atm_iv") is not None else "n/a",
            pc=f"{pc:.2f}" if pc is not None else "n/a",
            expiry=ch["expiry"], days=ch["days_to_expiry"],
        ))
        pp = protective_put_plan(hd_shares, spot, ch["puts"], ch["days_to_expiry"], floor_pct=hd_floor)
        cc = covered_call_plan(hd_shares, spot, ch["calls"], ch["days_to_expiry"], target_pct=hd_target)
        colp, colc = st.columns(2)
        with colp:
            st.markdown(f"**{tr(lang, 'hd.pp_title')}**")
            if pp:
                unc = tr(lang, "hd.uncovered", n=f"{pp['uncovered_shares']:.0f}") if pp["uncovered_shares"] else ""
                st.write(tr(
                    lang, "hd.pp_line", n=pp["contracts"], days=pp["days_to_expiry"],
                    strike=f"{pp['strike']:.0f}", prem=f"{pp['premium']:.2f}",
                    cost=f"{pp['cost']:,.0f}", cost_pct=f"{pp['cost_pct']:.1%}",
                    maxloss=f"{pp['max_loss_pct_covered']:.1%}", uncovered=unc,
                ))
            elif hd_shares < 100:
                st.info(tr(lang, "hd.lt100", shares=f"{hd_shares:.0f}"))
            else:
                st.caption(tr(lang, "hd.no_quote"))
        with colc:
            st.markdown(f"**{tr(lang, 'hd.cc_title')}**")
            if cc:
                st.write(tr(
                    lang, "hd.cc_line", n=cc["contracts"], strike=f"{cc['strike']:.0f}",
                    prem=f"{cc['premium']:.2f}", income=f"{cc['income']:,.0f}",
                    income_pct=f"{cc['income_pct']:.1%}", ann=f"{cc['annualized_pct']:.0%}",
                    cap=f"{cc['upside_capped_pct']:.1%}",
                ))
            elif hd_shares < 100:
                st.info(tr(lang, "hd.lt100", shares=f"{hd_shares:.0f}"))
            else:
                st.caption(tr(lang, "hd.no_quote"))

    _hedge_desk()

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
