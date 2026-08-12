"""自选股页：候选池里谁值得看？（快照表 + universe 管理，纯平移自 streamlit_app.py）"""
from __future__ import annotations


def render() -> None:  # pragma: no cover - 需 streamlit 运行时
    from datetime import datetime, timedelta

    import streamlit as st

    from quantai.analysis import realized_volatility, rsi
    from quantai.config import load_config
    from quantai.data.watchlist import add_symbol, load_watchlist, remove_symbol

    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
    st.title("QuantAI · 自选股" if lang == "zh" else "QuantAI - Watchlist")
    wl_file = load_config().portfolio.watchlist_file
    symbols = load_watchlist(wl_file)

    # 添加/删除
    c1, c2 = st.columns([3, 2])
    new_sym = c1.text_input(tr(lang, "wl.add_label"), "")
    if c1.button(tr(lang, "wl.add_btn")) and new_sym.strip():
        import yfinance as yf

        sym = new_sym.strip().upper()
        probe = yf.Ticker(sym).history(period="5d")  # 先验证代码有效，无效不入表
        if probe.empty:
            st.error(tr(lang, "wl.invalid", sym=sym))
        else:
            symbols = add_symbol(sym, wl_file)
            st.success(tr(lang, "wl.added", sym=sym))
            st.cache_data.clear()
    rm = c2.multiselect(tr(lang, "wl.remove_label"), symbols)
    if c2.button(tr(lang, "wl.remove_btn")) and rm:
        for s in rm:
            symbols = remove_symbol(s, wl_file)
        st.cache_data.clear()
        st.rerun()

    if not symbols:
        st.info(tr(lang, "wl.empty", file=wl_file))
        return

    @st.cache_data(ttl=300, show_spinner="拉取自选股行情…")
    def _quotes(syms: tuple, start: str) -> list[dict]:
        from quantai.data.prices import PriceFetcher

        prices = PriceFetcher().fetch_prices(list(syms), start)
        rows = []
        for sym in syms:
            df = prices.get(sym)
            if df is None or df.empty or "close" not in df.columns:
                rows.append({"标的": sym, "现价": None, "日涨跌%": None, "RSI14": None,
                             "20D已实现波动%(年化)": None})
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
                    # 口径入列名：20 个交易 bar 的样本 std（ddof=1）× sqrt(252) 年化
                    "20D已实现波动%(年化)": round(float(realized_volatility(close, 20).iloc[-1]) * 100, 1)
                    if len(close) > 20
                    else None,
                }
            )
        return rows

    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    st.dataframe(_quotes(tuple(symbols), start), use_container_width=True, height=560)
    st.caption(tr(lang, "wl.caption"))
