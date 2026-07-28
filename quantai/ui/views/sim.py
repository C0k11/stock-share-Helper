"""模拟盘页：这套规则历史上跑得如何？（YTD 参数化回放 + 实盘纸面会话，纯平移自 streamlit_app.py）"""
from __future__ import annotations

from typing import Any, Dict

from quantai.ui.common import fmt_money as _fmt_money


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
    # 默认只回放真实持仓（其它标的手动输入）——"我的模拟盘"先讲我的票
    default = ",".join(dict.fromkeys(held)) or "SPY"

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
    # 口径提级到图上方（原页脚 footnote 上移——读者先看到算法再看到曲线）
    st.caption(tr(lang, "sim.footnote"))
    st.line_chart(chart, use_container_width=True)

    st.subheader(tr(lang, "sim.positions"))
    if out["positions"]:
        st.dataframe(out["positions"], use_container_width=True, hide_index=True)
    else:
        st.caption(tr(lang, "sim.all_flat"))
    st.subheader(tr(lang, "sim.per_symbol"))
    st.dataframe(out["rows"], use_container_width=True, hide_index=True)


def render() -> None:  # pragma: no cover - 需 streamlit 运行时
    import streamlit as st

    from quantai.ui.client import QuantAIClient
    from quantai.ui.i18n import tr

    lang = st.session_state.get("ui_lang", "zh")
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
