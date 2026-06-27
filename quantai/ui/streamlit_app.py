"""薄 Streamlit 仪表盘 —— 读 `quantai.api` 的真实状态，控制实盘会话。

诚实：**只展示来自 API 的真实数据**（账户快照 / 成交 / active adapter）。旧版那些假数据面板
（regime/recommendations/performance/alerts，C-2）不再存在。

跑：
    venv311\\Scripts\\python.exe -m streamlit run quantai/ui/streamlit_app.py
    （需先 `python scripts/serve.py` 起 API；或改 sidebar 的 API 地址。）

本模块所有业务调用都走 `quantai.ui.client.QuantAIClient`（可单测）；streamlit 仅做渲染，
故 `import` 本文件无副作用（streamlit 在 `main()` 内懒导入）。
"""
from __future__ import annotations

from typing import Any, Dict


def _fmt_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def main() -> None:  # pragma: no cover - 需 streamlit 运行时
    import streamlit as st

    from quantai.ui.client import QuantAIClient

    st.set_page_config(page_title="QuantAI Dashboard", layout="wide")
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
