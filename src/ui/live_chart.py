# src/ui/live_chart.py
"""
Phase 3.4: Live Paper Trading Chart with Buy/Sell Markers

Usage:
    streamlit run src/ui/live_chart.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Live Paper Trading",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}
.buy-marker { color: #00ff88; font-weight: bold; }
.sell-marker { color: #ff4444; font-weight: bold; }
.chat-bubble {
    background: rgba(78, 140, 255, 0.15);
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
    border-left: 3px solid #4e8cff;
}
</style>
""", unsafe_allow_html=True)


def fetch_json(url: str, default=None):
    """Fetch JSON from API"""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return default


def render_price_chart(ticker: str, prices: list, trades: list):
    """Render candlestick chart with buy/sell markers"""
    if not prices:
        st.warning(f"No price data for {ticker}")
        return
    
    df = pd.DataFrame(prices)
    df["time"] = pd.to_datetime(df["time"])
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=ticker,
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
    ))
    
    # Add buy markers (green triangles)
    buy_trades = [t for t in trades if t.get("ticker") == ticker and t.get("action") == "BUY"]
    if buy_trades:
        buy_times = [t.get("time") for t in buy_trades]
        buy_prices = [t.get("price") for t in buy_trades]
        fig.add_trace(go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=15,
                color="#00ff88",
                line=dict(width=2, color="white"),
            ),
            name="BUY",
            hovertemplate="BUY @ $%{y:.2f}<extra></extra>",
        ))
    
    # Add sell markers (red triangles)
    sell_trades = [t for t in trades if t.get("ticker") == ticker and t.get("action") == "SELL"]
    if sell_trades:
        sell_times = [t.get("time") for t in sell_trades]
        sell_prices = [t.get("price") for t in sell_trades]
        fig.add_trace(go.Scatter(
            x=sell_times,
            y=sell_prices,
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=15,
                color="#ff4444",
                line=dict(width=2, color="white"),
            ),
            name="SELL",
            hovertemplate="SELL @ $%{y:.2f}<extra></extra>",
        ))
    
    # Layout
    fig.update_layout(
        title=f"{ticker} Live Chart",
        template="plotly_dark",
        height=500,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_chat_panel(messages: list):
    """Render Mari's chat messages"""
    st.subheader("Mari Chat")
    
    if not messages:
        st.info("Waiting for Mari's commentary...")
        return
    
    for msg in reversed(messages[-10:]):
        time_str = msg.get("time", "")[:19].replace("T", " ")
        text = msg.get("text", "")
        st.markdown(f"""
        <div class="chat-bubble">
            <small style="color: #888;">{time_str}</small><br>
            <b>Mari:</b> {text}
        </div>
        """, unsafe_allow_html=True)


def main():
    st.title("Live Paper Trading Dashboard")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh", type="primary"):
            st.rerun()
    
    # Fetch status
    status = fetch_json(f"{API_BASE}/live/status", {"active": False})
    
    if not status.get("active"):
        st.error("No active paper trading session")
        st.info("Start the engine: `python scripts/run_live_paper_trading.py`")
        return
    
    # Status metrics
    st.markdown("### Portfolio Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cash", f"${status.get('cash', 0):,.2f}")
    with col2:
        positions = status.get("positions", {})
        st.metric("Positions", len(positions))
    with col3:
        st.metric("Trades", status.get("trade_count", 0))
    with col4:
        tickers = status.get("tickers", [])
        st.metric("Tickers", len(tickers))
    
    # Fetch trades for markers
    trades_data = fetch_json(f"{API_BASE}/live/trades", {"trades": []})
    trades = trades_data.get("trades", [])
    
    # Charts for each ticker
    st.markdown("### Live Charts")
    tickers = status.get("tickers", ["NVDA", "TSLA", "AAPL"])
    
    # Ticker selector
    selected_ticker = st.selectbox("Select Ticker", tickers, index=0)
    
    # Fetch and render chart
    chart_data = fetch_json(f"{API_BASE}/live/chart/{selected_ticker}?limit=100", {"prices": []})
    prices = chart_data.get("prices", [])
    
    render_price_chart(selected_ticker, prices, trades)
    
    # Trade history
    st.markdown("### Recent Trades")
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df = trade_df[["time", "ticker", "action", "price", "shares"]]
        st.dataframe(trade_df.tail(10), use_container_width=True)
    else:
        st.info("No trades yet")
    
    # Mari chat panel
    st.markdown("---")
    chat_data = fetch_json(f"{API_BASE}/live/chat?limit=20", {"messages": []})
    render_chat_panel(chat_data.get("messages", []))
    
    # Auto-refresh
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
