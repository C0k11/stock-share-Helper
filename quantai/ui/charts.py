"""专业图表构建（plotly，纯函数层）：K 线 + 指标叠加、RSI/MACD 副图、盈亏条形。

设计：本模块**只产 plotly Figure**（输入 DataFrame/快照，无 streamlit、无网络），
故可离线单测（断言 trace 结构/主题），渲染层（streamlit_app）只管 `st.plotly_chart`。
指标全部复用 `quantai.analysis`（与 CLI/仓库同一套数字）。

主题：暗色专业风（moomoo 式配色）：涨 #26A69A（青绿）/ 跌 #EF5350（红），
背景 #131722，网格低对比。美股惯例：绿涨红跌。
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantai.analysis import bollinger, macd, rsi, sma

UP_COLOR = "#26A69A"
DOWN_COLOR = "#EF5350"
_BG = "#131722"
_GRID = "#1F2A38"
_TEXT = "#B2B5BE"

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, size=12),
    xaxis=dict(gridcolor=_GRID, rangeslider=dict(visible=False)),
    yaxis=dict(gridcolor=_GRID),
    margin=dict(l=40, r=20, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
)


def candlestick_figure(
    df: pd.DataFrame,
    symbol: str,
    ma_windows: Sequence[int] = (20, 50),
    show_bollinger: bool = True,
) -> go.Figure:
    """K 线蜡烛图 + 均线/布林叠加 + 成交量副图。

    df 需含 open/high/low/close（volume 可选）。指标热身期自然缺画（NaN 不连线）。
    """
    has_volume = "volume" in df.columns
    fig = make_subplots(
        rows=2 if has_volume else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22] if has_volume else [1.0],
        vertical_spacing=0.02,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DOWN_COLOR,
        ),
        row=1,
        col=1,
    )
    close = df["close"].astype(float)
    for w in ma_windows:
        fig.add_trace(
            go.Scatter(x=df.index, y=sma(close, w), name=f"MA{w}", mode="lines", line=dict(width=1)),
            row=1,
            col=1,
        )
    if show_bollinger:
        bb = bollinger(close)
        fig.add_trace(
            go.Scatter(x=df.index, y=bb["bb_upper"], name="BB上轨", mode="lines",
                       line=dict(width=1, dash="dot", color="#5C6BC0")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=bb["bb_lower"], name="BB下轨", mode="lines",
                       line=dict(width=1, dash="dot", color="#5C6BC0"),
                       fill="tonexty", fillcolor="rgba(92,107,192,0.08)"),
            row=1, col=1,
        )
    if has_volume:
        colors = [UP_COLOR if c >= o else DOWN_COLOR for o, c in zip(df["open"], df["close"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="成交量", marker_color=colors, opacity=0.6),
            row=2,
            col=1,
        )
    fig.update_layout(title=f"{symbol} 日线", **DARK_LAYOUT)
    return fig


def rsi_macd_figure(df: pd.DataFrame) -> go.Figure:
    """RSI(14) + MACD 副图（两行）。超买/超卖参考线 70/30。"""
    close = df["close"].astype(float)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("RSI(14)", "MACD(12,26,9)"))
    fig.add_trace(go.Scatter(x=df.index, y=rsi(close, 14), name="RSI", mode="lines",
                             line=dict(color="#FFB74D", width=1.5)), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=DOWN_COLOR, opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=UP_COLOR, opacity=0.5, row=1, col=1)

    m = macd(close)
    hist_colors = [UP_COLOR if (v == v and v >= 0) else DOWN_COLOR for v in m["macd_histogram"]]
    fig.add_trace(go.Bar(x=df.index, y=m["macd_histogram"], name="Hist",
                         marker_color=hist_colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=m["macd"], name="MACD", mode="lines",
                             line=dict(color="#4FC3F7", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=m["macd_signal"], name="Signal", mode="lines",
                             line=dict(color="#F06292", width=1.2)), row=2, col=1)
    fig.update_layout(height=420, **DARK_LAYOUT)
    return fig


def pnl_bar_figure(positions: list[dict]) -> go.Figure:
    """每标的未实现盈亏条形图（输入 `PositionSnapshot.as_dict()` 列表）。"""
    rows = sorted(positions, key=lambda p: p.get("unrealized_pnl", 0.0))
    symbols = [p["symbol"] for p in rows]
    pnl = [p.get("unrealized_pnl", float("nan")) for p in rows]
    colors = [UP_COLOR if (v == v and v >= 0) else DOWN_COLOR for v in pnl]
    fig = go.Figure(go.Bar(x=pnl, y=symbols, orientation="h", marker_color=colors,
                           text=[f"{v:+,.0f}" if v == v else "n/a" for v in pnl],
                           textposition="outside"))
    fig.add_vline(x=0, line_color=_TEXT, opacity=0.4)
    fig.update_layout(title="未实现盈亏（USD）", height=max(220, 60 * len(rows) + 120), **DARK_LAYOUT)
    return fig
