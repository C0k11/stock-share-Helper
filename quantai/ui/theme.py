"""QuantAI 设计令牌 —— 前端唯一颜色定义处。

与 `powerbi/theme/quantai-dark.json` 是同一套令牌（JSON 没有注释语法，双向同步
约定记录在这里与 powerbi/README）：改下面任何一个核心色，必须同步改 JSON，
反之亦然。核心四色：背景 #1B1B1F / 强调 #4C8BF5 / 涨 #26A69A / 跌 #EF5350。
`.streamlit/config.toml` 的三个背景/文字键也取自这里（toml 不能 import，值抄写）。
"""
from __future__ import annotations

# ---- 核心四色（与 Power BI 主题一一对应） ----
BG = "#1B1B1F"          # 页面/画布背景（= PBI background / 发散色阶 center）
ACCENT = "#4C8BF5"      # 强调色（= PBI tableAccent / dataColors[0]）
UP = "#26A69A"          # 涨/多头/good（= PBI good / maximum）
DOWN = "#EF5350"        # 跌/空头/bad（= PBI bad / minimum）

# ---- 派生令牌（前端专用；PBI 侧有同语义值时在注释标注） ----
BG_OUTER = "#141417"    # 侧栏/画布外（= PBI page.outspace）
TEXT = "#E8E8EC"        # 主文字（= PBI foreground）
TEXT_MUTED = "#8A8A93"  # 次级文字 / 悬浮十字线
GRID = "#26262B"        # 图表网格（低对比）
CHART_TEXT = "#B2B5BE"  # plotly 轴刻度文字
BAND = "#5C6BC0"        # 布林带/通道虚线
MA = "#FFB74D"          # 均线
MACD_FAST = "#4FC3F7"   # MACD 快线
MACD_SLOW = "#F06292"   # MACD 慢线/信号线
