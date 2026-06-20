"""Overlays：在专家方向之上叠加的修正层。

- `ChartistOverlay`：VLM 看 K 线给 +1/-1/0（默认关闭，重依赖）。
- `MacroGovernor`：全局宏观闸（诚实占位：旧版随机已删，默认中性）。
"""
from __future__ import annotations

from quantai.agents.overlays.chartist import ChartistOverlay
from quantai.agents.overlays.macro_governor import MacroGovernor
from quantai.agents.overlays.vlm_chartist import QwenVLChartist, render_candles

__all__ = ["ChartistOverlay", "MacroGovernor", "QwenVLChartist", "render_candles"]
