"""quantai.ui —— 表现层（薄客户端）。

只做展示/控制，**无业务逻辑**（业务在 agents/live/evolution，HTTP 在 api/）。
- `client.QuantAIClient`：对 `quantai.api` 的薄客户端（可单测，可注入 ASGI transport）。
- `streamlit_app`：薄 Streamlit 仪表盘（`streamlit run`），只展示 API 的真实状态。

桌面 PySide6 + Live2D 为旧版表现壳（表现壳，留 legacy）。
"""
from __future__ import annotations

from quantai.ui.client import QuantAIClient

__all__ = ["QuantAIClient"]
