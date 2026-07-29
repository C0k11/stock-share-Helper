"""quantai.ui —— 表现层（薄客户端）。

只做展示/控制，**无业务逻辑**（业务在 agents/live/evolution，HTTP 在 api/）。
- `client.QuantAIClient`：对 `quantai.api` 的薄客户端（可单测，可注入 ASGI transport）。
- `app` + `views/`：st.navigation 五页仪表盘（入口 `quantai-dashboard` 或
  `streamlit run quantai/ui/streamlit_app.py`，后者为兼容薄壳）。
"""
from __future__ import annotations

from quantai.ui.client import QuantAIClient

__all__ = ["QuantAIClient"]
