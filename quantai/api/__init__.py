"""quantai.api —— 干净 FastAPI 层（只暴露真实状态/操作）。

取代 6481 行 `src/api/main.py`：拆成 `app`（工厂）+ `deps`（ApiState）+ `schemas` + `routes/`。
**删光假数据 endpoint**（C-2：regime/recommendations/performance/alerts/news_summary；C-6：movers）。

用法：
    from quantai.api import create_app
    app = create_app()
"""
from __future__ import annotations

from quantai.api.app import app, create_app
from quantai.api.deps import ApiState, get_state, set_state

__all__ = ["create_app", "app", "ApiState", "get_state", "set_state"]
