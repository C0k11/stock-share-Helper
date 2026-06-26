"""FastAPI 应用工厂 —— 干净重写，取代 6481 行的 `src/api/main.py`。

只挂真实路由：health / live / feedback / evolution / chat（全部 `/api/v1` 前缀）。
**不含任何假数据 endpoint**（C-2 的 regime/recommendations/performance/alerts/news_summary、
C-6 的 movers 一律不迁）。重活（LLM/实盘线程）通过 `ApiState` 注入/按需启动。

用法：
    from quantai.api import create_app
    app = create_app()                      # 纯默认配置
    # uvicorn quantai.api.app:app  或  scripts/serve.py
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quantai.api.deps import ApiState, set_state
from quantai.api.routes import chat, evolution, feedback, health, live
from quantai.config.schema import AppConfig


def create_app(cfg: Optional[AppConfig] = None, *, llm: Any = None) -> FastAPI:
    set_state(ApiState(cfg, llm=llm))

    app = FastAPI(title="QuantAI API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api/v1")
    api.include_router(health.router)
    api.include_router(live.router)
    api.include_router(feedback.router)
    api.include_router(evolution.router)
    api.include_router(chat.router)
    app.include_router(api)

    @app.get("/")
    def root() -> dict:
        return {"service": "quantai-api", "version": "0.2.0", "docs": "/docs"}

    return app


# 给 `uvicorn quantai.api.app:app` 用的默认实例。
app = create_app()
