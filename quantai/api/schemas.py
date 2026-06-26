"""API 请求/响应模型（pydantic v2）。只描述真实可用的 endpoint。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StartLiveRequest(BaseModel):
    tickers: List[str] = Field(default_factory=lambda: ["NVDA"])
    source: str = "simulated"  # simulated / yfinance / auto
    cash: float = 100000.0
    interval_sec: float = 1.0
    allow_short: bool = False
    seed: Optional[int] = None
    collect: bool = False  # True 时开启数据飞轮采集


class FeedbackRequest(BaseModel):
    ref_id: str
    score: int  # 例如 +1 / -1
    comment: str = ""


class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class OkResponse(BaseModel):
    ok: bool = True
    detail: str = ""
