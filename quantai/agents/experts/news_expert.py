"""News 专家：有新新闻且 news adapter 可用时路由到的事件驱动专家。"""
from __future__ import annotations

from quantai.agents.experts.base import LLMExpert


class NewsExpert(LLMExpert):
    name = "news"
    default_adapter = "news"
