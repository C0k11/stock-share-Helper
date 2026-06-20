"""Analyst 专家：高波动/有新闻时路由到的分析型专家（通常挂 DPO 对齐后的 adapter）。"""
from __future__ import annotations

from quantai.agents.experts.base import LLMExpert


class AnalystExpert(LLMExpert):
    name = "analyst"
    default_adapter = "analyst"
