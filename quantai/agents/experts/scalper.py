"""Scalper 专家：短线/技术面默认专家（MoE 路由的兜底专家）。"""
from __future__ import annotations

from quantai.agents.experts.base import LLMExpert


class ScalperExpert(LLMExpert):
    name = "scalper"
    default_adapter = "scalper"
