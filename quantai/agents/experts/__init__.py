"""MoE 专家集合：scalper / analyst / news。

`make_expert(name)` 把 `HeuristicRouter.route()` 输出的专家名映射成专家实例；
未知名一律回退 scalper（与旧 `_model_infer` 的 adapter 兜底一致）。
"""
from __future__ import annotations

from typing import Optional

from quantai.agents.experts.analyst import AnalystExpert
from quantai.agents.experts.base import DEFAULT_SYSTEM_PROMPT, LLMExpert
from quantai.agents.experts.news_expert import NewsExpert
from quantai.agents.experts.scalper import ScalperExpert

_REGISTRY: dict[str, type[LLMExpert]] = {
    "scalper": ScalperExpert,
    "analyst": AnalystExpert,
    "news": NewsExpert,
}


def make_expert(
    name: str, *, adapter: Optional[str] = None, system_prompt: Optional[str] = None
) -> LLMExpert:
    cls = _REGISTRY.get(str(name or "").strip().lower(), ScalperExpert)
    return cls(adapter=adapter, system_prompt=system_prompt)


__all__ = [
    "LLMExpert",
    "ScalperExpert",
    "AnalystExpert",
    "NewsExpert",
    "DEFAULT_SYSTEM_PROMPT",
    "make_expert",
]
