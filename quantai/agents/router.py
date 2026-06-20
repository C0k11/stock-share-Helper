"""MoE 路由 —— 诚实命名为 `HeuristicRouter`。

旧 `strategy.py::_moe_route` 自称 "MoE Router"，但**它不是学习到的门控网络**，
而是一组规则：高波动或有新闻 -> 路由到 analyst/news，否则 scalper。重构按
「诚实命名」原则改名 `HeuristicRouter`，行为 100% 等价迁移，纯逻辑、可单测。

输入只读特征，输出 (expert_name, meta)；不碰模型、不碰 torch。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from quantai.config.schema import RouterConfig


class HeuristicRouter:
    """规则路由：features -> {"scalper" | "analyst" | "news"}。

    路由触发条件（任一命中即升级到 analyst）：
    - 年化波动 `vol >= vol_threshold`；
    - `any_news=True` 且出现新新闻（`news_new_count > 0`）；
    - `any_news=False` 且 `|news_score| >= news_threshold`。
    仅当 news adapter 可用（`news_adapter_available=True`）时才会进一步路由到 news 专家。
    """

    def __init__(
        self,
        *,
        vol_threshold: float = 60.0,
        news_threshold: float = 0.8,
        any_news: bool = True,
        news_adapter_available: bool = False,
    ) -> None:
        self.vol_threshold = float(vol_threshold)
        self.news_threshold = float(news_threshold)
        self.any_news = bool(any_news)
        self.news_adapter_available = bool(news_adapter_available)

    @classmethod
    def from_config(
        cls, cfg: RouterConfig, *, news_adapter_available: bool = False
    ) -> "HeuristicRouter":
        return cls(
            vol_threshold=cfg.vol_threshold,
            news_threshold=cfg.news_threshold,
            any_news=cfg.any_news,
            news_adapter_available=news_adapter_available,
        )

    def route(self, features: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        vol_f = _as_float(features.get("volatility_ann_pct", 20.0))

        sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
        news_score = _as_float(sig.get("news_score"))
        news_count = _as_float(sig.get("news_count"))
        news_new_count = _as_float(sig.get("news_new_count"))

        thr_vol = self.vol_threshold
        thr_news = self.news_threshold

        use_analyst = False
        if thr_vol > 0.0 and vol_f >= thr_vol:
            use_analyst = True
        if self.any_news:
            if news_new_count > 0.0:
                use_analyst = True
        else:
            if abs(news_score) >= thr_news:
                use_analyst = True

        use_news = False
        if self.news_adapter_available:
            if self.any_news and news_new_count > 0.0:
                use_news = True
            elif abs(news_score) >= thr_news:
                use_news = True

        expert = "news" if use_news else ("analyst" if use_analyst else "scalper")

        triggers: list[str] = []
        if thr_vol > 0.0 and vol_f >= thr_vol:
            triggers.append(f"vol={vol_f:.1f}%>=thr{thr_vol:g}")
        if self.any_news and news_new_count > 0.0:
            triggers.append(f"news_new={int(news_new_count)}")
        if (not self.any_news) and abs(news_score) >= thr_news:
            triggers.append(f"news_score={news_score:+.2f}>=thr{thr_news:g}")

        meta: Dict[str, Any] = {
            "vol": vol_f,
            "expert": expert,
            "news_score": news_score,
            "news_count": news_count,
            "news_new_count": news_new_count,
            "triggers": triggers,
            "any_news": self.any_news,
            "thr_vol": thr_vol,
            "thr_news": thr_news,
        }
        return expert, meta


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
