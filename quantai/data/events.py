"""事件概率通道：Polymarket Gamma API（公开、免 key）-> 预测市场隐含概率。

价值：RSS 头条告诉系统"发生了什么"，Polymarket 赔率给出"市场用真金白银定价的
**事件概率**"（Fed 决议/宏观数据区间/公司事件），是结构化的前瞻信号。

诚实口径：
- 这是**预测市场隐含概率**，不是官方预测——进简报/仓库都带此标注；
- 只读公开 Gamma API（https://gamma-api.polymarket.com，官方文档明确无需认证），
  不碰交易端点、不涉及任何下单；
- 字段坑（实测）：`outcomes`/`outcomePrices` 可能是 JSON 字符串而非数组，两种都兼容；
  找不到 "Yes" 结果的奇异市场直接跳过（不猜方向）。

纯解析 `events_to_odds` 零网络可测；网络只在 `EventsFetcher`（getter 可注入）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import requests
from loguru import logger

GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass
class EventOdd:
    """一个可交易二元市场的概率快照。"""

    market_id: str
    condition_id: str
    event_title: str
    question: str
    yes_price: float  # 市场隐含 P(Yes)  in  [0,1]
    volume_24h: float
    end_date: Optional[str]
    category: str  # 抓取时所用的 tag

    def as_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "condition_id": self.condition_id,
            "event_title": self.event_title,
            "question": self.question,
            "yes_price": self.yes_price,
            "volume_24h": self.volume_24h,
            "end_date": self.end_date,
            "category": self.category,
        }


def _as_list(v) -> list:
    """Gamma 的 outcomes/outcomePrices 可能是 JSON 字符串——归一成 list。"""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (ValueError, TypeError):
            return []
    return v if isinstance(v, list) else []


def events_to_odds(
    events_json: Iterable[dict], category: str, top_markets_per_event: int = 5
) -> list[EventOdd]:
    """Gamma /events 响应 -> EventOdd 列表（纯逻辑，可测）。

    每个 event 只取按 24h 成交量排序的前 N 个 market（World Cup 类事件有 60 个
    子市场，全要会淹没信号）；无 "Yes" 结果 / 价格解析失败的市场跳过。
    """
    out: list[EventOdd] = []
    for ev in events_json or []:
        markets = sorted(
            ev.get("markets") or [],
            key=lambda m: float(m.get("volume24hr") or 0),
            reverse=True,
        )[:top_markets_per_event]
        for m in markets:
            outcomes = [str(o) for o in _as_list(m.get("outcomes"))]
            prices = _as_list(m.get("outcomePrices"))
            if "Yes" not in outcomes or len(prices) != len(outcomes):
                continue
            try:
                yes_price = float(prices[outcomes.index("Yes")])
            except (ValueError, TypeError):
                continue
            if not 0.0 <= yes_price <= 1.0:
                continue
            out.append(
                EventOdd(
                    market_id=str(m.get("id", "")),
                    condition_id=str(m.get("conditionId", "")),
                    event_title=str(ev.get("title", "")).strip(),
                    question=str(m.get("question", "")).strip(),
                    yes_price=yes_price,
                    volume_24h=float(m.get("volume24hr") or 0),
                    end_date=m.get("endDate"),
                    category=category,
                )
            )
    return out


class EventsFetcher:
    """Gamma API 抓取（getter 可注入替身做零网络测试）。"""

    def __init__(self, getter: Callable = requests.get, timeout_sec: float = 20.0) -> None:
        self._get = getter
        self.timeout_sec = timeout_sec

    def fetch_tag(self, tag: str, limit: int = 20) -> list[EventOdd]:
        """按 tag 拉活跃事件（24h 量降序）。失败返回空列表并记日志（不炸调用方）。"""
        try:
            resp = self._get(
                f"{GAMMA_BASE}/events",
                params={
                    "limit": limit,
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                    "tag_slug": tag,
                },
                timeout=self.timeout_sec,
            )
            resp.raise_for_status()
            odds = events_to_odds(resp.json(), category=tag)
            logger.info(f"Fetched {len(odds)} market odds for tag={tag}")
            return odds
        except Exception as exc:  # noqa: BLE001 - 单 tag 失败不应中断整批
            logger.error(f"Polymarket fetch failed for tag={tag}: {exc}")
            return []

    def fetch_all(self, tags: Iterable[str], limit_per_tag: int = 20) -> list[EventOdd]:
        """多 tag 合并抓取，按 market_id 去重（同一市场可能挂多个 tag）。"""
        seen: dict[str, EventOdd] = {}
        for tag in tags:
            for odd in self.fetch_tag(tag, limit=limit_per_tag):
                seen.setdefault(odd.market_id, odd)
        return list(seen.values())
