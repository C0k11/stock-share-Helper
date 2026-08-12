"""新闻通道：按标的抓 RSS 头条（Yahoo Finance per-ticker feed + config 自定义源）。

设计：
- **纯解析可测**：`entries_to_items`（feedparser entries -> `NewsItem`）不碰网络，
  单测注入假 entries；网络只在 `NewsFetcher.fetch_*`（`parser` 可注入替身）。
- **诚实字段**：发布时间解析不出来就是 None（不用抓取时间冒充发布时间）；
  空标题的条目直接丢弃。
- 下游：仓库 `raw.news`（etl.load_news，按 link 去重幂等）-> dbt `fact_news`；
  streamlit 个股页展示；后续可接 agents 的 news expert / distill 场景。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

import feedparser
import requests
from loguru import logger

#: Yahoo Finance 按标的 RSS（免费、无需 key；美股口径）
YAHOO_TICKER_FEED = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
)

#: RSS 抓取超时（秒）。feedparser 自带的 urllib 路径**没有超时**（实测其安装包全文
#: 无 timeout），一个卡死的源会把无人值守的定时任务挂住数小时——改走 requests。
FEED_TIMEOUT_SEC = 20.0


def _parse_feed(url: str):
    """默认 parser：requests 限时拉取 -> feedparser 解析字节（不让 feedparser 碰网络）。"""
    resp = requests.get(url, timeout=FEED_TIMEOUT_SEC, headers={"User-Agent": "quantai-rss/1.0"})
    resp.raise_for_status()
    return feedparser.parse(resp.content)


@dataclass
class NewsItem:
    """一条新闻头条。symbol 为 None 表示来自非标的绑定的通用源。"""

    symbol: Optional[str]
    title: str
    summary: str
    link: str
    published: Optional[datetime]  # UTC；解析不出为 None（诚实缺失）
    source: str

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "title": self.title,
            "summary": self.summary,
            "link": self.link,
            "published": self.published,
            "source": self.source,
        }


def _entry_published(entry) -> Optional[datetime]:
    """feedparser entry -> UTC datetime；缺失/解析失败 -> None。"""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def entries_to_items(
    entries: Iterable, symbol: Optional[str], source: str
) -> list[NewsItem]:
    """feedparser entries -> NewsItem 列表（纯逻辑，可测）。

    丢弃：空标题、空链接（无法去重的条目不入库）。summary 缺失落空串。
    """
    items: list[NewsItem] = []
    for e in entries:
        title = str(e.get("title", "") or "").strip()
        link = str(e.get("link", "") or "").strip()
        if not title or not link:
            continue
        items.append(
            NewsItem(
                symbol=symbol.upper() if symbol else None,
                title=title,
                summary=str(e.get("summary", "") or "").strip(),
                link=link,
                published=_entry_published(e),
                source=source,
            )
        )
    return items


class NewsFetcher:
    """RSS 新闻抓取（Yahoo per-ticker + 任意 RSS 源）。

    Args:
        extra_feeds: config 的 `data.news_feeds`（通用 RSS 源，非标的绑定）。
        parser: 测试接缝，默认 `feedparser.parse`（替身注入即可零网络测试）。
    """

    def __init__(
        self,
        extra_feeds: Optional[list[str]] = None,
        parser: Callable = _parse_feed,
    ) -> None:
        self.extra_feeds = list(extra_feeds or [])
        self._parse = parser

    def fetch_symbol_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """单标的头条（Yahoo Finance RSS）。抓取失败返回空列表并记日志（不炸调用方）。"""
        sym = str(symbol).strip().upper()
        url = YAHOO_TICKER_FEED.format(symbol=sym)
        try:
            feed = self._parse(url)
            items = entries_to_items(feed.entries, symbol=sym, source="yahoo")
            logger.info(f"Fetched {len(items)} news items for {sym}")
            return items[:limit]
        except Exception as exc:  # noqa: BLE001 - 单源失败不应中断整批
            logger.error(f"News fetch failed for {sym}: {exc}")
            return []

    def fetch_all(self, symbols: Iterable[str], limit_per_symbol: int = 20) -> list[NewsItem]:
        """一组标的 + config 通用源的全部头条。"""
        items: list[NewsItem] = []
        for sym in symbols:
            items.extend(self.fetch_symbol_news(sym, limit=limit_per_symbol))
        for url in self.extra_feeds:
            try:
                feed = self._parse(url)
                items.extend(entries_to_items(feed.entries, symbol=None, source=url))
            except Exception as exc:  # noqa: BLE001
                logger.error(f"News feed failed {url}: {exc}")
        return items
