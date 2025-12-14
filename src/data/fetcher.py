"""
数据抓取模块 - 获取行情、宏观指标、新闻数据
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from loguru import logger


class DataFetcher:
    """统一数据抓取接口"""
    
    def __init__(self, source: str = "yfinance"):
        """
        Args:
            source: 数据源，支持 yfinance, stooq, polygon
        """
        self.source = source
        logger.info(f"DataFetcher initialized with source: {source}")
    
    def fetch_price(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        获取价格数据
        
        Args:
            symbols: 标的列表，如 ["SPY", "TLT", "GLD"]
            start_date: 开始日期，格式 "YYYY-MM-DD"
            end_date: 结束日期，默认今天
            interval: K线周期，1d/1h/5m等
        
        Returns:
            Dict[symbol, DataFrame]，每个DataFrame包含 open, high, low, close, volume, adj_close
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        result = {}
        
        if self.source == "yfinance":
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date, interval=interval)
                    
                    if df.empty:
                        logger.warning(f"No data for {symbol}")
                        continue
                    
                    # 标准化列名
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df.index.name = "date"
                    
                    result[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
        
        else:
            raise NotImplementedError(f"Source {self.source} not implemented")
        
        return result
    
    def fetch_vix(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取VIX波动率指数"""
        result = self.fetch_price(["^VIX"], start_date, end_date)
        return result.get("^VIX", pd.DataFrame())
    
    def fetch_treasury_yield(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取10年期国债收益率"""
        result = self.fetch_price(["^TNX"], start_date, end_date)
        return result.get("^TNX", pd.DataFrame())
    
    def fetch_news(
        self,
        keywords: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取新闻数据
        
        Args:
            keywords: 关键词过滤
            sources: 新闻源
            limit: 最大条数
        
        Returns:
            新闻列表，每条包含 title, content, source, published_at, url
        """
        import feedparser
        import requests
        from pathlib import Path
        import yaml

        def _match_keywords(title: str, content: str) -> bool:
            if not keywords:
                return True
            text = f"{title} {content}".lower()
            return any(str(k).lower() in text for k in keywords)

        news: List[Dict] = []

        def _load_config_sources() -> List[Dict]:
            try:
                cfg_path = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"
                if not cfg_path.exists():
                    return []
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                srcs = cfg.get("sources") or []
                out = []
                for s in srcs:
                    if not isinstance(s, dict):
                        continue
                    if not s.get("enabled", True):
                        continue
                    if (s.get("type") or "").lower() != "rss":
                        continue
                    url = (s.get("url") or "").strip()
                    if not url.lower().startswith("http"):
                        continue
                    out.append({
                        "id": s.get("id") or url,
                        "name": s.get("name") or "rss",
                        "url": url,
                        "category": s.get("category") or "market_news",
                        "weight": float(s.get("weight", 1.0) or 1.0),
                    })
                return out
            except Exception as e:
                logger.warning(f"Failed to load config sources.yaml: {e}")
                return []

        def _normalize_text(s: str) -> str:
            return " ".join((s or "").strip().split()).lower()

        def _dedup_key(item: Dict) -> str:
            url = (item.get("url") or "").strip()
            if url:
                return f"url::{url.lower()}"
            title = _normalize_text(item.get("title") or "")
            content = _normalize_text(item.get("content") or "")
            return f"tc::{title}::{content[:200]}"

        seen_keys = set()

        config_sources = []
        if not sources:
            config_sources = _load_config_sources()

        rss_sources: List[Dict] = []
        if sources:
            for src in sources:
                if not isinstance(src, str) or not src.strip():
                    continue
                if not src.lower().startswith("http"):
                    continue
                rss_sources.append({
                    "id": src,
                    "name": "rss",
                    "url": src,
                    "category": "market_news",
                    "weight": 1.0,
                })
        else:
            rss_sources = config_sources or [
                {
                    "id": "yahoo_spy",
                    "name": "Yahoo Finance - SPY",
                    "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
                    "category": "market_news",
                    "weight": 1.0,
                },
                {
                    "id": "yahoo_qqq",
                    "name": "Yahoo Finance - QQQ",
                    "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
                    "category": "market_news",
                    "weight": 1.0,
                },
                {
                    "id": "yahoo_tlt",
                    "name": "Yahoo Finance - TLT",
                    "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TLT&region=US&lang=en-US",
                    "category": "market_news",
                    "weight": 1.0,
                },
                {
                    "id": "yahoo_gld",
                    "name": "Yahoo Finance - GLD",
                    "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US",
                    "category": "market_news",
                    "weight": 1.0,
                },
            ]

        for src in rss_sources:
            src_url = (src.get("url") or "").strip()
            if not src_url.lower().startswith("http"):
                continue

            try:
                feed = feedparser.parse(src_url)
                for entry in getattr(feed, "entries", []) or []:
                    title = (getattr(entry, "title", None) or "").strip()
                    link = (getattr(entry, "link", None) or "").strip()
                    summary = (getattr(entry, "summary", None) or "").strip()
                    published = (getattr(entry, "published", None) or getattr(entry, "updated", None) or "")
                    source_name = src.get("name") or (getattr(feed, "feed", {}) or {}).get("title", "rss")

                    if not title and not summary:
                        continue
                    if not _match_keywords(title, summary):
                        continue

                    item = {
                        "title": title,
                        "content": summary,
                        "source": source_name,
                        "source_id": src.get("id") or src_url,
                        "category": src.get("category") or "market_news",
                        "weight": float(src.get("weight", 1.0) or 1.0),
                        "published_at": published,
                        "url": link,
                    }

                    key = _dedup_key(item)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    news.append(item)
                    if len(news) >= limit:
                        break
                if len(news) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to parse RSS source {src_url}: {e}")

        if news:
            return news[:limit]

        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return []

        try:
            q = " OR ".join(keywords) if keywords else "market OR stocks OR ETF"
            params = {
                "q": q,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": api_key,
            }
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json() or {}

            for a in data.get("articles", []) or []:
                title = (a.get("title") or "").strip()
                content = (a.get("content") or a.get("description") or "").strip()
                if not _match_keywords(title, content):
                    continue

                news.append({
                    "title": title,
                    "content": content,
                    "source": (a.get("source") or {}).get("name") or "newsapi",
                    "published_at": a.get("publishedAt") or "",
                    "url": a.get("url") or "",
                })
                if len(news) >= limit:
                    break

        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")

        return news[:limit]


# 便捷函数
def download_etf_data(
    symbols: List[str] = ["SPY", "QQQ", "TLT", "IEF", "GLD", "SHY"],
    years: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    下载ETF历史数据的便捷函数
    
    Args:
        symbols: ETF列表
        years: 历史年数
    
    Returns:
        价格数据字典
    """
    fetcher = DataFetcher()
    start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    return fetcher.fetch_price(symbols, start_date)
