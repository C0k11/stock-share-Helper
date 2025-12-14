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

        def _match_keywords(title: str, content: str) -> bool:
            if not keywords:
                return True
            text = f"{title} {content}".lower()
            return any(str(k).lower() in text for k in keywords)

        news: List[Dict] = []

        rss_sources = sources
        if not rss_sources:
            rss_sources = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TLT&region=US&lang=en-US",
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GLD&region=US&lang=en-US",
            ]

        for src in rss_sources:
            if not isinstance(src, str) or not src.strip():
                continue
            if not src.lower().startswith("http"):
                continue

            try:
                feed = feedparser.parse(src)
                for entry in getattr(feed, "entries", []) or []:
                    title = (getattr(entry, "title", None) or "").strip()
                    link = (getattr(entry, "link", None) or "").strip()
                    summary = (getattr(entry, "summary", None) or "").strip()
                    published = (getattr(entry, "published", None) or getattr(entry, "updated", None) or "")
                    source_name = (getattr(feed, "feed", {}) or {}).get("title", "rss")

                    if not title and not summary:
                        continue
                    if not _match_keywords(title, summary):
                        continue

                    news.append({
                        "title": title,
                        "content": summary,
                        "source": source_name,
                        "published_at": published,
                        "url": link,
                    })
                    if len(news) >= limit:
                        break
                if len(news) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to parse RSS source {src}: {e}")

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
