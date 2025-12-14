"""
数据抓取模块 - 获取行情、宏观指标、新闻数据
"""

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
        # TODO: 实现新闻抓取（RSS/NewsAPI）
        logger.warning("News fetching not implemented yet")
        return []


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
