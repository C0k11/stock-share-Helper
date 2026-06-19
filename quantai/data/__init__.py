"""quantai.data —— 数据层（行情抓取 / parquet 存储 / 美股交易日历）。

US-only：只支持美股（yfinance）；旧 akshare/CN 与 HK/CN 日历已移除。
新闻抓取不在本模块（将迁到 data/news.py）。

用法：
    from quantai.data import PriceFetcher, ParquetStorage, TradingCalendar
"""

from .calendar import TradingCalendar
from .prices import DEFAULT_US_ETFS, PriceFetcher, download_us_etfs
from .storage import ParquetStorage

__all__ = [
    "PriceFetcher",
    "download_us_etfs",
    "DEFAULT_US_ETFS",
    "ParquetStorage",
    "TradingCalendar",
]
