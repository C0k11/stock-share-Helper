"""美股行情抓取（yfinance）。

从旧 `src/data/fetcher.py` 的 `DataFetcher` 迁移，做了两处收敛（US-only）：
1. 移除 `akshare` 源与「CN 6 位代码 -> .SS/.SZ」后缀映射（CN 专属，且需未声明的依赖）。
2. 移除 `fetch_news`（混在数据抓取里违反单一职责）—— 新闻抓取迁到后续 `data/news.py`。

留下的就是干净的「美股价格抓取」：价格、VIX、10Y 国债收益率。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

import pandas as pd
import yfinance as yf
from loguru import logger

#: 默认美股 ETF 池（与旧 download_etf_data 一致）
DEFAULT_US_ETFS: tuple[str, ...] = ("SPY", "QQQ", "TLT", "IEF", "GLD", "SHY")


class PriceFetcher:
    """统一的美股价格抓取接口（仅 yfinance）。"""

    def __init__(self, source: str = "yfinance") -> None:
        if source != "yfinance":
            raise ValueError(
                f"US-only 版本仅支持 source='yfinance'，收到 {source!r}（旧 akshare/CN 路径已移除）"
            )
        self.source = source
        logger.info(f"PriceFetcher initialized with source: {source}")

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """把 yfinance 原始列名规范成小写下划线，并把索引命名为 'date'。"""
        out = df.copy()
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
        out.index.name = "date"
        return out

    def fetch_prices(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """抓取一组标的的 OHLCV 数据。

        Args:
            symbols: 标的列表，如 ["SPY", "TLT"]。
            start_date: 起始日期 'YYYY-MM-DD'。
            end_date: 结束日期，默认到明天（含今天）。
            interval: K 线周期，默认 '1d'。

        Returns:
            {symbol: DataFrame}；抓取失败/空数据的标的会被跳过（记录日志）。
        """
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            sym = str(symbol).strip()
            try:
                raw = yf.Ticker(sym).history(start=start_date, end=end_date, interval=interval)
                if raw.empty:
                    logger.warning(f"No data for {sym}")
                    continue
                result[sym] = self._standardize(raw)
                logger.info(f"Fetched {len(raw)} bars for {sym}")
            except Exception as exc:  # noqa: BLE001 - 单标的失败不应中断整批
                logger.error(f"Error fetching {sym}: {exc}")
        return result

    def fetch_vix(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """VIX 波动率指数（用于 regime 检测）。"""
        return self.fetch_prices(["^VIX"], start_date, end_date).get("^VIX", pd.DataFrame())

    def fetch_treasury_yield(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """10 年期美债收益率（^TNX）。"""
        return self.fetch_prices(["^TNX"], start_date, end_date).get("^TNX", pd.DataFrame())


def download_us_etfs(
    symbols: Optional[Sequence[str]] = None,
    years: int = 10,
) -> dict[str, pd.DataFrame]:
    """便捷函数：下载默认美股 ETF 池最近 `years` 年的日线。"""
    syms = list(symbols) if symbols is not None else list(DEFAULT_US_ETFS)
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    return PriceFetcher().fetch_prices(syms, start_date)
