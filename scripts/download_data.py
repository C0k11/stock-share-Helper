#!/usr/bin/env python
"""
下载历史数据脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from loguru import logger

from src.data.fetcher import DataFetcher
from src.data.storage import DataStorage


# 默认标的列表
DEFAULT_SYMBOLS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "SHY"]
MACRO_SYMBOLS = ["^VIX", "^TNX"]


def download_all(
    symbols: list = None,
    years: int = 10,
    include_macro: bool = True
):
    """
    下载所有数据
    
    Args:
        symbols: 标的列表
        years: 历史年数
        include_macro: 是否包含宏观指标
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    fetcher = DataFetcher()
    storage = DataStorage()
    
    start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    
    logger.info(f"Downloading data from {start_date} to today")
    logger.info(f"Symbols: {symbols}")
    
    # 下载ETF数据
    all_symbols = symbols.copy()
    if include_macro:
        all_symbols.extend(MACRO_SYMBOLS)
    
    price_data = fetcher.fetch_price(all_symbols, start_date)
    
    # 保存数据
    for symbol, df in price_data.items():
        storage.save_price_data(symbol.replace("^", ""), df)
        logger.info(f"Saved {symbol}: {len(df)} rows")
    
    # 打印摘要
    logger.info("=" * 50)
    logger.info("Download Summary:")
    for symbol in storage.list_symbols():
        info = storage.get_data_info(symbol)
        if info["exists"]:
            logger.info(f"  {symbol}: {info['rows']} rows, {info['start_date']} ~ {info['end_date']}")
    
    logger.info("Download completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载历史数据")
    parser.add_argument("--symbols", nargs="+", default=None, help="标的列表")
    parser.add_argument("--years", type=int, default=10, help="历史年数")
    parser.add_argument("--no-macro", action="store_true", help="不下载宏观指标")
    
    args = parser.parse_args()
    
    download_all(
        symbols=args.symbols,
        years=args.years,
        include_macro=not args.no_macro
    )
