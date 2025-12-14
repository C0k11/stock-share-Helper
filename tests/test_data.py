"""
数据模块测试
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta


class TestDataFetcher:
    """测试数据抓取"""
    
    def test_fetch_price_single_symbol(self):
        """测试单标的数据抓取"""
        from src.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        result = fetcher.fetch_price(["SPY"], start_date)
        
        assert "SPY" in result
        assert isinstance(result["SPY"], pd.DataFrame)
        assert len(result["SPY"]) > 0
        assert "close" in result["SPY"].columns
    
    def test_fetch_price_multiple_symbols(self):
        """测试多标的数据抓取"""
        from src.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        symbols = ["SPY", "TLT", "GLD"]
        
        result = fetcher.fetch_price(symbols, start_date)
        
        for symbol in symbols:
            assert symbol in result


class TestTradingCalendar:
    """测试交易日历"""
    
    def test_is_trading_day(self):
        """测试交易日判断"""
        from src.data.calendar import TradingCalendar
        from datetime import date
        
        cal = TradingCalendar(market="US")
        
        # 周末不是交易日
        saturday = date(2024, 1, 6)  # 周六
        assert not cal.is_trading_day(saturday)
        
        # 周一是交易日（如果不是节假日）
        monday = date(2024, 1, 8)
        assert cal.is_trading_day(monday)
    
    def test_get_market_hours(self):
        """测试交易时间"""
        from src.data.calendar import TradingCalendar
        
        cal = TradingCalendar(market="US")
        hours = cal.get_market_hours()
        
        assert hours["market_open"] == "09:30"
        assert hours["market_close"] == "16:00"


class TestDataStorage:
    """测试数据存储"""
    
    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        from src.data.storage import DataStorage
        
        storage = DataStorage(base_path=str(tmp_path))
        
        # 创建测试数据
        df = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        }, index=pd.date_range("2024-01-01", periods=3))
        
        # 保存
        storage.save_price_data("TEST", df)
        
        # 加载
        loaded = storage.load_price_data("TEST")
        
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded["close"]) == [100, 101, 102]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
