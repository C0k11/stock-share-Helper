"""
交易日历模块 - 处理交易日、时区、节假日
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional
from loguru import logger


class TradingCalendar:
    """交易日历管理"""
    
    # 美股主要节假日（简化版，实际应从数据源获取）
    US_HOLIDAYS_2024 = [
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # MLK Day
        "2024-02-19",  # Presidents Day
        "2024-03-29",  # Good Friday
        "2024-05-27",  # Memorial Day
        "2024-06-19",  # Juneteenth
        "2024-07-04",  # Independence Day
        "2024-09-02",  # Labor Day
        "2024-11-28",  # Thanksgiving
        "2024-12-25",  # Christmas
    ]
    
    def __init__(self, market: str = "US"):
        """
        Args:
            market: 市场代码，US/HK/CN
        """
        self.market = market
        self._load_holidays()
        logger.info(f"TradingCalendar initialized for market: {market}")
    
    def _load_holidays(self):
        """加载节假日数据"""
        if self.market == "US":
            self.holidays = set(pd.to_datetime(self.US_HOLIDAYS_2024).date)
        else:
            # TODO: 加载港股/A股节假日
            self.holidays = set()
    
    def is_trading_day(self, dt: date) -> bool:
        """判断是否为交易日"""
        # 周末不交易
        if dt.weekday() >= 5:
            return False
        # 节假日不交易
        if dt in self.holidays:
            return False
        return True
    
    def get_trading_days(
        self,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """获取日期范围内的所有交易日"""
        days = []
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days
    
    def get_previous_trading_day(self, dt: date) -> date:
        """获取前一个交易日"""
        prev = dt - timedelta(days=1)
        while not self.is_trading_day(prev):
            prev -= timedelta(days=1)
        return prev
    
    def get_next_trading_day(self, dt: date) -> date:
        """获取下一个交易日"""
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def get_market_hours(self) -> dict:
        """获取市场交易时间"""
        if self.market == "US":
            return {
                "timezone": "America/New_York",
                "pre_market_open": "04:00",
                "market_open": "09:30",
                "market_close": "16:00",
                "after_hours_close": "20:00"
            }
        elif self.market == "HK":
            return {
                "timezone": "Asia/Hong_Kong",
                "morning_open": "09:30",
                "morning_close": "12:00",
                "afternoon_open": "13:00",
                "afternoon_close": "16:00"
            }
        else:
            raise NotImplementedError(f"Market {self.market} not supported")
