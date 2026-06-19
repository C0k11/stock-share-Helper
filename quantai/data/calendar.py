"""美股(US / NYSE)交易日历。

US-only：旧版只硬编码了 2024 年的节假日、且把 HK/CN 留成空集合/抛
NotImplementedError 的半成品。这里用 pandas 的节假日规则引擎重写，**对任意年份**
都能算出 NYSE 全日休市日，并移除 HK/CN 分支。

简化范围（如实声明）：
- 只处理「全日休市」，不含感恩节翌日/圣诞前夕等**盘中提前收盘**(半日)。
- 采用现代观察规则；Juneteenth 自 2021 年起才计为休市日。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)


def _as_date(value: date | datetime | str | pd.Timestamp) -> date:
    """把 date/datetime/Timestamp/'YYYY-MM-DD' 统一转成 datetime.date。"""
    if isinstance(value, datetime):  # 注意：datetime 是 date 的子类，必须先判
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(value).date()


class USMarketHolidayCalendar(AbstractHolidayCalendar):
    """NYSE/Nasdaq 全日休市日（现代规则）。

    观察规则差异（NYSE 真实规则）：
    - 元旦：周日 -> 周一；周六**不**前挪到周五（用 sunday_to_monday）。
    - 独立日/Juneteenth/圣诞：周六 -> 前一个周五，周日 -> 周一（用 nearest_workday）。
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date=pd.Timestamp("2021-06-19"),
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


class TradingCalendar:
    """美股交易日历：判断交易日、枚举区间交易日、取前/后交易日。"""

    MARKET = "US"
    TIMEZONE = "America/New_York"

    def __init__(self) -> None:
        self._calendar = USMarketHolidayCalendar()
        self._holidays_by_year: dict[int, set[date]] = {}

    def _holidays_for_year(self, year: int) -> set[date]:
        cached = self._holidays_by_year.get(year)
        if cached is None:
            raw = self._calendar.holidays(
                start=pd.Timestamp(year, 1, 1), end=pd.Timestamp(year, 12, 31)
            )
            cached = {ts.date() for ts in raw}
            self._holidays_by_year[year] = cached
        return cached

    def is_trading_day(self, dt: date | datetime | str | pd.Timestamp) -> bool:
        """周一至周五且非 NYSE 休市日 -> True。"""
        d = _as_date(dt)
        if d.weekday() >= 5:  # 5=周六, 6=周日
            return False
        return d not in self._holidays_for_year(d.year)

    def get_trading_days(
        self,
        start_date: date | datetime | str | pd.Timestamp,
        end_date: date | datetime | str | pd.Timestamp,
    ) -> list[date]:
        """返回 [start, end] 闭区间内的所有交易日（升序）。"""
        start, end = _as_date(start_date), _as_date(end_date)
        days: list[date] = []
        cursor = start
        step = timedelta(days=1)
        while cursor <= end:
            if self.is_trading_day(cursor):
                days.append(cursor)
            cursor += step
        return days

    def previous_trading_day(self, dt: date | datetime | str | pd.Timestamp) -> date:
        """严格早于 dt 的最近一个交易日。"""
        cursor = _as_date(dt) - timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor -= timedelta(days=1)
        return cursor

    def next_trading_day(self, dt: date | datetime | str | pd.Timestamp) -> date:
        """严格晚于 dt 的最近一个交易日（next_open 撮合会用到）。"""
        cursor = _as_date(dt) + timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor += timedelta(days=1)
        return cursor

    def market_hours(self) -> dict[str, str]:
        """美东时间的常规交易时段（盘前/盘中/盘后）。"""
        return {
            "timezone": self.TIMEZONE,
            "pre_market_open": "04:00",
            "market_open": "09:30",
            "market_close": "16:00",
            "after_hours_close": "20:00",
        }
