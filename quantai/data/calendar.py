"""美股(US / NYSE)交易日历，底层用 pandas_market_calendars。

US-only：旧版只硬编码了 2024 年的节假日、且把 HK/CN 留成空集合/抛
NotImplementedError 的半成品。

为什么用 pandas_market_calendars 而不是手搓 pandas 节假日规则：
NYSE 除了固定/浮动节假日，还有两类规则几乎不可能手搓全：
1. **一次性休市**：飓风 Sandy（2012-10-29/30）、国丧日（老布什 2018-12-05、
   福特 2007-01-02、里根 2004-06-11）、9·11 后停市（2001-09-11~14）。
2. **半日市（提前收盘 13:00 ET）**：感恩节翌日、独立日/圣诞前夕等。
本模块直接复用社区维护的官方日历，并保留一份「手搓规则」做交叉校验（见 tests）。

简化范围（如实声明）：只覆盖 pandas_market_calendars 的 NYSE 数据范围；
盘中分钟级撮合不在此模块职责内。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pandas_market_calendars as mcal

# 取/后交易日时向外探查的窗口：足够覆盖最长的连续非交易段（长周末+假日）。
_LOOKUP_WINDOW = timedelta(days=15)


def _as_date(value: date | datetime | str | pd.Timestamp) -> date:
    """把 date/datetime/Timestamp/'YYYY-MM-DD' 统一转成 datetime.date。"""
    if isinstance(value, datetime):  # 注意：datetime 是 date 的子类，必须先判
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(value).date()


class TradingCalendar:
    """美股交易日历：判断交易日、枚举区间交易日、取前/后交易日、识别半日市。"""

    MARKET = "NYSE"
    TIMEZONE = "America/New_York"

    def __init__(self) -> None:
        self._cal = mcal.get_calendar(self.MARKET)

    def _schedule(
        self,
        start: date | datetime | str | pd.Timestamp,
        end: date | datetime | str | pd.Timestamp,
    ) -> pd.DataFrame:
        """[start, end] 内交易日的开收盘时刻表（index=交易日，列含 market_open/close）。"""
        s, e = _as_date(start), _as_date(end)
        if e < s:
            return self._cal.schedule(start_date=s, end_date=s).iloc[0:0]
        return self._cal.schedule(start_date=s, end_date=e)

    def is_trading_day(self, dt: date | datetime | str | pd.Timestamp) -> bool:
        """该日是否为 NYSE 交易日（含半日市）。"""
        d = _as_date(dt)
        return not self._schedule(d, d).empty

    def get_trading_days(
        self,
        start_date: date | datetime | str | pd.Timestamp,
        end_date: date | datetime | str | pd.Timestamp,
    ) -> list[date]:
        """返回 [start, end] 闭区间内的所有交易日（升序）。"""
        return [ts.date() for ts in self._schedule(start_date, end_date).index]

    def previous_trading_day(self, dt: date | datetime | str | pd.Timestamp) -> date:
        """严格早于 dt 的最近一个交易日。"""
        d = _as_date(dt)
        sched = self._schedule(d - _LOOKUP_WINDOW, d - timedelta(days=1))
        if sched.empty:
            raise ValueError(f"在 {d} 前 {_LOOKUP_WINDOW.days} 天内找不到交易日")
        return sched.index[-1].date()

    def next_trading_day(self, dt: date | datetime | str | pd.Timestamp) -> date:
        """严格晚于 dt 的最近一个交易日（next_open 撮合会用到）。"""
        d = _as_date(dt)
        sched = self._schedule(d + timedelta(days=1), d + _LOOKUP_WINDOW)
        if sched.empty:
            raise ValueError(f"在 {d} 后 {_LOOKUP_WINDOW.days} 天内找不到交易日")
        return sched.index[0].date()

    def is_early_close(self, dt: date | datetime | str | pd.Timestamp) -> bool:
        """该日是否为半日市（提前收盘）。非交易日返回 False。"""
        d = _as_date(dt)
        sched = self._schedule(d, d)
        if sched.empty:
            return False
        return not self._cal.early_closes(sched).empty

    def early_closes(
        self,
        start_date: date | datetime | str | pd.Timestamp,
        end_date: date | datetime | str | pd.Timestamp,
    ) -> dict[date, str]:
        """返回 [start, end] 内所有半日市 -> 其收盘时刻(美东 HH:MM)。"""
        sched = self._schedule(start_date, end_date)
        if sched.empty:
            return {}
        early = self._cal.early_closes(sched)
        return {
            ts.date(): row["market_close"].tz_convert(self.TIMEZONE).strftime("%H:%M")
            for ts, row in early.iterrows()
        }

    def market_hours(self) -> dict[str, str]:
        """美东时间的常规交易时段（盘前/盘中/盘后）。"""
        return {
            "timezone": self.TIMEZONE,
            "pre_market_open": "04:00",
            "market_open": "09:30",
            "market_close": "16:00",
            "after_hours_close": "20:00",
        }
