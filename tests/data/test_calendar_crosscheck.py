"""交叉校验：生产用的 pandas_market_calendars vs 手搓的 pandas 节假日规则。

目的（对应需求「拿库的输出对拍手搓那版 >= 5 年，确认零差异」）：
在 14 年区间(2010-2023)上对拍两套日历，证明二者的差异**恰好等于**已知的
「一次性全日休市」(飓风 Sandy、国丧日)——即换用库**只多了正确的休市**，
没有引入任何无法解释的回归。这把「手搓易漏」这件事钉成了可执行的断言。
"""

from __future__ import annotations

from datetime import date

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

from quantai.data.calendar import TradingCalendar


class _RuleBasedNYSE(AbstractHolidayCalendar):
    """手搓 NYSE 全日休市规则（仅固定/浮动节假日；不含一次性休市/半日市）。"""

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
            start_date=pd.Timestamp("2022-06-19"),  # NYSE 自 2022 起才休 Juneteenth
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


START, END = date(2010, 1, 1), date(2023, 12, 31)  # 14 年 >> 5 年

#: 区间内「库有、手搓规则没有」的一次性全日休市（人工确认）：
KNOWN_SPECIAL_CLOSURES = {
    date(2012, 10, 29),  # 飓风 Sandy
    date(2012, 10, 30),  # 飓风 Sandy
    date(2018, 12, 5),   # 老布什国丧日
}


def _rule_based_trading_days(start: date, end: date) -> set[date]:
    weekdays = {ts.date() for ts in pd.bdate_range(start, end)}
    holidays = {
        ts.date()
        for ts in _RuleBasedNYSE().holidays(start=pd.Timestamp(start), end=pd.Timestamp(end))
    }
    return weekdays - holidays


def test_pmc_is_superset_of_rule_holidays() -> None:
    """库不会在『手搓规则认为是节假日』的日子开市（pmc 交易日 ⊆ 规则交易日）。"""
    pmc_days = set(TradingCalendar().get_trading_days(START, END))
    rule_days = _rule_based_trading_days(START, END)
    assert pmc_days - rule_days == set()


def test_diff_is_exactly_known_special_closures() -> None:
    """规则比库『多开市』的日子，恰好等于已知的一次性休市——零意外偏差。"""
    pmc_days = set(TradingCalendar().get_trading_days(START, END))
    rule_days = _rule_based_trading_days(START, END)
    assert rule_days - pmc_days == KNOWN_SPECIAL_CLOSURES
