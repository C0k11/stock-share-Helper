"""quantai.data.calendar 的交易日/节假日规则测试（无网络）。"""

from __future__ import annotations

from datetime import date

from quantai.data.calendar import TradingCalendar


def _cal() -> TradingCalendar:
    return TradingCalendar()


def test_fixed_and_floating_holidays_2024() -> None:
    cal = _cal()
    assert not cal.is_trading_day(date(2024, 1, 1))   # New Year (Mon)
    assert not cal.is_trading_day(date(2024, 1, 15))  # MLK (3rd Mon)
    assert not cal.is_trading_day(date(2024, 2, 19))  # Presidents
    assert not cal.is_trading_day(date(2024, 3, 29))  # Good Friday
    assert not cal.is_trading_day(date(2024, 5, 27))  # Memorial
    assert not cal.is_trading_day(date(2024, 6, 19))  # Juneteenth
    assert not cal.is_trading_day(date(2024, 7, 4))   # Independence
    assert not cal.is_trading_day(date(2024, 9, 2))   # Labor
    assert not cal.is_trading_day(date(2024, 11, 28)) # Thanksgiving
    assert not cal.is_trading_day(date(2024, 12, 25)) # Christmas


def test_weekends_are_not_trading_days() -> None:
    cal = _cal()
    assert not cal.is_trading_day(date(2024, 1, 6))  # Saturday
    assert not cal.is_trading_day(date(2024, 1, 7))  # Sunday


def test_regular_weekday_is_trading() -> None:
    cal = _cal()
    assert cal.is_trading_day(date(2024, 1, 2))  # Tuesday


def test_independence_day_saturday_observed_on_friday() -> None:
    """2020-07-04 是周六 -> NYSE 前挪到周五 07-03 休市（nearest_workday）。"""
    cal = _cal()
    assert not cal.is_trading_day(date(2020, 7, 3))


def test_new_year_saturday_does_not_shift_to_friday() -> None:
    """2022-01-01 是周六 -> 元旦用 sunday_to_monday，周五 2021-12-31 仍开市。"""
    cal = _cal()
    assert cal.is_trading_day(date(2021, 12, 31))


def test_juneteenth_only_from_2021() -> None:
    cal = _cal()
    assert cal.is_trading_day(date(2019, 6, 19))       # 2021 前不是休市日
    assert not cal.is_trading_day(date(2022, 6, 20))   # 2022-06-19 周日 -> 周一休市


def test_get_trading_days_excludes_weekend_and_holiday() -> None:
    cal = _cal()
    days = cal.get_trading_days(date(2024, 1, 1), date(2024, 1, 7))
    # Jan1 holiday, Jan6 Sat, Jan7 Sun -> 剩 2/3/4/5
    assert days == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]


def test_previous_and_next_trading_day_skip_holiday_weekend() -> None:
    cal = _cal()
    # 2024-01-01 是周一假日；其前一交易日应是 2023-12-29(周五)
    assert cal.previous_trading_day(date(2024, 1, 1)) == date(2023, 12, 29)
    # 其后一交易日应是 2024-01-02(周二)
    assert cal.next_trading_day(date(2024, 1, 1)) == date(2024, 1, 2)


def test_accepts_string_and_datetime_inputs() -> None:
    cal = _cal()
    assert cal.is_trading_day("2024-01-02") is True
    assert cal.is_trading_day("2024-01-01") is False


def test_market_hours_is_us() -> None:
    hours = _cal().market_hours()
    assert hours["timezone"] == "America/New_York"
    assert hours["market_open"] == "09:30"


# --- pandas_market_calendars 才覆盖得到的「一次性休市 / 半日市」--- #

def test_one_off_full_day_closure_bush_mourning_2018() -> None:
    """2018-12-05 老布什国丧日：手搓规则会漏，库能识别为休市。"""
    assert not _cal().is_trading_day(date(2018, 12, 5))


def test_hurricane_sandy_closures_2012() -> None:
    """飓风 Sandy：2012-10-29/30 全日休市。"""
    cal = _cal()
    assert not cal.is_trading_day(date(2012, 10, 29))
    assert not cal.is_trading_day(date(2012, 10, 30))


def test_thanksgiving_friday_is_early_close_but_trading() -> None:
    """感恩节翌日(2023-11-24)是半日市：仍是交易日，但提前收盘。"""
    cal = _cal()
    friday = date(2023, 11, 24)
    assert cal.is_trading_day(friday) is True
    assert cal.is_early_close(friday) is True


def test_regular_day_is_not_early_close() -> None:
    assert _cal().is_early_close(date(2024, 1, 2)) is False


def test_early_closes_maps_to_one_pm_et() -> None:
    """半日市收盘时刻为美东 13:00。"""
    early = _cal().early_closes(date(2023, 1, 1), date(2023, 12, 31))
    assert date(2023, 11, 24) in early
    assert early[date(2023, 11, 24)] == "13:00"
