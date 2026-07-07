"""tactician 单测：五因子+盘中修正计分、动作映射、止损参考位、警报、简报组装。"""

from __future__ import annotations

import pytest

from quantai.agents.tactician import (
    ACTION_ADD,
    ACTION_ENTRY,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_TRIM,
    ACTION_WATCH,
    advice_row,
    advise,
    alerts,
    build_tactical_brief,
)


def _daily(**over) -> dict:
    # retrace_from_20d_high 生产口径恒为正（越大回撤越深）——夹具必须用真实符号
    base = {
        "ma_alignment": 1.0, "in_uptrend": True, "macd_hist": 0.5, "ret_20d_pct": 5.0,
        "rsi_14": 55.0, "retrace_from_20d_high": 0.02, "sma_20": 100.0,
        "last_close": 105.0, "is_pullback": False, "realized_vol_20_ann": 0.25,
    }
    base.update(over)
    return base


def _intraday(**over) -> dict:
    base = {
        "last": 104.0, "chg_pct": -0.01, "day_low": 102.0, "day_high": 106.0,
        "close_position": 0.5, "vs_vwap_pct": 0.002, "down_volume_share": 0.45,
        "vol_vs_avg": 0.8,
    }
    base.update(over)
    return base


class TestScoringAndActions:
    def test_strong_bull_held_says_add(self):
        a = advise("AAA", _daily(), held=True)
        assert a["action"] == ACTION_ADD and a["score"] >= 3

    def test_strong_bear_held_says_exit(self):
        a = advise(
            "BBB",
            _daily(ma_alignment=0.0, in_uptrend=False, macd_hist=-1.0, ret_20d_pct=-10.0,
                   retrace_from_20d_high=0.2),
            held=True,
        )
        assert a["action"] == ACTION_EXIT

    def test_deep_retrace_actually_deducts(self):
        """生产符号（正值）下深回撤必须真触发——旧版 `<=-0.15` 是永假死代码。"""
        base = advise("R", _daily(), held=True)
        deep = advise("R", _daily(retrace_from_20d_high=0.22), held=True)
        assert deep["score"] == base["score"] - 1
        assert any("20 日高点" in r for r in deep["risks"])

    def test_high_volatility_deducts(self):
        """第五因子（波动）：>=60% 年化必须减分（与 rule_v1 同门）。"""
        base = advise("V", _daily(), held=True)
        hot = advise("V", _daily(realized_vol_20_ann=0.8), held=True)
        assert hot["score"] == base["score"] - 1
        assert any("波动率" in r for r in hot["risks"])

    def test_rsi_70_threshold_aligned_with_rule_v1(self):
        calm = advise("S", _daily(rsi_14=69.0), held=True)
        hot = advise("S", _daily(rsi_14=72.0), held=True)
        assert hot["score"] == calm["score"] - 1

    def test_mid_scores_map_hold_and_trim(self):
        hold = advise("C", _daily(macd_hist=-0.1, ret_20d_pct=-1.0), held=True)  # +1+1-1-1=0
        assert hold["action"] == ACTION_HOLD
        trim = advise("D", _daily(ma_alignment=0.2, macd_hist=-0.1, ret_20d_pct=-1.0), held=True)
        assert trim["action"] == ACTION_TRIM

    def test_not_held_entry_vs_watch(self):
        assert advise("E", _daily(), held=False)["action"] == ACTION_ENTRY
        assert advise("F", _daily(ma_alignment=0.5, macd_hist=-0.1), held=False)["action"] == ACTION_WATCH

    def test_pullback_lowers_entry_bar(self):
        vals = _daily(macd_hist=-0.1, is_pullback=True)  # score=2
        assert advise("G", vals, held=False)["action"] == ACTION_ENTRY


class TestIntradayOverlay:
    def test_sell_pressure_downgrades(self):
        base = advise("H", _daily(), held=True)
        hit = advise("H", _daily(), _intraday(down_volume_share=0.7, vs_vwap_pct=-0.02), held=True)
        assert hit["score"] == base["score"] - 1
        assert any("抛压" in r for r in hit["risks"])

    def test_volume_crash_downgrades(self):
        hit = advise("I", _daily(), _intraday(vol_vs_avg=2.0, chg_pct=-0.05), held=True)
        assert any("放量急跌" in r for r in hit["risks"])

    def test_intraday_never_upgrades(self):
        base = advise("J", _daily(), held=True)
        calm = advise("J", _daily(), _intraday(down_volume_share=0.2, vs_vwap_pct=0.02), held=True)
        assert calm["score"] <= base["score"]


class TestStops:
    def test_stop_is_ma20_when_above(self):
        a = advise("K", _daily(sma_20=100.0), _intraday(last=105.0), held=True)
        assert a["stop"] == pytest.approx(100.0)

    def test_stop_is_day_low_when_below_ma20(self):
        a = advise("L", _daily(sma_20=100.0), _intraday(last=98.0, day_low=96.5), held=True)
        assert a["stop"] == pytest.approx(96.5)

    def test_no_stop_for_not_held(self):
        assert advise("M", _daily(), _intraday(), held=False)["stop"] is None


class TestPresentation:
    def test_alerts_prioritize_held_trim(self):
        advs = [
            advise("BBB", _daily(ma_alignment=0.0, in_uptrend=False, macd_hist=-1.0,
                                 ret_20d_pct=-10.0), held=True),
            advise("E", _daily(), held=False),
        ]
        out = alerts(advs)
        assert any("BBB" in x and "⚠" in x for x in out)
        assert any("E" in x and "🟢" in x for x in out)

    def test_row_and_brief_contain_numbers(self):
        a = advise("N", _daily(), _intraday(), held=True)
        row = advice_row(a)
        assert row["动作"] == a["action"] and "📌" in row["标的"]
        brief = build_tactical_brief([a], as_of="12:00")
        assert "N" in brief and "操作卡" in brief and "12:00" in brief

    def test_brief_handles_missing_price(self):
        a = advise("O", {"sma_20": float("nan")}, None, held=False)
        assert "现价 n/a" in build_tactical_brief([a])
