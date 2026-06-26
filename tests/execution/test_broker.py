"""quantai.execution.broker.PaperBroker 核心撮合测试。

覆盖：开多/加多/均价、减多平仓、开空/加空、空头回补+反手、盘前风控（杠杆/保证金）、
非法/不支持单、FILL/LOG 注入接缝、recorder PnL 回填、计费（利息/借券费）、维持保证金强平、
on_market_data 更新最新价。
"""

from __future__ import annotations

from quantai.execution.broker import PaperBroker, Position


class FakeRecorder:
    def __init__(self):
        self.calls = []

    def log_outcome(self, *, ref_id, outcome, comment):
        self.calls.append({"ref_id": ref_id, "outcome": outcome, "comment": comment})


def _broker(cash=100000.0, **kw):
    fills, logs = [], []
    rec = FakeRecorder()
    b = PaperBroker(
        cash=cash,
        on_fill=fills.append,
        on_log=lambda m, p: logs.append((m, p)),
        recorder=rec,
        **kw,
    )
    return b, fills, logs, rec


# --------------------------------------------------------------------- #
# 开仓 / 加仓 / 均价
# --------------------------------------------------------------------- #
def test_buy_opens_long():
    b, fills, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 100.0, "shares": 100})
    assert b.cash == 100000.0 - 100 * 100
    pos = b.positions["NVDA"]
    assert pos.shares == 100 and pos.avg_price == 100.0
    assert fills and fills[-1]["action"] == "BUY" and fills[-1]["shares"] == 100


def test_buy_adds_to_long_averages_price():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 100.0, "shares": 100})
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 200.0, "shares": 100})
    pos = b.positions["NVDA"]
    assert pos.shares == 200
    assert pos.avg_price == (100 * 100 + 200 * 100) / 200  # 150


def test_sell_reduces_long_and_records_pnl():
    b, fills, _, rec = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 100.0, "shares": 100, "trace_id": "t1"})
    cash_after_buy = b.cash
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 110.0, "shares": 100})
    assert "NVDA" not in b.positions  # 全平
    assert b.cash == cash_after_buy + 110 * 100
    assert rec.calls and rec.calls[-1]["ref_id"] == "t1"
    assert abs(rec.calls[-1]["outcome"] - 1000.0) < 1e-6  # (110-100)*100


def test_partial_sell_keeps_remainder():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 100.0, "shares": 100})
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 120.0, "shares": 40})
    assert b.positions["NVDA"].shares == 60


# --------------------------------------------------------------------- #
# 做空 / 回补 / 反手
# --------------------------------------------------------------------- #
def test_sell_opens_short():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 100.0, "shares": 50})
    pos = b.positions["NVDA"]
    assert pos.shares == -50
    assert b.cash == 100000.0 + 100 * 50


def test_add_to_short_averages():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 100.0, "shares": 50})
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 120.0, "shares": 50})
    pos = b.positions["NVDA"]
    assert pos.shares == -100
    assert pos.avg_price == (100 * 50 + 120 * 50) / 100  # 110


def test_buy_covers_short_and_flips_to_long():
    b, _, _, rec = _broker()
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 100.0, "shares": 50, "trace_id": "s1"})
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 90.0, "shares": 80})
    pos = b.positions["NVDA"]
    assert pos.shares == 30  # 回补 50 + 反手 30 多
    assert pos.avg_price == 90.0
    # 回补盈利 (entry 100 - cover 90) * 50 = 500
    assert rec.calls and abs(rec.calls[-1]["outcome"] - 500.0) < 1e-6


# --------------------------------------------------------------------- #
# 盘前风控
# --------------------------------------------------------------------- #
def test_leverage_exceeded_rejected():
    b, fills, logs, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 500.0, "shares": 1000})  # lev 5 > 3
    assert b.positions == {}
    assert fills == []
    assert any("leverage_exceeded" in m for m, _ in logs)


def test_initial_margin_exceeded_rejected():
    b, _, logs, _ = _broker()
    b.initial_margin = 0.6
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 500.0, "shares": 400})  # lev 2, eq 0.5*gross < 0.6
    assert b.positions == {}
    assert any("margin_exceeded" in m for m, _ in logs)


def test_within_limits_buy_passes():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 500.0, "shares": 100})  # notional 50k, lev 0.5
    assert b.positions["NVDA"].shares == 100


# --------------------------------------------------------------------- #
# 非法单 / 接缝
# --------------------------------------------------------------------- #
def test_invalid_order_ignored():
    b, fills, logs, _ = _broker()
    b.place_order({"ticker": "", "action": "BUY", "price": 100.0, "shares": 10})
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 0.0, "shares": 10})
    assert fills == [] and b.positions == {}
    assert any("invalid order" in m for m, _ in logs)


def test_unsupported_action():
    b, fills, logs, _ = _broker()
    # HOLD 通不过 pretrade（unsupported_action）
    b.place_order({"ticker": "NVDA", "action": "HOLD", "price": 100.0, "shares": 10})
    assert fills == [] and b.positions == {}


def test_default_sinks_are_noop():
    b = PaperBroker(cash=1000.0)  # 不注入任何回调
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 10.0, "shares": 5})  # 不应抛
    assert b.positions["NVDA"].shares == 5


def test_recorder_none_safe():
    fills = []
    b = PaperBroker(cash=100000.0, on_fill=fills.append, recorder=None)
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 100.0, "shares": 10, "trace_id": "x"})
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 110.0, "shares": 10})  # 不应抛
    assert "NVDA" not in b.positions


# --------------------------------------------------------------------- #
# 计费 / 盯市 / 强平
# --------------------------------------------------------------------- #
def test_margin_interest_accrues_on_negative_cash():
    b, _, _, _ = _broker()
    b.cash = -10000.0
    b._last_settle_ts = 0.0
    b.mark_to_market(asof_ts=3600.0)  # 1h
    assert b.cash < -10000.0  # 利息让负现金更负


def test_short_borrow_fee_accrues():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "SELL", "price": 100.0, "shares": 100})  # 开空
    cash_after = b.cash
    b._last_settle_ts = 0.0
    b.mark_to_market(asof_ts=3600.0)
    assert b.cash < cash_after  # 借券费扣减


def test_maintenance_margin_liquidation():
    b, fills, _, _ = _broker()
    # 杠杆买入：500 股 @ 500 -> notional 250k, cash -150k, lev 2.5 (<3 通过)
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 500.0, "shares": 500})
    assert b.positions["NVDA"].shares == 500
    fills.clear()
    # 价格暴跌到 380 -> eq 40k < maintenance req 47.5k -> 触发强平
    b.last_price["NVDA"] = 380.0
    b._maintenance_check_and_liquidate(asof_ts=1.0)
    assert b.positions.get("NVDA") is None or b.positions["NVDA"].shares < 500
    assert any(f["expert"] == "liquidation" for f in fills)


def test_equity_helper():
    b, _, _, _ = _broker()
    b.place_order({"ticker": "NVDA", "action": "BUY", "price": 100.0, "shares": 100})
    assert abs(b.equity() - 100000.0) < 1e-6  # 现金 90k + 持仓 10k


# --------------------------------------------------------------------- #
# on_market_data
# --------------------------------------------------------------------- #
def test_on_market_data_updates_last_price():
    b, _, _, _ = _broker()
    b.on_market_data({"ticker": "nvda", "close": 123.0})
    assert b.last_price["NVDA"] == 123.0


def test_on_market_data_alt_price_keys():
    b, _, _, _ = _broker()
    b.on_market_data({"symbol": "TSLA", "price": 250.0})
    assert b.last_price["TSLA"] == 250.0
