"""quantai.live.engine.TradingEngine 测试（同步分发，不依赖线程时序）。"""

from __future__ import annotations

from datetime import datetime

from quantai.live.engine import TradingEngine
from quantai.live.events import Event, EventType


class FakeBroker:
    def __init__(self):
        self.market_data = []
        self.orders = []

    def on_market_data(self, md):
        self.market_data.append(md)

    def place_order(self, order):
        self.orders.append(order)


class FakeStrategy:
    def __init__(self, out=None):
        self.out = out
        self.bars = []

    def on_bar(self, md):
        self.bars.append(md)
        return self.out


class FakePortfolio:
    def __init__(self):
        self.fills = []

    def on_fill(self, fill):
        self.fills.append(fill)


def _md_event(payload=None):
    return Event(EventType.MARKET_DATA, datetime.now(), payload or {"ticker": "NVDA", "close": 100.0})


def test_market_data_calls_broker_and_strategy():
    eng = TradingEngine()
    eng.broker = FakeBroker()
    eng.strategy = FakeStrategy(out={"ticker": "NVDA", "action": "BUY"})
    eng._handle_event(_md_event())
    assert len(eng.broker.market_data) == 1
    assert len(eng.strategy.bars) == 1
    # 策略产出被回灌成 SIGNAL 事件
    ev = eng.events.get_nowait()
    assert ev.type == EventType.SIGNAL
    assert ev.payload["action"] == "BUY"


def test_strategy_none_output_no_signal():
    eng = TradingEngine()
    eng.strategy = FakeStrategy(out=None)
    eng._handle_event(_md_event())
    assert eng.events.empty()


def test_strategy_list_output_fans_out():
    eng = TradingEngine()
    eng.strategy = FakeStrategy(out=[{"a": 1}, None, {"a": 2}])
    eng._handle_event(_md_event())
    sigs = [eng.events.get_nowait().payload for _ in range(eng.events.qsize())]
    assert sigs == [{"a": 1}, {"a": 2}]  # None 被过滤


def test_signal_event_places_order():
    eng = TradingEngine()
    eng.broker = FakeBroker()
    eng._handle_event(Event(EventType.SIGNAL, datetime.now(), {"action": "SELL"}))
    assert eng.broker.orders == [{"action": "SELL"}]


def test_fill_event_updates_portfolio():
    eng = TradingEngine()
    eng.portfolio = FakePortfolio()
    eng._handle_event(Event(EventType.FILL, datetime.now(), {"qty": 10}))
    assert eng.portfolio.fills == [{"qty": 10}]


def test_paused_drops_market_data():
    eng = TradingEngine()
    eng.broker = FakeBroker()
    eng.strategy = FakeStrategy(out={"x": 1})
    eng.paused = True
    eng._handle_event(_md_event())
    assert eng.broker.market_data == []
    assert eng.events.empty()


def test_paused_still_handles_fill():
    eng = TradingEngine()
    eng.portfolio = FakePortfolio()
    eng.paused = True
    eng._handle_event(Event(EventType.FILL, datetime.now(), {"q": 1}))
    assert eng.portfolio.fills == [{"q": 1}]


def test_broker_exception_does_not_break_strategy():
    class BadBroker:
        def on_market_data(self, md):
            raise RuntimeError("boom")

    eng = TradingEngine()
    eng.broker = BadBroker()
    eng.strategy = FakeStrategy(out={"ok": 1})
    eng._handle_event(_md_event())  # 不应抛
    assert eng.strategy.bars  # 策略仍被调用


def test_push_helpers():
    eng = TradingEngine()
    eng.push(EventType.LOG, "hello", priority=2)
    ev = eng.events.get_nowait()
    assert ev.type == EventType.LOG
    assert ev.payload == "hello"
    assert ev.priority == 2


def test_on_event_fallback():
    class EvStrategy:
        def on_event(self, event):
            return {"from": "on_event"}

    eng = TradingEngine()
    eng.strategy = EvStrategy()
    eng._handle_event(_md_event())
    assert eng.events.get_nowait().payload == {"from": "on_event"}
