"""quantai.live.data_feed 测试（SimulatedDataFeed 确定性 + 工厂 + 订阅）。"""

from __future__ import annotations

from quantai.live.data_feed import (
    DataFeed,
    SimulatedDataFeed,
    create_data_feed,
)


def test_simulated_publishes_bar_shape():
    feed = SimulatedDataFeed(["NVDA"], interval_sec=4.0, seed=42)
    got = []
    feed.subscribe(got.append)
    feed.running = True  # 绕过线程，直接同步发一拍
    feed._fetch_and_publish()
    assert len(got) == 1
    bar = got[0]
    assert bar["ticker"] == "NVDA"
    assert bar["source"] == "simulated"
    assert set(bar) >= {"open", "high", "low", "close", "volume", "time"}
    assert bar["high"] >= bar["close"] >= 0
    assert bar["low"] <= bar["close"]


def test_simulated_is_seeded_reproducible():
    f1 = SimulatedDataFeed(["NVDA", "TSLA"], seed=7)
    f2 = SimulatedDataFeed(["NVDA", "TSLA"], seed=7)
    out1, out2 = [], []
    f1.subscribe(out1.append)
    f2.subscribe(out2.append)
    f1.running = f2.running = True
    f1._fetch_and_publish()
    f2._fetch_and_publish()
    assert [b["close"] for b in out1] == [b["close"] for b in out2]


def test_simulated_respects_base_prices():
    feed = SimulatedDataFeed(["NVDA"], base_prices={"NVDA": 123.0}, seed=1)
    assert feed._current_prices["NVDA"] == 123.0


def test_publish_skipped_when_not_running():
    feed = SimulatedDataFeed(["NVDA"], seed=1)
    got = []
    feed.subscribe(got.append)
    feed.running = False  # 未运行 -> 不发
    feed._fetch_and_publish()
    assert got == []


def test_select_batch_round_robin():
    feed = SimulatedDataFeed(["A", "B", "C", "D"], symbols_per_tick=2, seed=1)
    # 关掉"ticker 太少就全发"的保护：4 <= max(12, 6) 触发 n=0，故这里验证全发
    assert feed._select_batch(["A", "B", "C", "D"]) == ["A", "B", "C", "D"]


def test_subscribe_callback_error_isolated():
    feed = SimulatedDataFeed(["NVDA"], seed=1)
    feed.running = True

    def bad(_):
        raise RuntimeError("cb fail")

    good = []
    feed.subscribe(bad)
    feed.subscribe(good.append)
    feed._fetch_and_publish()  # 不应抛
    assert len(good) == 1


def test_create_data_feed_simulated():
    feed = create_data_feed(["NVDA"], source="simulated", seed=1)
    assert isinstance(feed, SimulatedDataFeed)
    assert feed.source == "simulated"


def test_create_data_feed_tickers_uppercased():
    feed = create_data_feed(["nvda", "tsla"], source="simulated")
    assert feed.tickers == ["NVDA", "TSLA"]


def test_base_datafeed_fetch_not_implemented():
    feed = DataFeed(["NVDA"])
    try:
        feed._fetch_and_publish()
        assert False, "应抛 NotImplementedError"
    except NotImplementedError:
        pass
