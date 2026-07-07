"""events.py 单测：Gamma 响应解析（两种编码）、Yes 定位、过滤、抓取降级、ETL 幂等。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quantai.data.events import EventOdd, EventsFetcher, events_to_odds
from quantai.warehouse import connect, load_event_odds


def _event(markets=None, title="Fed Decision in July?"):
    return {
        "title": title,
        "volume24hr": 4307468.85,
        "markets": markets if markets is not None else [
            {
                "id": "558934",
                "conditionId": "0xabc",
                "question": "Will the Fed raise rates in July?",
                "outcomes": '["Yes", "No"]',  # 实测：可能是 JSON 字符串
                "outcomePrices": '["0.63", "0.37"]',
                "volume24hr": 4017611.16,
                "endDate": "2026-07-31T00:00:00Z",
            }
        ],
    }


class TestParsing:
    def test_json_string_encoded_fields(self):
        odds = events_to_odds([_event()], category="economy")
        assert len(odds) == 1
        o = odds[0]
        assert o.yes_price == pytest.approx(0.63)
        assert o.event_title == "Fed Decision in July?"
        assert o.category == "economy"

    def test_native_list_fields(self):
        ev = _event()
        ev["markets"][0]["outcomes"] = ["Yes", "No"]
        ev["markets"][0]["outcomePrices"] = ["0.4", "0.6"]
        assert events_to_odds([ev], "x")[0].yes_price == pytest.approx(0.4)

    def test_yes_position_respected(self):
        ev = _event()
        ev["markets"][0]["outcomes"] = '["No", "Yes"]'  # Yes 在第二位
        ev["markets"][0]["outcomePrices"] = '["0.8", "0.2"]'
        assert events_to_odds([ev], "x")[0].yes_price == pytest.approx(0.2)

    def test_no_yes_outcome_skipped(self):
        ev = _event()
        ev["markets"][0]["outcomes"] = '["Spain", "Brazil"]'
        assert events_to_odds([ev], "x") == []

    def test_out_of_range_price_skipped(self):
        ev = _event()
        ev["markets"][0]["outcomePrices"] = '["1.7", "-0.7"]'
        assert events_to_odds([ev], "x") == []

    def test_top_n_markets_by_volume(self):
        markets = []
        for i in range(8):
            m = _event()["markets"][0].copy()
            m["id"] = str(i)
            m["question"] = f"Q{i}"
            m["volume24hr"] = float(i)
            markets.append(m)
        odds = events_to_odds([_event(markets=markets)], "x", top_markets_per_event=3)
        assert [o.market_id for o in odds] == ["7", "6", "5"]  # 量最大的 3 个


class TestFetcher:
    class _FakeResp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    def test_fetch_tag_and_dedup_across_tags(self):
        calls = []

        def getter(url, params=None, timeout=None):
            calls.append(params["tag_slug"])
            return self._FakeResp([_event()])

        f = EventsFetcher(getter=getter)
        odds = f.fetch_all(["economy", "finance"])
        assert calls == ["economy", "finance"]
        assert len(odds) == 1  # 同一 market_id 跨 tag 去重

    def test_fetch_failure_returns_empty(self):
        def getter(url, params=None, timeout=None):
            raise RuntimeError("down")

        assert EventsFetcher(getter=getter).fetch_tag("economy") == []


class TestLoadEventOdds:
    def _odds(self, price=0.63):
        return [EventOdd(market_id="558934", condition_id="0xabc",
                         event_title="Fed Decision in July?",
                         question="Will the Fed raise rates in July?",
                         yes_price=price, volume_24h=4017611.16,
                         end_date="2026-07-31T00:00:00Z", category="economy")]

    def test_same_day_idempotent_cross_day_accumulates(self):
        con = connect(":memory:")
        try:
            assert load_event_odds(con, self._odds(0.63), as_of="2026-07-07") == 1
            load_event_odds(con, self._odds(0.70), as_of="2026-07-07")  # 同日覆盖
            assert con.execute("SELECT count(*) FROM raw.event_odds").fetchone()[0] == 1
            assert con.execute("SELECT yes_price FROM raw.event_odds").fetchone()[0] == 0.70
            load_event_odds(con, self._odds(0.55), as_of="2026-07-08")  # 跨日累积
            assert con.execute("SELECT count(*) FROM raw.event_odds").fetchone()[0] == 2
        finally:
            con.close()


class TestBriefSection:
    def test_events_in_brief(self):
        import numpy as np
        import pandas as pd

        from quantai.agents.analyst import build_brief
        from quantai.portfolio import Portfolio, PortfolioAnalyzer, Position

        close = pd.Series(np.linspace(90, 100, 250), index=pd.bdate_range("2023-01-02", periods=250))
        df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
        snap = PortfolioAnalyzer({"AAA": df}, df).analyze(
            Portfolio(positions=[Position(symbol="AAA", shares=1, cost_basis=90, open_date="2025-01-02")])
        )
        brief = build_brief(snap, [], {}, None, event_rows=[o.as_dict() for o in TestLoadEventOdds()._odds()])
        assert "事件概率" in brief and "Fed" in brief and "Yes 63%" in brief
        assert "非官方预测" in brief  # 诚实标注必须在

    def test_empty_events_honest(self):
        from quantai.agents.analyst import _events_section

        assert "无事件概率数据" in _events_section([])