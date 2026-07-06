"""news.py 单测：纯解析（零网络）、坏条目丢弃、抓取失败降级、ETL link 去重幂等。"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from quantai.data.news import NewsFetcher, entries_to_items
from quantai.warehouse import connect, load_news


def _entry(title="Fed holds rates", link="https://x.test/a", published=True, **kw):
    d = {"title": title, "link": link, "summary": kw.get("summary", "sum")}
    if published:
        d["published_parsed"] = time.struct_time((2026, 7, 6, 12, 30, 0, 0, 187, 0))
    return d


class _FakeParser:
    """feedparser.parse 替身：记录 URL，返回预置 entries。"""

    def __init__(self, entries=None, raise_exc=False):
        self.entries = entries if entries is not None else [_entry()]
        self.raise_exc = raise_exc
        self.urls: list[str] = []

    def __call__(self, url):
        self.urls.append(url)
        if self.raise_exc:
            raise RuntimeError("network down")
        return SimpleNamespace(entries=self.entries)


class TestParsing:
    def test_entries_to_items(self):
        items = entries_to_items([_entry()], symbol="spcx", source="yahoo")
        assert len(items) == 1
        it = items[0]
        assert it.symbol == "SPCX"  # 大写归一
        assert it.title == "Fed holds rates"
        assert it.published is not None and it.published.year == 2026

    def test_missing_published_is_none_not_fabricated(self):
        items = entries_to_items([_entry(published=False)], symbol="A", source="s")
        assert items[0].published is None

    def test_blank_title_or_link_dropped(self):
        entries = [_entry(title="  "), _entry(link=""), _entry(link="https://x.test/ok")]
        items = entries_to_items(entries, symbol=None, source="s")
        assert len(items) == 1 and items[0].symbol is None


class TestFetcher:
    def test_fetch_symbol_news_uses_yahoo_url_and_limit(self):
        parser = _FakeParser(entries=[_entry(link=f"https://x.test/{i}") for i in range(30)])
        items = NewsFetcher(parser=parser).fetch_symbol_news("spcx", limit=5)
        assert len(items) == 5
        assert "s=SPCX" in parser.urls[0]

    def test_fetch_failure_returns_empty_not_raises(self):
        items = NewsFetcher(parser=_FakeParser(raise_exc=True)).fetch_symbol_news("SPCX")
        assert items == []

    def test_fetch_all_includes_extra_feeds(self):
        parser = _FakeParser()
        items = NewsFetcher(extra_feeds=["https://feed.test/rss"], parser=parser).fetch_all(["AAA"])
        assert len(parser.urls) == 2  # 1 ticker + 1 extra feed
        assert any(i.symbol == "AAA" for i in items)
        assert any(i.symbol is None for i in items)  # 通用源不绑定标的


class TestLoadNews:
    def test_dedup_by_link_idempotent(self):
        con = connect(":memory:")
        try:
            items = entries_to_items(
                [_entry(link="https://x.test/1"), _entry(link="https://x.test/2")],
                symbol="AAA", source="yahoo",
            )
            assert load_news(con, items) == 2
            assert load_news(con, items) == 0  # 重跑：link 已存在，零新增
            n = con.execute("SELECT count(*) FROM raw.news").fetchone()[0]
            assert n == 2
        finally:
            con.close()

    def test_batch_internal_duplicates_collapsed(self):
        con = connect(":memory:")
        try:
            items = entries_to_items(
                [_entry(link="https://x.test/same"), _entry(link="https://x.test/same")],
                symbol="AAA", source="yahoo",
            )
            assert load_news(con, items) == 1
        finally:
            con.close()

    def test_empty_input(self):
        con = connect(":memory:")
        try:
            assert load_news(con, []) == 0
        finally:
            con.close()
