"""watchlist.py 单测：加载/保存/增删、归一化、边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantai.data.watchlist import add_symbol, load_watchlist, remove_symbol, save_watchlist


class TestWatchlist:
    def test_missing_file_empty(self, tmp_path: Path):
        assert load_watchlist(tmp_path / "nope.yaml") == []

    def test_roundtrip_normalized(self, tmp_path: Path):
        f = tmp_path / "w.yaml"
        out = save_watchlist(["spy", " nvda ", "SPY", "btc-usd"], f)
        assert out == ["SPY", "NVDA", "BTC-USD"]  # 大写、去空格、去重保序
        assert load_watchlist(f) == ["SPY", "NVDA", "BTC-USD"]

    def test_bare_list_tolerated(self, tmp_path: Path):
        f = tmp_path / "w.yaml"
        f.write_text("- spy\n- qqq\n", encoding="utf-8")
        assert load_watchlist(f) == ["SPY", "QQQ"]

    def test_add_idempotent(self, tmp_path: Path):
        f = tmp_path / "w.yaml"
        save_watchlist(["SPY"], f)
        assert add_symbol("nvda", f) == ["SPY", "NVDA"]
        assert add_symbol("NVDA", f) == ["SPY", "NVDA"]  # 重复添加不翻倍

    def test_remove(self, tmp_path: Path):
        f = tmp_path / "w.yaml"
        save_watchlist(["SPY", "NVDA"], f)
        assert remove_symbol("spy", f) == ["NVDA"]
        assert remove_symbol("GONE", f) == ["NVDA"]  # 不存在不报错

    def test_bad_toplevel_rejected(self, tmp_path: Path):
        f = tmp_path / "w.yaml"
        f.write_text("just a string", encoding="utf-8")
        with pytest.raises(ValueError):
            load_watchlist(f)

    def test_repo_example_valid(self):
        example = Path(__file__).resolve().parents[2] / "watchlist.example.yaml"
        assert len(load_watchlist(example)) >= 1
