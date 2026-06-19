"""quantai.data.storage 的读写往返测试（无网络，用 tmp_path）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantai.data.storage import ParquetStorage


def _sample_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    idx.name = "date"
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [100, 200, 300],
        },
        index=idx,
    )


def test_init_creates_subdirs(tmp_path: Path) -> None:
    ParquetStorage(tmp_path)
    for sub in ("raw", "processed", "cache", "features"):
        assert (tmp_path / sub).is_dir()


def test_save_load_prices_roundtrip(tmp_path: Path) -> None:
    store = ParquetStorage(tmp_path)
    df = _sample_df()
    path = store.save_prices("SPY", df)
    assert path.exists()
    loaded = store.load_prices("SPY")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_load_missing_returns_none(tmp_path: Path) -> None:
    store = ParquetStorage(tmp_path)
    assert store.load_prices("NOPE") is None
    assert store.load_features("NOPE") is None


def test_features_roundtrip(tmp_path: Path) -> None:
    store = ParquetStorage(tmp_path)
    df = _sample_df()
    store.save_features("SPY", df, feature_set="tech")
    loaded = store.load_features("SPY", feature_set="tech")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_list_symbols_and_info(tmp_path: Path) -> None:
    store = ParquetStorage(tmp_path)
    store.save_prices("SPY", _sample_df())
    store.save_prices("QQQ", _sample_df())
    assert store.list_symbols() == ["QQQ", "SPY"]

    info = store.info("SPY")
    assert info["exists"] is True
    assert info["rows"] == 3
    assert "close" in info["columns"]
    assert store.info("NOPE") == {"exists": False}
