"""本地数据持久化（parquet）。

从旧 `src/data/storage.py` 的 `DataStorage` 迁移并加类型标注；行为保持一致，
重命名为 `ParquetStorage` 以点明存储后端（单一职责更清晰）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class ParquetStorage:
    """把行情/特征数据按类别存成 parquet 文件。

    目录结构：<base>/{raw, processed, cache, features}/
    """

    SUBDIRS: tuple[str, ...] = ("raw", "processed", "cache", "features")

    def __init__(self, base_path: str | Path = "data") -> None:
        self.base_path = Path(base_path)
        self._ensure_dirs()
        logger.info(f"ParquetStorage initialized at: {self.base_path}")

    def _ensure_dirs(self) -> None:
        for sub in self.SUBDIRS:
            (self.base_path / sub).mkdir(parents=True, exist_ok=True)

    # --- 行情数据 --- #
    def save_prices(self, symbol: str, df: pd.DataFrame, category: str = "raw") -> Path:
        """保存某标的的价格数据，返回写入路径。"""
        path = self.base_path / category / f"{symbol}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved {symbol} -> {path} ({len(df)} rows)")
        return path

    def load_prices(self, symbol: str, category: str = "raw") -> Optional[pd.DataFrame]:
        """加载某标的的价格数据；不存在返回 None。"""
        path = self.base_path / category / f"{symbol}.parquet"
        if not path.exists():
            logger.warning(f"Price data not found: {path}")
            return None
        return pd.read_parquet(path)

    # --- 特征数据 --- #
    def save_features(self, symbol: str, df: pd.DataFrame, feature_set: str = "default") -> Path:
        path = self.base_path / "features" / f"{symbol}_{feature_set}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved features {symbol}/{feature_set} -> {path}")
        return path

    def load_features(self, symbol: str, feature_set: str = "default") -> Optional[pd.DataFrame]:
        path = self.base_path / "features" / f"{symbol}_{feature_set}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    # --- 元信息 --- #
    def list_symbols(self, category: str = "raw") -> list[str]:
        """列出某类别下已存储的标的代码（按文件名）。"""
        directory = self.base_path / category
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.parquet"))

    def info(self, symbol: str, category: str = "raw") -> dict:
        """返回某标的数据的概要（行数/列/日期范围/文件大小）。"""
        path = self.base_path / category / f"{symbol}.parquet"
        df = self.load_prices(symbol, category)
        if df is None:
            return {"exists": False}
        return {
            "exists": True,
            "rows": len(df),
            "columns": list(df.columns),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 4),
        }
