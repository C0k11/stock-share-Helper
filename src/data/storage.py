"""
数据存储模块 - 本地数据持久化与缓存
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from loguru import logger


class DataStorage:
    """数据存储管理"""
    
    def __init__(self, base_path: str = "data"):
        """
        Args:
            base_path: 数据存储根目录
        """
        self.base_path = Path(base_path)
        self._ensure_dirs()
        logger.info(f"DataStorage initialized at: {self.base_path}")
    
    def _ensure_dirs(self):
        """确保目录结构存在"""
        dirs = ["raw", "processed", "cache", "features"]
        for d in dirs:
            (self.base_path / d).mkdir(parents=True, exist_ok=True)
    
    def save_price_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        category: str = "raw"
    ):
        """
        保存价格数据
        
        Args:
            symbol: 标的代码
            df: 价格DataFrame
            category: 存储类别 raw/processed
        """
        path = self.base_path / category / f"{symbol}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved {symbol} to {path}")
    
    def load_price_data(
        self,
        symbol: str,
        category: str = "raw"
    ) -> Optional[pd.DataFrame]:
        """
        加载价格数据
        
        Args:
            symbol: 标的代码
            category: 存储类别
        
        Returns:
            价格DataFrame，不存在则返回None
        """
        path = self.base_path / category / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded {symbol} from {path}, {len(df)} rows")
            return df
        else:
            logger.warning(f"Data not found: {path}")
            return None
    
    def save_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        feature_set: str = "default"
    ):
        """保存特征数据"""
        path = self.base_path / "features" / f"{symbol}_{feature_set}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved features for {symbol} to {path}")
    
    def load_features(
        self,
        symbol: str,
        feature_set: str = "default"
    ) -> Optional[pd.DataFrame]:
        """加载特征数据"""
        path = self.base_path / "features" / f"{symbol}_{feature_set}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def list_symbols(self, category: str = "raw") -> List[str]:
        """列出已存储的标的"""
        path = self.base_path / category
        symbols = [f.stem for f in path.glob("*.parquet")]
        return symbols
    
    def get_data_info(self, symbol: str, category: str = "raw") -> dict:
        """获取数据信息"""
        df = self.load_price_data(symbol, category)
        if df is None:
            return {"exists": False}
        
        return {
            "exists": True,
            "rows": len(df),
            "columns": list(df.columns),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "file_size_mb": (self.base_path / category / f"{symbol}.parquet").stat().st_size / 1024 / 1024
        }
