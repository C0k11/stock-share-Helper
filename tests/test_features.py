"""
特征模块测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_price_data(days=300):
    """创建测试用价格数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    
    # 生成随机价格
    np.random.seed(42)
    returns = np.random.randn(days) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(days) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(days) * 0.01)),
        "low": close * (1 - np.abs(np.random.randn(days) * 0.01)),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    return df


class TestTechnicalFeatures:
    """测试技术指标"""
    
    def test_moving_averages(self):
        """测试移动平均"""
        from src.features.technical import TechnicalFeatures
        
        df = create_sample_price_data()
        tech = TechnicalFeatures(df)
        features = tech.add_moving_averages().get_features()
        
        assert "ma_20" in features.columns
        assert "ma_200" in features.columns
        assert "price_vs_ma20" in features.columns
    
    def test_momentum(self):
        """测试动量指标"""
        from src.features.technical import TechnicalFeatures
        
        df = create_sample_price_data()
        tech = TechnicalFeatures(df)
        features = tech.add_momentum().get_features()
        
        assert "return_21d" in features.columns
        assert "return_63d" in features.columns
    
    def test_volatility(self):
        """测试波动率指标"""
        from src.features.technical import TechnicalFeatures
        
        df = create_sample_price_data()
        tech = TechnicalFeatures(df)
        features = tech.add_volatility().get_features()
        
        assert "volatility_20d" in features.columns
        # 波动率应该是正数
        assert (features["volatility_20d"].dropna() >= 0).all()
    
    def test_drawdown(self):
        """测试回撤指标"""
        from src.features.technical import TechnicalFeatures
        
        df = create_sample_price_data()
        tech = TechnicalFeatures(df)
        features = tech.add_drawdown().get_features()
        
        assert "drawdown" in features.columns
        # 回撤应该是负数或零
        assert (features["drawdown"].dropna() <= 0).all()
    
    def test_add_all(self):
        """测试添加所有指标"""
        from src.features.technical import TechnicalFeatures
        
        df = create_sample_price_data()
        tech = TechnicalFeatures(df)
        features = tech.add_all().get_features()
        
        # 应该包含多种指标
        assert len(features.columns) > 10


class TestRegimeDetector:
    """测试风险状态检测"""
    
    def test_detect_regime(self):
        """测试风险状态检测"""
        from src.features.regime import RegimeDetector
        
        df = create_sample_price_data()
        detector = RegimeDetector()
        
        result = detector.get_current_regime(df)
        
        assert "regime" in result
        assert result["regime"] in ["risk_on", "risk_off", "transition"]
    
    def test_regime_history(self):
        """测试历史风险状态"""
        from src.features.regime import RegimeDetector
        
        df = create_sample_price_data()
        detector = RegimeDetector()
        
        history = detector.get_regime_history(df)
        
        assert "regime" in history.columns
        assert len(history) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
