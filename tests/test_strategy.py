"""
策略模块测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


def create_sample_price_data(days=300):
    """创建测试用价格数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
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


class TestSignalGenerator:
    """测试信号生成"""
    
    def test_generate_signals(self):
        """测试信号生成"""
        from src.strategy.signals import SignalGenerator
        
        df = create_sample_price_data()
        gen = SignalGenerator()
        
        signals = gen.generate(df)
        
        assert "trend_signal" in signals.columns
        assert "momentum_signal" in signals.columns
        assert "composite_signal" in signals.columns
    
    def test_get_current_signal(self):
        """测试获取当前信号"""
        from src.strategy.signals import SignalGenerator
        
        df = create_sample_price_data()
        gen = SignalGenerator()
        
        result = gen.get_current_signal(df)
        
        assert "trend" in result
        assert "momentum" in result
        assert "strength" in result
        assert result["strength"] in ["strong_long", "weak_long", "neutral", "weak_short", "strong_short"]


class TestPositionSizer:
    """测试仓位计算"""
    
    def test_vol_target_position(self):
        """测试波动率目标仓位"""
        from src.strategy.position import PositionSizer
        
        df = create_sample_price_data()
        sizer = PositionSizer(target_volatility=0.10)
        
        position = sizer.compute_vol_target_position(df)
        
        assert len(position) == len(df)
        # 仓位应该在0-1之间
        assert (position.dropna() >= 0).all()
        assert (position.dropna() <= 1.5).all()  # 允许一定超过1
    
    def test_apply_risk_profile(self):
        """测试风险档位约束"""
        from src.strategy.position import PositionSizer
        
        sizer = PositionSizer()
        
        # 保守档位应该限制仓位
        conservative = sizer.apply_risk_profile(0.8, "conservative")
        assert conservative <= 0.4
        
        # 进取档位允许更高仓位
        aggressive = sizer.apply_risk_profile(0.8, "aggressive")
        assert aggressive >= conservative


class TestTradingRules:
    """测试交易规则"""
    
    def test_check_stop_loss(self):
        """测试止损检查"""
        from src.strategy.rules import TradingRules
        
        rules = TradingRules(stop_loss_pct=0.08)
        
        # 亏损10%应该触发止损
        assert rules.check_stop_loss(90, 100) == True
        
        # 亏损5%不应该触发止损
        assert rules.check_stop_loss(95, 100) == False
    
    def test_check_take_profit(self):
        """测试止盈检查"""
        from src.strategy.rules import TradingRules
        
        rules = TradingRules(take_profit_pct=0.25)
        
        # 盈利30%应该触发止盈
        assert rules.check_take_profit(130, 100) == True
        
        # 盈利10%不应该触发止盈
        assert rules.check_take_profit(110, 100) == False
    
    def test_determine_action(self):
        """测试动作决定"""
        from src.strategy.rules import TradingRules, Action
        
        rules = TradingRules()
        
        # 从0加到0.5应该是买入
        action = rules.determine_action(0, 0.5, 100)
        assert action == Action.BUY
        
        # 从0.5减到0应该是清仓
        action = rules.determine_action(0.5, 0, 100)
        assert action == Action.CLEAR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
