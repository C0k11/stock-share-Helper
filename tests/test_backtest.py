"""
回测模块测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


def create_sample_price_data(days=300, seed=42):
    """创建测试用价格数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    np.random.seed(seed)
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


class TestCostModel:
    """测试成本模型"""
    
    def test_compute_cost(self):
        """测试成本计算"""
        from src.backtest.costs import CostModel
        
        model = CostModel(
            commission_rate=0.001,
            min_commission=1.0,
            slippage_bps=5.0
        )
        
        cost = model.compute_cost(10000)
        
        # 佣金: 10000 * 0.001 = 10
        # 滑点: 10000 * 5/10000 = 5
        # 总计: 15
        assert cost == 15.0
    
    def test_min_commission(self):
        """测试最低佣金"""
        from src.backtest.costs import CostModel
        
        model = CostModel(
            commission_rate=0.001,
            min_commission=5.0,
            slippage_bps=0
        )
        
        # 小额交易应该使用最低佣金
        cost = model.compute_cost(100)
        assert cost == 5.0
    
    def test_market_presets(self):
        """测试市场预设"""
        from src.backtest.costs import CostModel
        
        us_model = CostModel.us_stock()
        hk_model = CostModel.hk_stock()
        cn_model = CostModel.cn_stock()
        
        # 港股成本应该高于美股
        us_cost = us_model.compute_cost(10000)
        hk_cost = hk_model.compute_cost(10000)
        
        assert hk_cost > us_cost


class TestPerformanceMetrics:
    """测试绩效指标"""
    
    def test_total_return(self):
        """测试总收益率"""
        from src.backtest.metrics import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        # 从100涨到120
        equity = pd.Series([100, 105, 110, 115, 120])
        
        total_ret = metrics.total_return(equity)
        assert abs(total_ret - 0.2) < 0.001
    
    def test_max_drawdown(self):
        """测试最大回撤"""
        from src.backtest.metrics import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        # 100 -> 120 -> 100 -> 110
        equity = pd.Series([100, 110, 120, 110, 100, 110])
        
        max_dd = metrics.max_drawdown(equity)
        
        # 最大回撤: (100-120)/120 = -16.67%
        assert abs(max_dd - (-0.1667)) < 0.01
    
    def test_sharpe_ratio(self):
        """测试夏普比率"""
        from src.backtest.metrics import PerformanceMetrics
        
        metrics = PerformanceMetrics(risk_free_rate=0.02)
        
        # 创建收益序列
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # 日收益
        
        sharpe = metrics.sharpe_ratio(returns)
        
        # 应该返回一个合理的数值
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5


class TestBacktestEngine:
    """测试回测引擎"""
    
    def test_simple_backtest(self):
        """测试简单回测"""
        from src.backtest.engine import BacktestEngine, BacktestConfig
        
        # 创建测试数据
        price_data = {
            "SPY": create_sample_price_data(seed=42),
            "TLT": create_sample_price_data(seed=43)
        }
        
        # 固定权重策略
        def fixed_weights(data, date):
            return {"SPY": 0.6, "TLT": 0.4}
        
        config = BacktestConfig(initial_capital=100000)
        engine = BacktestEngine(config=config)
        
        result = engine.run(price_data, fixed_weights)
        
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        assert result.metrics is not None
        assert "total_return" in result.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
