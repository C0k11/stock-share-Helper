#!/usr/bin/env python
"""
运行回测脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.data.storage import DataStorage
from src.features.technical import TechnicalFeatures
from src.features.regime import RegimeDetector
from src.strategy.signals import SignalGenerator
from src.strategy.position import PositionSizer
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.costs import CostModel
from src.backtest.metrics import PerformanceMetrics


def run_momentum_strategy(
    symbols: list = ["SPY", "TLT", "GLD"],
    initial_capital: float = 100000,
    target_volatility: float = 0.10
):
    """
    运行动量策略回测
    
    简单策略逻辑：
    - 根据趋势信号和风险状态分配仓位
    - 使用目标波动率控制风险
    """
    logger.info("=" * 50)
    logger.info("Running Momentum Strategy Backtest")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Initial Capital: ${initial_capital:,.0f}")
    logger.info(f"Target Volatility: {target_volatility:.1%}")
    logger.info("=" * 50)
    
    # 加载数据
    storage = DataStorage()
    price_data = {}
    
    for symbol in symbols:
        df = storage.load_price_data(symbol)
        if df is not None:
            price_data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} rows")
        else:
            logger.warning(f"No data for {symbol}, skipping")
    
    if not price_data:
        logger.error("No data loaded, exiting")
        return
    
    # 加载SPY用于regime检测
    spy_data = price_data.get("SPY")
    if spy_data is None:
        spy_data = storage.load_price_data("SPY")
    
    # 初始化组件
    signal_gen = SignalGenerator()
    position_sizer = PositionSizer(target_volatility=target_volatility)
    regime_detector = RegimeDetector()
    
    # 定义信号函数
    def generate_weights(data, date_str):
        weights = {}
        
        # 获取风险状态
        if spy_data is not None:
            spy_slice = spy_data.loc[:date_str]
            if len(spy_slice) > 200:
                regime_result = regime_detector.get_current_regime(spy_slice)
                regime = regime_result["regime"]
            else:
                regime = "transition"
        else:
            regime = "transition"
        
        # 为每个标的计算权重
        for symbol, df in data.items():
            df_slice = df.loc[:date_str]
            
            if len(df_slice) < 200:
                weights[symbol] = 0.0
                continue
            
            # 获取信号
            signal_result = signal_gen.get_current_signal(df_slice)
            
            # 计算基础仓位
            signal_position = {
                "strong_long": 1.0,
                "weak_long": 0.6,
                "neutral": 0.3,
                "weak_short": 0.1,
                "strong_short": 0.0
            }.get(signal_result["strength"], 0.3)
            
            # 应用风险状态调整
            regime_factor = {
                "risk_on": 1.0,
                "transition": 0.7,
                "risk_off": 0.4
            }.get(regime, 0.7)
            
            # 计算目标波动率仓位
            vol_position = position_sizer.compute_vol_target_position(df_slice).iloc[-1]
            
            # 最终权重
            base_weight = 1.0 / len(data)  # 等权基础
            weights[symbol] = base_weight * min(signal_position, vol_position) * regime_factor
        
        # 归一化
        total = sum(weights.values())
        if total > 1:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    # 运行回测
    config = BacktestConfig(
        initial_capital=initial_capital,
        rebalance_frequency="weekly"
    )
    engine = BacktestEngine(config=config, cost_model=CostModel.us_stock())
    
    result = engine.run(price_data, generate_weights, list(price_data.keys()))
    
    # 打印结果
    metrics_calc = PerformanceMetrics()
    report = metrics_calc.format_report(result.metrics)
    print(report)
    
    # 保存结果
    result.equity_curve.to_csv("data/backtest_equity.csv")
    logger.info("Equity curve saved to data/backtest_equity.csv")
    
    return result


def run_equal_weight_benchmark(
    symbols: list = ["SPY", "TLT", "GLD"],
    initial_capital: float = 100000
):
    """运行等权重基准"""
    logger.info("Running Equal Weight Benchmark")
    
    storage = DataStorage()
    price_data = {}
    
    for symbol in symbols:
        df = storage.load_price_data(symbol)
        if df is not None:
            price_data[symbol] = df
    
    # 固定等权重
    weight = 1.0 / len(symbols)
    weights = {s: weight for s in symbols}
    
    def fixed_weights(data, date_str):
        return weights
    
    config = BacktestConfig(
        initial_capital=initial_capital,
        rebalance_frequency="monthly"
    )
    engine = BacktestEngine(config=config, cost_model=CostModel.us_stock())
    
    result = engine.run(price_data, fixed_weights, list(price_data.keys()))
    
    metrics_calc = PerformanceMetrics()
    report = metrics_calc.format_report(result.metrics)
    print("\n=== Equal Weight Benchmark ===")
    print(report)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行回测")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "TLT", "GLD"])
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--benchmark", action="store_true", help="同时运行基准")
    
    args = parser.parse_args()
    
    # 运行策略
    result = run_momentum_strategy(
        symbols=args.symbols,
        initial_capital=args.capital,
        target_volatility=args.target_vol
    )
    
    # 运行基准对比
    if args.benchmark:
        benchmark = run_equal_weight_benchmark(
            symbols=args.symbols,
            initial_capital=args.capital
        )
