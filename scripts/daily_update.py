#!/usr/bin/env python
"""
每日更新脚本 - 更新数据、生成建议
"""

import sys
from pathlib import Path
from datetime import datetime, date
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.data.fetcher import DataFetcher
from src.data.storage import DataStorage
from src.features.technical import TechnicalFeatures
from src.features.regime import RegimeDetector
from src.strategy.signals import SignalGenerator
from src.strategy.position import PositionSizer
from src.strategy.rules import TradingRules
from src.risk.alerts import RiskAlerts


def daily_update(
    symbols: list = ["SPY", "QQQ", "TLT", "IEF", "GLD", "SHY"],
    risk_profile: str = "balanced"
):
    """
    执行每日更新
    
    1. 更新最新数据
    2. 计算信号和风险状态
    3. 生成投资建议
    4. 检查风险预警
    """
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"=" * 50)
    logger.info(f"Daily Update: {today}")
    logger.info(f"Risk Profile: {risk_profile}")
    logger.info(f"=" * 50)
    
    # 1. 更新数据
    logger.info("Step 1: Updating price data...")
    fetcher = DataFetcher()
    storage = DataStorage()
    
    # 获取最近30天数据（确保有足够的历史）
    from datetime import timedelta
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    new_data = fetcher.fetch_price(symbols + ["^VIX"], start)
    
    for symbol, df in new_data.items():
        # 合并到现有数据
        existing = storage.load_price_data(symbol.replace("^", ""))
        if existing is not None:
            # 合并并去重
            combined = existing.combine_first(df)
            storage.save_price_data(symbol.replace("^", ""), combined)
        else:
            storage.save_price_data(symbol.replace("^", ""), df)
    
    # 2. 计算风险状态
    logger.info("Step 2: Detecting market regime...")
    spy_data = storage.load_price_data("SPY")
    vix_data = storage.load_price_data("VIX")
    
    regime_detector = RegimeDetector()
    regime_result = regime_detector.get_current_regime(spy_data, vix_data)
    
    logger.info(f"Current Regime: {regime_result['regime']}")
    logger.info(f"Regime Score: {regime_result['score']}")
    if regime_result.get('vix'):
        logger.info(f"VIX: {regime_result['vix']:.1f}")
    
    # 3. 生成投资建议
    logger.info("Step 3: Generating recommendations...")
    
    signal_gen = SignalGenerator()
    position_sizer = PositionSizer()
    trading_rules = TradingRules()
    
    # 风险档位参数
    profile_params = {
        "conservative": {"max_position": 0.25, "target_vol": 0.06},
        "balanced": {"max_position": 0.35, "target_vol": 0.10},
        "aggressive": {"max_position": 0.50, "target_vol": 0.15}
    }
    params = profile_params.get(risk_profile, profile_params["balanced"])
    
    recommendations = []
    
    for symbol in symbols:
        df = storage.load_price_data(symbol)
        if df is None or len(df) < 200:
            continue
        
        # 计算信号
        signal = signal_gen.get_current_signal(df)
        
        # 计算特征
        features = TechnicalFeatures(df).add_all().get_latest()
        
        # 计算目标仓位
        base_position = {
            "strong_long": 1.0,
            "weak_long": 0.6,
            "neutral": 0.3,
            "weak_short": 0.1,
            "strong_short": 0.0
        }.get(signal["strength"], 0.3)
        
        regime_factor = {
            "risk_on": 1.0,
            "transition": 0.7,
            "risk_off": 0.4
        }.get(regime_result["regime"], 0.7)
        
        target_position = min(
            base_position * regime_factor * (1 / len(symbols)),
            params["max_position"]
        )
        
        # 生成交易信号
        current_price = df["close"].iloc[-1]
        trade_signal = trading_rules.generate_trade_signal(
            symbol=symbol,
            current_position=0.2,  # TODO: 从状态获取实际持仓
            target_position=target_position,
            current_price=current_price,
            reason=f"Regime: {regime_result['regime']}, Signal: {signal['strength']}"
        )
        
        rec = trading_rules.format_recommendation(trade_signal)
        rec["current_price"] = round(current_price, 2)
        rec["volatility"] = round(features.get("volatility_20d", 0) * 100, 1)
        rec["momentum"] = round(features.get("return_63d", 0) * 100, 1)
        
        recommendations.append(rec)
        
        logger.info(f"{symbol}: {rec['action']} -> {rec['target_position']}%")
    
    # 4. 检查风险预警
    logger.info("Step 4: Checking risk alerts...")
    
    risk_alerts = RiskAlerts()
    # TODO: 获取实际组合数据
    alerts_summary = {"total_alerts": 0, "alerts": []}
    
    if regime_result["regime"] == "risk_off":
        alerts_summary["alerts"].append({
            "type": "regime_change",
            "severity": "high",
            "message": "市场处于Risk-Off状态，建议降低风险敞口"
        })
        alerts_summary["total_alerts"] += 1
    
    # 5. 生成报告
    report = {
        "date": today,
        "regime": regime_result,
        "recommendations": recommendations,
        "risk_alerts": alerts_summary,
        "risk_profile": risk_profile
    }
    
    # 保存报告
    report_path = Path("data/reports")
    report_path.mkdir(parents=True, exist_ok=True)
    
    with open(report_path / f"daily_{today}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    
    logger.info(f"Report saved to data/reports/daily_{today}.json")
    
    # 打印摘要
    print("\n" + "=" * 50)
    print(f"Daily Recommendations - {today}")
    print("=" * 50)
    print(f"\nMarket Regime: {regime_result['regime'].upper()}")
    print(f"VIX: {regime_result.get('vix', 'N/A')}")
    print(f"\nRecommendations ({risk_profile} profile):")
    print("-" * 40)
    
    for rec in recommendations:
        print(f"  {rec['symbol']:6s} | {rec['action']:8s} | 目标: {rec['target_position']:5.1f}% | 价格: ${rec['current_price']}")
    
    if alerts_summary["alerts"]:
        print("\nRisk Alerts:")
        for alert in alerts_summary["alerts"]:
            print(f"  [{alert['severity'].upper()}] {alert['message']}")
    
    print("=" * 50)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="每日更新")
    parser.add_argument("--profile", choices=["conservative", "balanced", "aggressive"],
                        default="balanced", help="风险档位")
    parser.add_argument("--symbols", nargs="+", 
                        default=["SPY", "QQQ", "TLT", "IEF", "GLD", "SHY"])
    
    args = parser.parse_args()
    
    daily_update(symbols=args.symbols, risk_profile=args.profile)
