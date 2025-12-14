"""
绩效指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class PerformanceMetrics:
    """绩效指标计算器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
    
    def compute_all(self, equity_curve: pd.Series) -> Dict:
        """计算所有绩效指标"""
        returns = equity_curve.pct_change().dropna()
        
        return {
            # 收益指标
            "total_return": self.total_return(equity_curve),
            "annual_return": self.annual_return(equity_curve),
            "monthly_returns": self.monthly_returns(equity_curve),
            
            # 风险指标
            "volatility": self.volatility(returns),
            "max_drawdown": self.max_drawdown(equity_curve),
            "max_drawdown_duration": self.max_drawdown_duration(equity_curve),
            
            # 风险调整收益
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "calmar_ratio": self.calmar_ratio(equity_curve),
            
            # 其他
            "win_rate": self.win_rate(returns),
            "profit_factor": self.profit_factor(returns),
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            
            # 时间信息
            "start_date": str(equity_curve.index[0]),
            "end_date": str(equity_curve.index[-1]),
            "trading_days": len(equity_curve)
        }
    
    def total_return(self, equity_curve: pd.Series) -> float:
        """总收益率"""
        return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
    
    def annual_return(self, equity_curve: pd.Series) -> float:
        """年化收益率"""
        total_ret = self.total_return(equity_curve)
        years = len(equity_curve) / 252
        if years <= 0:
            return 0.0
        return float((1 + total_ret) ** (1 / years) - 1)
    
    def monthly_returns(self, equity_curve: pd.Series) -> Dict:
        """月度收益统计"""
        monthly = equity_curve.resample("M").last().pct_change().dropna()
        return {
            "mean": float(monthly.mean()),
            "std": float(monthly.std()),
            "best": float(monthly.max()),
            "worst": float(monthly.min()),
            "positive_months": int((monthly > 0).sum()),
            "negative_months": int((monthly < 0).sum())
        }
    
    def volatility(self, returns: pd.Series) -> float:
        """年化波动率"""
        return float(returns.std() * np.sqrt(252))
    
    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """最大回撤"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())
    
    def max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """最大回撤持续时间（天）"""
        peak = equity_curve.expanding().max()
        is_underwater = equity_curve < peak
        
        if not is_underwater.any():
            return 0
        
        # 计算每段回撤的持续时间
        groups = (~is_underwater).cumsum()
        durations = is_underwater.groupby(groups).sum()
        
        return int(durations.max())
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """夏普比率"""
        excess_return = returns.mean() * 252 - self.risk_free_rate
        vol = returns.std() * np.sqrt(252)
        
        if vol == 0:
            return 0.0
        return float(excess_return / vol)
    
    def sortino_ratio(self, returns: pd.Series) -> float:
        """索提诺比率（只考虑下行风险）"""
        excess_return = returns.mean() * 252 - self.risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float("inf")
        
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        if downside_vol == 0:
            return 0.0
        return float(excess_return / downside_vol)
    
    def calmar_ratio(self, equity_curve: pd.Series) -> float:
        """卡玛比率（年化收益/最大回撤）"""
        annual_ret = self.annual_return(equity_curve)
        max_dd = abs(self.max_drawdown(equity_curve))
        
        if max_dd == 0:
            return float("inf")
        return float(annual_ret / max_dd)
    
    def win_rate(self, returns: pd.Series) -> float:
        """胜率"""
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).sum() / len(returns))
    
    def profit_factor(self, returns: pd.Series) -> float:
        """盈亏比"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float("inf")
        return float(gains / losses)
    
    def format_report(self, metrics: Dict) -> str:
        """格式化绩效报告"""
        return f"""
========== 绩效报告 ==========
期间: {metrics['start_date']} ~ {metrics['end_date']}
交易日: {metrics['trading_days']}

【收益指标】
总收益率: {metrics['total_return']:.2%}
年化收益: {metrics['annual_return']:.2%}

【风险指标】
年化波动率: {metrics['volatility']:.2%}
最大回撤: {metrics['max_drawdown']:.2%}
最大回撤持续: {metrics['max_drawdown_duration']} 天

【风险调整收益】
夏普比率: {metrics['sharpe_ratio']:.2f}
索提诺比率: {metrics['sortino_ratio']:.2f}
卡玛比率: {metrics['calmar_ratio']:.2f}

【交易统计】
胜率: {metrics['win_rate']:.2%}
盈亏比: {metrics['profit_factor']:.2f}
================================
"""
