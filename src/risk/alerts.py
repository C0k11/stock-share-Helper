"""
风险预警模块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from loguru import logger


class AlertSeverity(Enum):
    """预警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """预警类型"""
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    NEWS_EVENT = "news_event"
    REGIME_CHANGE = "regime_change"


@dataclass
class RiskAlert:
    """风险预警"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RiskAlerts:
    """风险预警管理器"""
    
    def __init__(self):
        self.alerts: List[RiskAlert] = []
    
    def check_all(
        self,
        portfolio_value: pd.Series,
        positions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        regime: str,
        prev_regime: Optional[str] = None
    ) -> List[RiskAlert]:
        """
        检查所有风险预警
        
        Args:
            portfolio_value: 组合净值序列
            positions: 当前持仓 {symbol: weight}
            price_data: 价格数据 {symbol: DataFrame}
            regime: 当前风险状态
            prev_regime: 上一期风险状态
        """
        alerts = []
        
        # 1. 回撤预警
        alerts.extend(self._check_drawdown(portfolio_value))
        
        # 2. 波动率预警
        alerts.extend(self._check_volatility(portfolio_value))
        
        # 3. 集中度预警
        alerts.extend(self._check_concentration(positions))
        
        # 4. 相关性预警
        alerts.extend(self._check_correlation(price_data, positions))
        
        # 5. 风险状态变化预警
        if prev_regime and regime != prev_regime:
            alerts.append(self._regime_change_alert(prev_regime, regime))
        
        self.alerts = alerts
        return alerts
    
    def _check_drawdown(self, portfolio_value: pd.Series) -> List[RiskAlert]:
        """检查回撤预警"""
        alerts = []
        
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        current_dd = drawdown.iloc[-1]
        
        if current_dd < -0.15:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                severity=AlertSeverity.CRITICAL,
                message=f"组合回撤{abs(current_dd):.1%}，超过15%",
                details={"drawdown": float(current_dd)}
            ))
        elif current_dd < -0.10:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                severity=AlertSeverity.HIGH,
                message=f"组合回撤{abs(current_dd):.1%}，超过10%",
                details={"drawdown": float(current_dd)}
            ))
        elif current_dd < -0.05:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                severity=AlertSeverity.MEDIUM,
                message=f"组合回撤{abs(current_dd):.1%}，接近预警线",
                details={"drawdown": float(current_dd)}
            ))
        
        return alerts
    
    def _check_volatility(self, portfolio_value: pd.Series) -> List[RiskAlert]:
        """检查波动率预警"""
        alerts = []
        
        returns = portfolio_value.pct_change()
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        current_vol = vol_20d.iloc[-1]
        
        if current_vol > 0.25:
            alerts.append(RiskAlert(
                alert_type=AlertType.VOLATILITY,
                severity=AlertSeverity.HIGH,
                message=f"组合波动率{current_vol:.1%}，处于极端水平",
                details={"volatility": float(current_vol)}
            ))
        elif current_vol > 0.15:
            alerts.append(RiskAlert(
                alert_type=AlertType.VOLATILITY,
                severity=AlertSeverity.MEDIUM,
                message=f"组合波动率{current_vol:.1%}，高于正常水平",
                details={"volatility": float(current_vol)}
            ))
        
        return alerts
    
    def _check_concentration(self, positions: Dict[str, float]) -> List[RiskAlert]:
        """检查集中度预警"""
        alerts = []
        
        if not positions:
            return alerts
        
        max_position = max(positions.values())
        
        if max_position > 0.5:
            max_symbol = max(positions, key=positions.get)
            alerts.append(RiskAlert(
                alert_type=AlertType.CONCENTRATION,
                severity=AlertSeverity.MEDIUM,
                message=f"{max_symbol}仓位{max_position:.1%}，集中度较高",
                details={"symbol": max_symbol, "weight": max_position}
            ))
        
        return alerts
    
    def _check_correlation(
        self,
        price_data: Dict[str, pd.DataFrame],
        positions: Dict[str, float]
    ) -> List[RiskAlert]:
        """检查相关性预警"""
        alerts = []
        
        if len(price_data) < 2:
            return alerts
        
        # 计算收益率相关性
        returns = pd.DataFrame({
            symbol: df["close"].pct_change()
            for symbol, df in price_data.items()
        }).dropna()
        
        if len(returns) < 60:
            return alerts
        
        corr_matrix = returns.iloc[-60:].corr()
        
        # 检查高相关性
        for i, sym1 in enumerate(corr_matrix.columns):
            for j, sym2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr = corr_matrix.loc[sym1, sym2]
                    if corr > 0.8 and positions.get(sym1, 0) > 0.1 and positions.get(sym2, 0) > 0.1:
                        alerts.append(RiskAlert(
                            alert_type=AlertType.CORRELATION,
                            severity=AlertSeverity.LOW,
                            message=f"{sym1}与{sym2}相关性{corr:.2f}，分散效果有限",
                            details={"symbols": [sym1, sym2], "correlation": float(corr)}
                        ))
        
        return alerts
    
    def _regime_change_alert(self, prev: str, current: str) -> RiskAlert:
        """风险状态变化预警"""
        if current == "risk_off":
            severity = AlertSeverity.HIGH
        elif current == "transition":
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        return RiskAlert(
            alert_type=AlertType.REGIME_CHANGE,
            severity=severity,
            message=f"风险状态从{prev}转为{current}",
            details={"prev_regime": prev, "current_regime": current}
        )
    
    def get_summary(self) -> Dict:
        """获取预警摘要"""
        return {
            "total_alerts": len(self.alerts),
            "critical": len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]),
            "high": len([a for a in self.alerts if a.severity == AlertSeverity.HIGH]),
            "medium": len([a for a in self.alerts if a.severity == AlertSeverity.MEDIUM]),
            "low": len([a for a in self.alerts if a.severity == AlertSeverity.LOW]),
            "alerts": [
                {
                    "type": a.alert_type.value,
                    "severity": a.severity.value,
                    "message": a.message
                }
                for a in self.alerts
            ]
        }
