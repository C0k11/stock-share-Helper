"""组合级风险预警：回撤 / 波动率 / 集中度 / 相关性 / 风险状态切换。

从旧 `src/risk/alerts.py` 迁移，口径不变，加类型标注、显式 `fill_method=None`。
全部基于 trailing 窗口与当前持仓，无 lookahead。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    NEWS_EVENT = "news_event"
    REGIME_CHANGE = "regime_change"


@dataclass
class RiskAlert:
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RiskAlerts:
    """汇总各类组合风险预警。"""

    def __init__(self) -> None:
        self.alerts: list[RiskAlert] = []

    def check_all(
        self,
        portfolio_value: pd.Series,
        positions: dict[str, float],
        price_data: dict[str, pd.DataFrame],
        regime: str,
        prev_regime: Optional[str] = None,
    ) -> list[RiskAlert]:
        alerts: list[RiskAlert] = []
        alerts.extend(self._check_drawdown(portfolio_value))
        alerts.extend(self._check_volatility(portfolio_value))
        alerts.extend(self._check_concentration(positions))
        alerts.extend(self._check_correlation(price_data, positions))
        if prev_regime and regime != prev_regime:
            alerts.append(self._regime_change_alert(prev_regime, regime))
        self.alerts = alerts
        return alerts

    def _check_drawdown(self, portfolio_value: pd.Series) -> list[RiskAlert]:
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        current_dd = drawdown.iloc[-1]
        thresholds = [
            (-0.15, AlertSeverity.CRITICAL, "超过15%"),
            (-0.10, AlertSeverity.HIGH, "超过10%"),
            (-0.05, AlertSeverity.MEDIUM, "接近预警线"),
        ]
        for limit, severity, tag in thresholds:
            if current_dd < limit:
                return [
                    RiskAlert(
                        alert_type=AlertType.DRAWDOWN,
                        severity=severity,
                        message=f"组合回撤{abs(current_dd):.1%}，{tag}",
                        details={"drawdown": float(current_dd)},
                    )
                ]
        return []

    def _check_volatility(self, portfolio_value: pd.Series) -> list[RiskAlert]:
        returns = portfolio_value.pct_change(fill_method=None)
        current_vol = (returns.rolling(20).std() * np.sqrt(252)).iloc[-1]
        if current_vol > 0.25:
            return [
                RiskAlert(
                    AlertType.VOLATILITY,
                    AlertSeverity.HIGH,
                    f"组合波动率{current_vol:.1%}，处于极端水平",
                    {"volatility": float(current_vol)},
                )
            ]
        if current_vol > 0.15:
            return [
                RiskAlert(
                    AlertType.VOLATILITY,
                    AlertSeverity.MEDIUM,
                    f"组合波动率{current_vol:.1%}，高于正常水平",
                    {"volatility": float(current_vol)},
                )
            ]
        return []

    def _check_concentration(self, positions: dict[str, float]) -> list[RiskAlert]:
        if not positions:
            return []
        max_symbol = max(positions, key=positions.get)
        max_position = positions[max_symbol]
        if max_position > 0.5:
            return [
                RiskAlert(
                    AlertType.CONCENTRATION,
                    AlertSeverity.MEDIUM,
                    f"{max_symbol}仓位{max_position:.1%}，集中度较高",
                    {"symbol": max_symbol, "weight": max_position},
                )
            ]
        return []

    def _check_correlation(
        self, price_data: dict[str, pd.DataFrame], positions: dict[str, float]
    ) -> list[RiskAlert]:
        if len(price_data) < 2:
            return []
        returns = pd.DataFrame(
            {sym: df["close"].pct_change(fill_method=None) for sym, df in price_data.items()}
        ).dropna()
        if len(returns) < 60:
            return []
        corr_matrix = returns.iloc[-60:].corr()
        alerts: list[RiskAlert] = []
        cols = list(corr_matrix.columns)
        for i, sym1 in enumerate(cols):
            for sym2 in cols[i + 1 :]:
                corr = corr_matrix.loc[sym1, sym2]
                if corr > 0.8 and positions.get(sym1, 0) > 0.1 and positions.get(sym2, 0) > 0.1:
                    alerts.append(
                        RiskAlert(
                            AlertType.CORRELATION,
                            AlertSeverity.LOW,
                            f"{sym1}与{sym2}相关性{corr:.2f}，分散效果有限",
                            {"symbols": [sym1, sym2], "correlation": float(corr)},
                        )
                    )
        return alerts

    def _regime_change_alert(self, prev: str, current: str) -> RiskAlert:
        severity = {
            "risk_off": AlertSeverity.HIGH,
            "transition": AlertSeverity.MEDIUM,
        }.get(current, AlertSeverity.LOW)
        return RiskAlert(
            AlertType.REGIME_CHANGE,
            severity,
            f"风险状态从{prev}转为{current}",
            {"prev_regime": prev, "current_regime": current},
        )

    def get_summary(self) -> dict:
        return {
            "total_alerts": len(self.alerts),
            "critical": sum(a.severity == AlertSeverity.CRITICAL for a in self.alerts),
            "high": sum(a.severity == AlertSeverity.HIGH for a in self.alerts),
            "medium": sum(a.severity == AlertSeverity.MEDIUM for a in self.alerts),
            "low": sum(a.severity == AlertSeverity.LOW for a in self.alerts),
            "alerts": [
                {"type": a.alert_type.value, "severity": a.severity.value, "message": a.message}
                for a in self.alerts
            ],
        }
