"""quantai.risk.alerts 测试。"""

from __future__ import annotations

import pandas as pd

from quantai.risk.alerts import AlertSeverity, AlertType, RiskAlerts


def _equity(drawdown_pct: float) -> pd.Series:
    idx = pd.bdate_range("2021-01-01", periods=5)
    peak = 100.0
    trough = peak * (1 - drawdown_pct)
    return pd.Series([100, 100, 100, 100, trough], index=idx, dtype=float)


def test_drawdown_alert_severity_critical() -> None:
    alerts = RiskAlerts()._check_drawdown(_equity(0.20))
    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.DRAWDOWN
    assert alerts[0].severity == AlertSeverity.CRITICAL


def test_concentration_alert_triggers() -> None:
    alerts = RiskAlerts()._check_concentration({"SPY": 0.6, "QQQ": 0.4})
    assert len(alerts) == 1
    assert alerts[0].details["symbol"] == "SPY"


def test_no_concentration_when_balanced() -> None:
    assert RiskAlerts()._check_concentration({"SPY": 0.3, "QQQ": 0.3}) == []


def test_regime_change_alert() -> None:
    alert = RiskAlerts()._regime_change_alert("risk_on", "risk_off")
    assert alert.alert_type == AlertType.REGIME_CHANGE
    assert alert.severity == AlertSeverity.HIGH


def test_summary_counts() -> None:
    ra = RiskAlerts()
    ra.alerts = ra._check_drawdown(_equity(0.20))
    summary = ra.get_summary()
    assert summary["total_alerts"] == 1
    assert summary["critical"] == 1
