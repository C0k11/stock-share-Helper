"""quantai.risk.drawdown 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.risk.drawdown import DrawdownController


def _equity_with_drawdown() -> pd.Series:
    idx = pd.bdate_range("2021-01-01", periods=10)
    # 涨到 120 再跌到 90（相对峰值回撤 25%）再创新高 125（完成恢复）
    vals = [100, 110, 120, 110, 100, 95, 90, 95, 100, 125]
    return pd.Series(vals, index=idx, dtype=float)


def test_drawdown_is_non_positive() -> None:
    dc = DrawdownController()
    dd = dc.compute_drawdown(_equity_with_drawdown())
    assert (dd["drawdown"] <= 1e-9).all()
    assert dd["drawdown"].min() < 0


def test_max_historical_drawdown_value() -> None:
    dc = DrawdownController()
    out = dc.get_current_drawdown(_equity_with_drawdown())
    # 峰值 120，谷值 90 -> -25%
    assert out["max_historical_drawdown"] == np.float64(-0.25)


def test_check_risk_level_thresholds() -> None:
    dc = DrawdownController(max_drawdown=0.10, warning_threshold=0.05, halt_threshold=0.25)
    assert dc.check_risk_level(-0.30)["level"] == "halt"
    assert dc.check_risk_level(-0.12)["level"] == "danger"
    assert dc.check_risk_level(-0.06)["level"] == "warning"
    assert dc.check_risk_level(-0.01)["level"] == "normal"


def test_max_drawdown_period_dates() -> None:
    dc = DrawdownController()
    out = dc.compute_max_drawdown_period(_equity_with_drawdown())
    assert out["max_drawdown"] == np.float64(-0.25)
    assert out["recovery_date"] is not None
