"""quantai.risk.volatility 测试。"""

from __future__ import annotations

import pandas as pd

from quantai.risk.volatility import VolatilityManager


def test_realized_volatility_non_negative(prices: pd.DataFrame) -> None:
    vol = VolatilityManager().compute_realized_volatility(prices["close"]).dropna()
    assert (vol >= 0).all()


def test_position_scalar_is_clipped() -> None:
    vm = VolatilityManager(target_volatility=0.10)
    assert vm.get_position_scalar(0.0) == 1.0       # 非正 -> 1.0
    assert vm.get_position_scalar(0.001) == 1.5     # 极低波动 -> 上限 1.5
    assert vm.get_position_scalar(10.0) == 0.2      # 极高波动 -> 下限 0.2


def test_vol_regime_labels_present(prices: pd.DataFrame) -> None:
    out = VolatilityManager().compute_vol_regime(prices["close"])
    assert set(out["vol_regime"].unique()) <= {"normal", "expanding", "contracting", "extreme"}


def test_check_vol_alert_shape(prices: pd.DataFrame) -> None:
    out = VolatilityManager().check_vol_alert(prices["close"])
    assert set(out) == {"current_vol", "vol_regime", "position_scalar", "alerts"}
