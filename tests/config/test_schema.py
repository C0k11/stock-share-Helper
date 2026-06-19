"""quantai.config.schema 的校验行为测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantai.config.schema import (
    AppConfig,
    APIConfig,
    BacktestConfig,
    DataConfig,
    MarketConfig,
    RiskConfig,
)


def test_appconfig_zero_arg_defaults() -> None:
    """AppConfig() 应能零参构造，且默认值符合预期。"""
    cfg = AppConfig()
    assert cfg.market.primary == "US"
    assert cfg.market.timezone == "America/New_York"
    assert cfg.backtest.fill_timing == "next_open"  # 默认修好的行为
    assert cfg.data.price_source == "yfinance"
    assert cfg.api.port == 8000
    assert cfg.llm.finetune.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_market_us_only_rejects_other_markets() -> None:
    """market.primary 锁死 US，填 HK/CN 报错。"""
    with pytest.raises(ValidationError):
        MarketConfig(primary="HK")
    with pytest.raises(ValidationError):
        MarketConfig(primary="CN")


def test_price_source_locked_to_yfinance() -> None:
    """US-only：价格源只接受 yfinance（旧 akshare 已移除）。"""
    with pytest.raises(ValidationError):
        DataConfig(price_source="akshare")


def test_unknown_key_is_forbidden() -> None:
    """extra='forbid'：未知键(拼写错误)在加载时就报错。"""
    with pytest.raises(ValidationError):
        MarketConfig(primaryy="US")  # 故意拼错


def test_numeric_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        APIConfig(port=70000)  # > 65535
    with pytest.raises(ValidationError):
        RiskConfig(min_cash=2.0)  # > 1
    with pytest.raises(ValidationError):
        BacktestConfig(initial_capital=0)  # 必须 > 0


def test_fill_timing_close_is_allowed_for_legacy_repro() -> None:
    """'close' 仍合法：用于显式复现旧(虚高)回测，但不是默认。"""
    bt = BacktestConfig(fill_timing="close")
    assert bt.fill_timing == "close"
    with pytest.raises(ValidationError):
        BacktestConfig(fill_timing="midnight")
