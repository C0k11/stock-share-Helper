"""quantai.config —— 唯一的类型化配置入口。

典型用法：
    from quantai.config import load_config
    cfg = load_config()                 # 读 configs/default.yaml(+.local+env)，校验
    print(cfg.backtest.fill_timing)     # "next_open"
"""

from .loader import DEFAULT_CONFIG_PATH, deep_merge, load_config, load_yaml
from .schema import (
    AppConfig,
    APIConfig,
    BacktestConfig,
    CostModel,
    DataConfig,
    LLMConfig,
    LoggingConfig,
    MarketConfig,
    RiskConfig,
    StrategyConfig,
)

__all__ = [
    "load_config",
    "load_yaml",
    "deep_merge",
    "DEFAULT_CONFIG_PATH",
    "AppConfig",
    "MarketConfig",
    "DataConfig",
    "BacktestConfig",
    "CostModel",
    "StrategyConfig",
    "RiskConfig",
    "LLMConfig",
    "APIConfig",
    "LoggingConfig",
]
