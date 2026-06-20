"""类型化配置 schema（pydantic v2）。

设计目标：
- 把旧 `config/settings.yaml` 的全部字段变成**有类型、有校验、有默认值**的模型。
- `extra="forbid"`：YAML 里出现 schema 未定义的键 -> 加载时立刻报错（揪拼写错误）。
- US-only：`market.primary` 与 `data.price_source` 用 `Literal` 锁死，
  填 HK/CN 或非 yfinance 源会在加载时被拒绝。
- 修 lookahead：`backtest.fill_timing` 默认 `next_open`。

所有嵌套模型都给了默认值，因此 `AppConfig()` 可零参构造，YAML 只需写覆盖项。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """统一配置基类：禁止未知字段 + 赋值时校验。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# project
# --------------------------------------------------------------------------- #
class ProjectConfig(_Base):
    name: str = "QuantAI"
    version: str = "0.2.0"
    description: str = ""


# --------------------------------------------------------------------------- #
# market（只支持 US）
# --------------------------------------------------------------------------- #
class MarketConfig(_Base):
    primary: Literal["US"] = "US"
    timezone: str = "America/New_York"
    currency: str = "USD"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class StorageConfig(_Base):
    type: Literal["parquet", "sqlite", "csv"] = "parquet"
    path: str = "data/"


class CacheConfig(_Base):
    enabled: bool = True
    ttl_hours: int = Field(default=24, ge=0)


class DataConfig(_Base):
    # US-only：旧 akshare/CN 路径已移除，价格源锁定 yfinance。
    price_source: Literal["yfinance"] = "yfinance"
    news_source: str = "rss"
    news_feeds: list[str] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
class CostModel(_Base):
    commission_rate: float = Field(default=0.0005, ge=0)
    min_commission: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)


class RebalanceConfig(_Base):
    frequency: Literal["daily", "weekly", "monthly"] = "weekly"
    threshold: float = Field(default=0.05, ge=0, le=1)


class BacktestConfig(_Base):
    start_date: str = "2010-01-01"
    end_date: Optional[str] = None  # None = 到今天
    costs: CostModel = Field(default_factory=CostModel)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    initial_capital: float = Field(default=100_000.0, gt=0)
    # 修 lookahead：d 日用 ≤d 信息决策，成交价取 open[d+1]。
    # "close" 保留为旧（虚高）行为，仅用于显式复现对比。
    fill_timing: Literal["next_open", "close"] = "next_open"


# --------------------------------------------------------------------------- #
# strategy
# --------------------------------------------------------------------------- #
class TrendConfig(_Base):
    ma_short: int = Field(default=20, gt=0)
    ma_long: int = Field(default=200, gt=0)
    momentum_lookback: int = Field(default=63, gt=0)


class VolatilityConfig(_Base):
    target_annual: float = Field(default=0.10, ge=0)
    lookback_days: int = Field(default=20, gt=0)
    vol_ceiling: float = Field(default=0.25, ge=0)


class RegimeConfig(_Base):
    vix_threshold_high: float = 25.0
    vix_threshold_low: float = 15.0
    trend_ma: int = Field(default=50, gt=0)


class StrategyConfig(_Base):
    trend: TrendConfig = Field(default_factory=TrendConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)


# --------------------------------------------------------------------------- #
# risk
# --------------------------------------------------------------------------- #
class RiskConfig(_Base):
    max_drawdown_halt: float = Field(default=0.25, ge=0, le=1)
    max_single_position: float = Field(default=0.5, ge=0, le=1)
    min_cash: float = Field(default=0.05, ge=0, le=1)


# --------------------------------------------------------------------------- #
# llm
# --------------------------------------------------------------------------- #
class LLMInferenceConfig(_Base):
    device: str = "cuda"
    # 量化模式：4bit(省显存) / 8bit(速度质量平衡，旧默认) / fp16(全质量)。
    # 取代旧 local_chat.py 里 use_4bit/use_8bit 两个互斥布尔（4bit 优先）的脆弱写法。
    quantization: Literal["4bit", "8bit", "fp16"] = "8bit"
    temperature: float = Field(default=0.7, ge=0)
    max_new_tokens: int = Field(default=512, gt=0)
    # 0 = 不截断上下文；>0 时对输入做 truncation。
    max_context: int = Field(default=0, ge=0)
    # generate 的墙钟超时（秒），桌面端防卡死。
    gen_max_time_sec: float = Field(default=12.0, gt=0)
    # MoE 多 adapter 时推理后复位到的默认专家。
    default_adapter: str = "scalper"


class LLMFinetuneConfig(_Base):
    method: str = "lora"
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = Field(default=2e-4, gt=0)
    num_epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    max_seq_length: int = Field(default=2048, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    save_steps: int = Field(default=100, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
    gradient_checkpointing: bool = False
    # QLoRA：4-bit NF4 基座 + LoRA 适配器（省显存的微调）。
    load_in_4bit: bool = False


class LLMDPOConfig(_Base):
    """DPO（直接偏好优化）对齐训练参数。"""

    beta: float = Field(default=0.1, gt=0)
    # DPO 比 SFT 用更低的 LR 以稳住偏好对齐。
    learning_rate: float = Field(default=5e-6, gt=0)
    num_epochs: int = Field(default=1, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    max_prompt_length: int = Field(default=1024, gt=0)
    max_length: int = Field(default=2048, gt=0)
    save_steps: int = Field(default=50, gt=0)
    logging_steps: int = Field(default=10, gt=0)
    # reference_free：跳过参考模型（dry-run/冒烟用；正式对齐应为 False）。
    reference_free: bool = False


class LLMConfig(_Base):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # HuggingFace 模型/权重缓存目录（取代旧 local_chat.py 硬编码的 "D:/Project/ml_cache/models"）。
    cache_dir: str = "models/hf_cache"
    # MoE 适配器：专家名 -> LoRA 权重路径（如 {"scalper": "...", "analyst": "..."}）。
    adapters: dict[str, str] = Field(default_factory=dict)
    inference: LLMInferenceConfig = Field(default_factory=LLMInferenceConfig)
    finetune: LLMFinetuneConfig = Field(default_factory=LLMFinetuneConfig)
    dpo: LLMDPOConfig = Field(default_factory=LLMDPOConfig)


# --------------------------------------------------------------------------- #
# api / logging
# --------------------------------------------------------------------------- #
class APIConfig(_Base):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = True


class LoggingConfig(_Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    file: str = "logs/quantai.log"
    rotation: str = "10 MB"


# --------------------------------------------------------------------------- #
# 根配置
# --------------------------------------------------------------------------- #
class AppConfig(_Base):
    """QuantAI 全局配置根。`load_config()` 返回的就是它。"""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
