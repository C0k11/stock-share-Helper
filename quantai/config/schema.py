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
# agents（多 Agent 大脑，拆自 150KB strategy.py）
# --------------------------------------------------------------------------- #
class RouterConfig(_Base):
    """MoE 路由阈值（诚实命名 HeuristicRouter：是规则路由，不是学习到的门控网络）。"""

    # 年化波动 % >= 此值 -> 路由到 analyst（高波动用更稳的分析专家）。
    vol_threshold: float = Field(default=60.0, ge=0)
    # |news_score| >= 此值视为强新闻信号。
    news_threshold: float = Field(default=0.8, ge=0)
    # True: 任意一条新新闻即路由到 analyst/news；False: 用 news_threshold 判定。
    any_news: bool = True


class PlannerConfig(_Base):
    """Planner：日级市场状态评估（rule 规则 / sft 轻量 MLP）。"""

    policy: Literal["rule", "sft"] = "rule"
    sft_model_path: str = "models/planner_sft_v1.pt"
    # 规则阈值（年化波动 %、5 日收益 %）：高波动/大跌 -> cash_preservation/defensive。
    cash_vol_ann_pct: float = 120.0
    cash_ret_5d_pct: float = -10.0
    defensive_vol_ann_pct: float = 80.0
    defensive_ret_5d_pct: float = -5.0


class GatekeeperConfig(_Base):
    """Gatekeeper：RL 门控 MLP（Q 值阈值过滤交易）。"""

    model_path: str = "models/rl_gatekeeper_v2.pt"
    threshold: float = 0.0
    # 无 RL 模型时的兜底（旧版用 random.random()>0.3，已删）：
    # True  -> 无模型即拒绝（架构建议，最诚实）；
    # False -> 用确定性波动门（vol <= vol_trigger_ann_pct 才放行），无随机。
    require_model: bool = True
    vol_trigger_ann_pct: float = 120.0


class System2Config(_Base):
    """System2 辩论（Critic + Judge）：高风险决策的二次复核。"""

    enabled: bool = True
    # 只对 BUY 提案触发辩论（省推理预算）。
    buy_only: bool = True
    lenient: bool = False


class ChartistConfig(_Base):
    """Chartist overlay：VLM（Qwen2.5-VL）看 K 线图给方向分（默认关闭，重依赖）。

    打分逻辑常驻（纯逻辑）；VLM 后端（mplfinance 渲染 + 推理）懒加载，缺权重/缺 GPU
    自动降级为不评分（返回 0），不影响主管线。
    """

    enabled: bool = False
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    # VLM 模型名/路径（HF repo 或本地目录）。
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_4bit: bool = True
    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.2, ge=0)
    # processor 像素上限/下限（0 = 用模型默认）。
    min_image_pixels: int = Field(default=0, ge=0)
    max_image_pixels: int = Field(default=0, ge=0)
    # 可选 LoRA adapter 路径。
    adapter: str = ""
    # 渲染用的回看 bar 数。
    lookback: int = Field(default=60, gt=0)


class MacroConfig(_Base):
    """Macro Governor：基于 VIX / 10Y 收益率的**确定性**宏观风险闸。

    B-2：旧 `_macro_governor_assess` 是 `random.uniform(0.3,0.8)`（纯随机，已删）。
    现接 `PriceFetcher` 已在抓的 ^VIX / ^TNX，按阈值确定性映射 gear/label：
    VIX 高 -> RISK_OFF(gear=-1)；VIX 低且非 risk-off -> RISK_ON(gear=+1)；否则 NEUTRAL(0)。
    """

    enabled: bool = False
    # VIX >= 此值 -> RISK_OFF（缩风险）。
    vix_risk_off: float = Field(default=28.0, gt=0)
    # VIX <= 此值（且非 risk_off）-> RISK_ON。
    vix_risk_on: float = Field(default=15.0, gt=0)
    # 10Y 收益率(%) >= 此值 -> 也偏 RISK_OFF（利率冲击）。
    tnx_risk_off: float = Field(default=4.8, gt=0)


class AgentsConfig(_Base):
    """多 Agent 大脑配置。编排顺序：Planner -> Gatekeeper -> Router -> Expert
    -> Chartist -> Macro -> System2。"""

    router: RouterConfig = Field(default_factory=RouterConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    gatekeeper: GatekeeperConfig = Field(default_factory=GatekeeperConfig)
    system2: System2Config = Field(default_factory=System2Config)
    chartist: ChartistConfig = Field(default_factory=ChartistConfig)
    # Macro Governor：确定性 VIX/10Y 风险闸（B-2，已接真实信号；默认关闭）。
    macro: MacroConfig = Field(default_factory=MacroConfig)
    # all_agents_mode：同时跑 scalper+analyst 再合议（旧版委员会模式）。
    all_agents_mode: bool = False


# --------------------------------------------------------------------------- #
# evolution（数据飞轮：采集 → 建数据集 → 离线训练 → 热切换）
# --------------------------------------------------------------------------- #
class EvolutionConfig(_Base):
    """数据飞轮配置。

    **诚实命名**（B-1）：本层只做"采集 + 离线训练编排"。在线实时梯度更新**未实现**，
    `online_gradient_enabled` 永远 False（占位标注），真训练走离线 DPO（`scripts/train_dpo.py`）。
    路径全部 config 驱动，替换旧代码里的硬编码 `data/rl_experiences` 等。
    """

    # 落盘目录（config 驱动，替换硬编码路径）
    trajectories_dir: str = "data/evolution/trajectories"
    experiences_dir: str = "data/evolution/experiences"
    preferences_dir: str = "data/evolution/preferences"
    adapters_dir: str = "data/evolution/adapters"
    active_pointer: str = "data/evolution/active_adapters.json"

    # 采集器
    max_experiences: int = Field(default=100_000, gt=0)
    update_interval_trades: int = Field(default=50, gt=0)
    min_experiences: int = Field(default=100, ge=0)

    # 偏好对（DPO 数据集）
    min_pnl_diff: float = Field(default=100.0, ge=0)

    # 奖励整形
    reward_pnl_scale: float = Field(default=0.001, ge=0)
    reward_sharpe_weight: float = Field(default=0.3, ge=0)
    reward_drawdown_penalty: float = Field(default=-2.0, le=0)
    reward_max_drawdown_trigger: float = Field(default=-0.05, le=0)

    # 在线实时梯度：未实现（B-1 诚实占位）。保持 False。
    online_gradient_enabled: bool = False


# --------------------------------------------------------------------------- #
# portfolio（真实持仓分析）
# --------------------------------------------------------------------------- #
class PortfolioConfig(_Base):
    """真实持仓录入 + 组合分析配置。

    `file` 指向真实持仓（`*.local.yaml`，被 .gitignore 排除、永不入库）；
    仓库只带 `portfolio.example.yaml` 示例。
    """

    file: str = "portfolio.local.yaml"
    benchmark: str = "SPY"
    # 分析用的历史回看年数（beta/波动/回撤的合成历史长度）。
    history_years: int = Field(default=2, gt=0)


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
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
