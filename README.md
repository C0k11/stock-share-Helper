# QuantAI - Intelligent Quantitative Investment Assistant

> A multimodal AI-powered personal securities investment assistant providing ETF portfolio strategy recommendations, risk alerts, and explainable decisions.

[中文版](#quantai---智能量化投顾助手) | [Engineering Log](docs/engineering_log_phase2_cn_infusion.md)

---

## Milestone Overview

| Phase | Topic | Status | Description |
| :--- | :--- | :--- | :--- |
| 1 | Bulletproof JSON Pipeline | ✅ Done | Strict JSON repair + schema validation.
| 2 | Teacher Data Generation (CN/US) | ✅ Done | Multi-market teacher dataset generation.
| 3 | LLM Fine-tuning (News LoRA) | ✅ Done | News LoRA training + inference.
| 4 | Production Pipeline (Daily Automation) | ✅ Done | Daily pipeline automation + health/fallback.
| 5 | ETF Trader + RAG + RiskGate | ✅ Done | RAG retrieval + deterministic risk gate.
| 6 | Stock Trader v1.1 (Tech + News) + Dual Tower | ✅ Done | News Tower + Trader Tower integration.
| 7 | Backtest & Execution | ✅ Done | NAV curve backtest + `Hold=Keep` + `Confirm=2` execution filter.
| 8 | Paper Trading | ✅ Done | Rolling daily simulation with state persistence + RiskGate CLEAR.
| 9 | Dashboard | ✅ Done | Streamlit cockpit for NAV, orders, and risk monitoring.
| 10 | RL (Reinforcement Learning) | Future | Only after longer-horizon backtest is stable.

---

##  Project Overview

**Not a black-box promising profits, but an intelligent assistant providing explainable recommendations under risk constraints.**

Core Capabilities:
- **Strategy Recommendations**: Daily target positions, entry/exit conditions, stop-loss/take-profit
- **Risk Alerts**: Volatility spikes, drawdown warnings, correlation surge detection
- **News Understanding**: LLM-powered news structuring, explaining "why reduce position"
- **Historical Analysis**: Analyze historical trends + news to identify key market drivers
- **Personalization**: Adjust positions based on user risk profiles (Conservative/Balanced/Aggressive)

---

##  MVP Definition (v1)

| Dimension | Decision |
|-----------|----------|
| **Market** | US ETFs first, HK ETFs in Phase 2 |
| **Symbol Pool** | TLT, IEF, GLD, SPY, QQQ, SHY |
| **Frequency** | Daily (post-market), intraday as Phase 2 |
| **Output** | Strategy recommendations + simulated portfolio curve |
| **Risk Profiles** | Conservative(5% DD) / Balanced(10%) / Aggressive(20%) |
| **Tech Stack** | Python + Local LLM fine-tuning (RTX 4090) |
| **Data Sources** | Free sources first (yfinance/Stooq/NewsAPI) |

---

##  System Architecture

![System Architecture](System%20Architecture.png)

---

##  Project Structure

```
Stock/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore
├── run_pipeline.bat          # One-click daily pipeline (Windows)
├── config/                   # Configuration files
│   ├── settings.yaml         # Global settings
│   ├── symbols.yaml          # Symbol pool config
│   ├── risk_profiles.yaml    # Risk profile config
│   └── sources.yaml          # News sources (RSS/API) config
├── data/                     # Data directory (not in git)
│   ├── daily/                # Daily outputs (news/signals/reports/features)
│   ├── raw/                  # Raw price data (parquet)
│   └── finetune/             # Fine-tuning datasets
├── src/                      # Source code
│   ├── data/                 # Data layer (fetcher/calendar/rag)
│   ├── features/             # Features (technical/regime/news)
│   ├── strategy/             # Strategy (signals/position/rules)
│   ├── risk/                 # Risk management (drawdown/vol/gate)
│   ├── backtest/             # Backtesting (engine/costs/metrics)
│   ├── llm/                  # LLM (news_parser/explainer/finetune)
│   ├── utils/                # Utilities (llm_tools/json repair)
│   └── api/                  # FastAPI interface
├── scripts/                  # Scripts
│   ├── fetch_daily_rss.py            # Fetch news (RSS + API fallback)
│   ├── build_daily_etf_features.py   # Generate ETF feature snapshots
│   ├── run_daily_inference.py        # News structuring (Qwen + LoRA)
│   ├── run_trading_inference.py      # Trading decisions (RAG + RiskGate)
│   ├── generate_daily_report.py      # Generate Markdown report
│   ├── generate_etf_teacher_dataset.py # Teacher distillation (DeepSeek)
│   ├── process_rag_data.py           # Process training data with denoising
│   ├── finetune_llm.py               # LoRA/QLoRA fine-tuning
│   ├── evaluate_signal.py            # Signal evaluation (T+1 alignment)
│   ├── backtest_engine.py            # Backtesting engine
│   └── dashboard.py                  # Streamlit visualization
├── models/                   # Model files (not in git)
├── docs/                     # Documentation
│   └── engineering_log_phase2_cn_infusion.md  # Engineering log
└── tests/                    # Unit tests
```

---

##  Roadmap

### Phase 1: Data & Backtest Foundation 
- [x] Data fetcher module (yfinance)
- [x] Trading calendar & timezone handling
- [x] Technical factors (MA/Momentum/Volatility)
- [x] Backtest engine (costs/slippage/rebalancing)
- [x] Performance metrics

### Phase 2: Strategy MVP 
- [x] Regime detection (Risk-On/Off/Transition)
- [x] Volatility-targeted position sizing
- [x] Trend/Momentum signals
- [x] Drawdown protection rules
- [ ] Walk-forward validation

### Phase 3: LLM Fine-tuning 
- [x] News data collection (multi-source RSS)
- [x] Fine-tuning dataset construction (append/dedup/confidence + train/val split)
- [x] LoRA/QLoRA fine-tuning (Qwen2.5-14B supported)
- [x] News structuring inference (base vs LoRA)
- [x] Decision explanation generation (base vs LoRA)

### Phase 4: Production Pipeline 
- [x] Daily news pipeline (fetch → infer → report)
- [x] Signal evaluation (T+1 alignment, event-type analysis)
- [x] Health monitoring & fallback (CN RSS → JSON API)
- [x] Windows Task Scheduler integration

### Phase 5: ETF Trader + RAG + RiskGate (Done (ETF line))
- [x] ETF feature snapshot (`build_daily_etf_features.py`)
- [x] Teacher distillation dataset (DeepSeek, 25k samples)
- [x] RAG retrieval (FAISS-based similar history)
- [x] RiskGate deterministic constraints
- [x] Trading inference pipeline (`run_trading_inference.py`)
- [x] CN concept_hype post-filter & denoising
- [x] Model D training (Qwen2.5-7B)
- [x] Model D backtest report (see engineering log)

### Phase 6: Stock Trader v1.1 (Tech + News) (Done)
- [x] News Tower 3B LoRA (noise killer) trained and validated
- [x] Historical signals backfill (`signals_YYYY-MM-DD.json`)
- [x] Training-set news injection for Stock SFT
- [x] Stock Trader v1.1 (Qwen2.5-7B LoRA) trained
- [x] Inference-side stock news injection (A/B controllable)
- [x] Final Exam (news-conditioning / ablation) validated

### Phase 7: Simulation / Backtest (Done)
- [x] 2025-12 stock feature backfill (daily `stock_features_YYYY-MM-DD.json`)
- [x] Stock signal-quality backtest (LoRA adapter swap on a single base model)
- [x] Upgrade to NAV curve backtest (positions/costs/drawdown)
- [ ] Re-run on horizon=T+5 and stratify by strong-news vs quiet days (requires later data coverage)

### Phase 8: Multi-Market Expansion / RL (Future)
- [ ] A-share support (CN_Trader LoRA)
- [ ] Canadian stocks (CA ETFs)
- [ ] Gold/Commodities (Macro_Gold LoRA)
- [ ] Funnel filtering (Python → 3B → 7B)
- [ ] RL/DPO (only after backtest curve is stable)

---

##  Output Protocol

Daily output for each ETF:

```json
{
  "date": "2024-01-15",
  "symbol": "SPY",
  "recommendation": {
    "action": "reduce",
    "target_position": 0.3,
    "stop_loss": "$465",
    "exit_condition": "Execute if price drops below $470"
  },
  "risk_alerts": [
    {"type": "volatility_up", "severity": "medium", "message": "VIX rose to 22"}
  ],
  "explanation": {
    "technical": "SPY broke below 20-day MA",
    "regime": "Risk state shifted to Transition",
    "news": "Fed minutes suggest rates stay higher for longer"
  },
  "confidence": 0.72
}
```

---

##  Risk Profile Configuration

| Profile | Max Drawdown | Target Vol | Max Single Position | Min Cash |
|---------|--------------|------------|---------------------|----------|
| Conservative | 5% | 6% | 25% | 30% |
| Balanced | 10% | 10% | 35% | 15% |
| Aggressive | 20% | 15% | 50% | 5% |

---

##  Setup

### Hardware Requirements
- **Minimum**: 8GB RAM, no GPU
- **Recommended**: 16GB RAM, 8GB VRAM
- **Current**: RTX 4090 + AMD 7950X3D (LLM fine-tuning ready)

### Installation

```bash
git clone https://github.com/C0k11/stock-share-Helper.git
cd stock-share-Helper

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_backtest.py
```

---

## Daily News Pipeline (US + CN)

This repo includes a production-oriented daily pipeline:

- **Fetch** news (US RSS + CN RSS with JSON fallback)
- **Infer** structured signals with Qwen2.5 + LoRA
- **Generate** a Markdown daily report
- **Evaluate** signals vs market ground truth (optional)

### One-click (Windows)

Run:

```powershell
run_pipeline.bat
```

Outputs (under `data/daily/`):

- `news_YYYY-MM-DD.json`
- `signals_YYYY-MM-DD.json`
- `report_YYYY-MM-DD.md`
- `health_YYYY-MM-DD.json` (fetch health report)
- `etf_features_YYYY-MM-DD.json` (daily ETF/index feature snapshot)

The daily report also includes a **Risk Watch** section for CN `regulation_crackdown` signals.

### Run scripts manually

```powershell
.\venv311\Scripts\python.exe scripts\fetch_daily_rss.py --date 2025-12-14 --health-out auto
.\venv311\Scripts\python.exe scripts\build_daily_etf_features.py --date 2025-12-14
.\venv311\Scripts\python.exe scripts\run_daily_inference.py --date 2025-12-14 --use-lora --load-in-4bit --batch-size 4 --max-input-chars 6000
.\venv311\Scripts\python.exe scripts\generate_daily_report.py --date 2025-12-14
```

---

## 📊 Visualization (Dashboard)
Monitor your paper trading status, NAV curve, orders flow, and risk events using the Streamlit cockpit.

1. **Install Dependencies**:

```bash
pip install streamlit pandas plotly
```

2. **Launch Dashboard**:

```bash
streamlit run scripts/dashboard.py
```

3. **Select Data Directory**:

In the sidebar, set `Paper Dir` to:

- `data/paper_rolltest` (recommended, includes the rolling drill: Hold / ForceFlat / Pending / Flip)
- `data/paper` (your current live paper trading directory)

### Evaluation (T+1 alignment)

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --signals data/daily/signals_2025-12-14.json --date 2025-12-14 --align-mode run_date --sample 20
```

Multi-day aggregated evaluation (scan `data/daily/signals_*.json`):

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --scan-daily --daily-dir data/daily --start-date 2025-12-01 --end-date 2025-12-31 --align-mode run_date --auto-fetch
```

Event-based analysis is printed automatically (by `event_type`). You can also filter the evaluation to specific event types:

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --signals data/daily/signals_2025-12-14.json --date 2025-12-14 --align-mode run_date --types regulation_crackdown
```

### Scheduling (Windows Task Scheduler)

To accumulate history automatically, schedule `run_pipeline.bat` daily.

- Create a basic task in Task Scheduler, set **Program/script** to the absolute path of `run_pipeline.bat`.
- Set **Start in** to the repo root folder.

Optional CLI example (run PowerShell as Administrator, adjust paths):

```powershell
schtasks /Create /TN "QuantAI_DailyPipeline" /SC DAILY /ST 07:30 /RL HIGHEST /F /TR "\"D:\\Project\\Stock\\run_pipeline.bat\""
```

---

##  LLM Fine-tuning (Quickstart)

This repo includes a practical pipeline for building a weak-labeled dataset from public RSS sources and fine-tuning Qwen2.5 models with LoRA/QLoRA.

### 1) Configure news sources

Edit `config/sources.yaml` to enable/disable sources and adjust weights/categories.

### 2) Build finetune dataset (append + dedup + quality filters)

Outputs are written under `data/` (ignored by git). Recommended command:

```bash
.\venv311\Scripts\python.exe scripts\build_finetune_dataset.py --limit 800 --add-explain --append --dedup --split-val --val-ratio 0.05
```

This will generate:

- `data/finetune/train.json`
- `data/finetune/val.json`

### 3) Train (Recommended: 3B/7B QLoRA)

Recommended local setup is Qwen2.5-3B (News) + Qwen2.5-7B (Trader) under QLoRA 4-bit.

Note: 14B experiments are treated as legacy and can be archived (see engineering log).

```bash
.\venv311\Scripts\python.exe scripts\finetune_llm.py --model Qwen/Qwen2.5-3B-Instruct --data data/finetune/train.json --qlora --grad-ckpt --max-seq-len 1024 --batch-size 1 --grad-acc 16 --lr 1e-4 --epochs 3 --save-steps 50 --save-total-limit 5 --outdir models/news_final_3b_v1
```

### 4) Inference (base vs LoRA)

For 3B/7B + LoRA inference, 4-bit loading is recommended to avoid CPU/disk offloading issues:

```bash
.\venv311\Scripts\python.exe scripts\infer_llm.py --model Qwen/Qwen2.5-3B-Instruct --task news --load-in-4bit
.\venv311\Scripts\python.exe scripts\infer_llm.py --model Qwen/Qwen2.5-3B-Instruct --task news --use-lora --lora models\news_final_3b_v1\lora_weights --load-in-4bit
```

---

## Stock Trader (Phase 6/7)

This repo includes a Stock Trader pipeline with optional news-context injection.

- Feature generation: `scripts/build_daily_features_universal.py` -> `data/daily/stock_features_YYYY-MM-DD.json`
- News backfill: `scripts/backfill_news_signals.py` -> `data/daily/signals_YYYY-MM-DD.json`
- Stock backtest: `scripts/backtest_trader.py` -> `data/backtest/report_*.json`

### 5) Teacher distillation dataset (DeepSeek, ETF)

This repo supports generating a high-quality teacher dataset from daily ETF feature snapshots using an OpenAI-compatible API (e.g., DeepSeek).

Dual-model architecture (Staff Officer vs. Field Commander):

- News LoRA (intelligence staff): turns noisy news into structured `signals_*.json` (a daily intel brief).
- Trading model/LoRA (field commander): reads both the intel brief (`signals_*.json` / `risk_watch`) and the battlefield map (`etf_features_*.json`), then outputs the final action + target position.

Future architecture (planned):

- Evidence/RAG layer: retrieve original news snippets, macro items, and similar historical cases to reduce hallucinations.
- Risk gate (deterministic): hard constraints on max position, drawdown, leverage, regime overrides; treat trading LoRA output as a proposal.
- Execution planner: convert target position into orders (rebalance threshold, slicing, constraints).
- Evaluation & monitoring: backtest, A/B, drift detection, daily QA dashboards.
- Multi-adapter management: keep one base model with multiple LoRA adapters (news/trading/etc.), route by task.

Environment variables:

- `TEACHER_API_KEY`
- `TEACHER_BASE_URL` (e.g. `https://api.deepseek.com`)
- `TEACHER_MODEL` (e.g. `deepseek-chat`)

Example:

```bash
.\venv311\Scripts\python.exe scripts\generate_etf_teacher_dataset.py --daily-dir data/daily --start-date 2025-12-01 --end-date 2025-12-31 --include-cot --resume
```

---

##  Disclaimer

 **For educational and research purposes only. Not investment advice.**

- Past performance does not guarantee future results
- All recommendations are for reference only
- Users bear all responsibility for investment decisions

---

##  License

MIT License

---
---

# QuantAI - 智能量化投顾助手

> 基于多模态AI的个人证券投资助手，提供ETF组合策略建议、风险预警与可解释决策。

[English Version](#quantai---intelligent-quantitative-investment-assistant) | [工程日志](docs/engineering_log_phase2_cn_infusion.md)

---

## 里程碑总览

| Phase | 主题 | 状态 | 说明 |
| :--- | :--- | :--- | :--- |
| 1 | JSON Pipeline 稳定性 | ✅ 完成 | JSON 强修复 + schema 校验。
| 2 | Teacher 数据生成 (CN/US) | ✅ 完成 | 多市场 teacher 数据产出。
| 3 | LLM 微调 (News LoRA) | ✅ 完成 | News LoRA 训练与推理。
| 4 | 生产流水线 (日更自动化) | ✅ 完成 | 日更自动化 + 兜底/健康检查。
| 5 | ETF Trader + RAG + RiskGate | ✅ 完成 | RAG 检索 + 风控闸门。
| 6 | Stock Trader v1.1（Tech + News）+ 双塔接线 | ✅ 完成 | News/Trader 双塔接线。
| 7 | 回测与执行层调优 | ✅ 完成 | NAV 回测 + `Hold=Keep` + `Confirm=2` 防抖。
| 8 | 模拟盘（Paper Trading） | ✅ 完成 | 滚动模拟 + 状态持久化 + RiskGate CLEAR。
| 9 | 监控看板（Dashboard） | ✅ 完成 | Streamlit 驾驶舱：NAV/订单/风险。
| 10 | RL 强化学习 | 远期 | 仅在更长窗口回测稳定后考虑。

---

##  项目定位

**不是"保证赚钱"的黑箱，而是"风险约束下可解释建议"的智能助手。**

核心能力：
- **策略建议**：每日输出目标仓位、入场/离场条件、止损止盈
- **风险预警**：波动上升、回撤预警、相关性飙升检测
- **新闻理解**：LLM结构化新闻事件，解释"为什么建议减仓"
- **历史分析**：分析历史走势+新闻，识别影响股市的关键因素
- **个性化**：根据用户风险偏好（保守/平衡/进取）调整仓位

---

##  MVP定义（第1版）

| 维度 | 决定 |
|------|------|
| **市场** | 美股ETF优先，港股ETF第二阶段 |
| **标的池** | TLT, IEF, GLD, SPY, QQQ, SHY |
| **频率** | 日频（收盘后生成建议），盘中作为二期增强 |
| **输出** | 策略建议/提醒 + 模拟组合收益曲线 |
| **风险档位** | 保守(5%回撤) / 平衡(10%) / 进取(20%) |
| **技术栈** | Python + 本地LLM微调(4090) |
| **数据源** | 免费源先跑通（yfinance/Stooq/NewsAPI） |

---

##  系统架构

![系统架构](System%20Architecture%20CN.png)

---

##  项目结构

```
Stock/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── .gitignore
├── run_pipeline.bat          # 一键运行每日流水线（Windows）
├── config/                   # 配置文件
│   ├── settings.yaml         # 全局配置
│   ├── symbols.yaml          # 标的池配置
│   ├── risk_profiles.yaml    # 风险档位配置
│   └── sources.yaml          # 新闻源（RSS/API）配置
├── data/                     # 数据目录（不提交到git）
│   ├── daily/                # 每日输出（新闻/信号/日报/特征）
│   ├── raw/                  # 原始价格数据（parquet）
│   └── finetune/             # 微调数据集
├── src/                      # 源代码
│   ├── data/                 # 数据层（fetcher/calendar/rag）
│   ├── features/             # 特征工程（technical/regime/news）
│   ├── strategy/             # 策略模块（signals/position/rules）
│   ├── risk/                 # 风控模块（drawdown/vol/gate）
│   ├── backtest/             # 回测模块（engine/costs/metrics）
│   ├── llm/                  # LLM模块（news_parser/explainer/finetune）
│   ├── utils/                # 工具模块（llm_tools/JSON修复）
│   └── api/                  # API接口
├── scripts/                  # 脚本
│   ├── fetch_daily_rss.py            # 抓取新闻（RSS + API兜底）
│   ├── build_daily_etf_features.py   # 生成ETF特征快照
│   ├── run_daily_inference.py        # 新闻结构化（Qwen + LoRA）
│   ├── run_trading_inference.py      # 交易决策（RAG + RiskGate）
│   ├── generate_daily_report.py      # 生成Markdown日报
│   ├── generate_etf_teacher_dataset.py # Teacher蒸馏（DeepSeek）
│   ├── process_rag_data.py           # 处理训练数据（含降噪）
│   ├── finetune_llm.py               # LoRA/QLoRA微调
│   ├── evaluate_signal.py            # 信号评测（T+1对齐）
│   ├── backtest_engine.py            # 回测引擎
│   └── dashboard.py                  # Streamlit可视化
├── models/                   # 模型文件（不提交到git）
├── docs/                     # 文档
│   └── engineering_log_phase2_cn_infusion.md  # 工程日志
└── tests/                    # 单元测试
```

---

##  迭代路线图

### Phase 1: 数据与回测基础 
- [x] 数据抓取模块（yfinance）
- [x] 交易日历与时区处理
- [x] 技术因子计算（MA/动量/波动率）
- [x] 回测引擎（成本/滑点/再平衡）
- [x] 基础绩效指标

### Phase 2: 策略MVP 
- [x] 风险状态检测（Regime: Risk-On/Off/Transition）
- [x] 目标波动仓位计算
- [x] 趋势/动量信号
- [x] 回撤保护规则
- [ ] Walk-forward验证

### Phase 3: LLM微调 
- [x] 新闻数据收集（多源RSS）
- [x] 微调数据集构建（append/dedup/confidence + train/val切分）
- [x] LoRA/QLoRA微调（支持Qwen2.5-14B）
- [x] 新闻结构化推理（base vs LoRA）
- [x] 决策解释生成（base vs LoRA）

### Phase 4: 生产流水线 
- [x] 每日新闻流水线（抓取 → 推理 → 日报）
- [x] 信号评测（T+1 对齐，按事件类型分析）
- [x] 健康监控与兜底（CN RSS → JSON API fallback）
- [x] Windows 任务计划程序集成

### Phase 5: ETF 交易模型 + RAG + RiskGate（完成：ETF line）
- [x] ETF 特征快照（`build_daily_etf_features.py`）
- [x] Teacher 蒸馏数据集（DeepSeek，25k 样本）
- [x] RAG 检索（FAISS 相似历史）
- [x] RiskGate 确定性约束
- [x] 交易推理流水线（`run_trading_inference.py`）
- [x] CN concept_hype 后处理与降噪
- [x] Model D 训练（Qwen2.5-7B）
- [x] Model D 回测报告（详见工程日志）

### Phase 6: Stock Trader v1.1（Tech + News）（完成）
- [x] News Tower 3B LoRA（noise killer）训练与验收
- [x] 历史 signals 回放与落盘（`signals_YYYY-MM-DD.json`）
- [x] Stock SFT 训练集注入 News Context
- [x] Stock Trader v1.1（Qwen2.5-7B LoRA）训练完成
- [x] 推理侧 stock 模式 news 注入（支持 A/B 对照）
- [x] Final Exam（news-conditioning / ablation）验收通过

### Phase 7: Simulation / Backtest（完成）
- [x] 2025-12 stock_features 批量补齐
- [x] Stock 回测（信号质量评估器，adapter 轮换）
- [x] 升级为 NAV 曲线回测（仓位/费用/回撤）
- [ ] 对齐 T+5 并按强新闻日/平静日分层评估（依赖后续数据覆盖）

### Phase 8: 全市场扩张 / RL（远期）
- [ ] A股支持（CN_Trader LoRA）
- [ ] 加股（CA ETFs）
- [ ] 黄金/大宗商品（Macro_Gold LoRA）
- [ ] 漏斗筛选（Python → 3B → 7B）
- [ ] RL/DPO（仅在回测曲线稳定后考虑）

---

##  输出协议

每日为每个ETF输出：

```json
{
  "date": "2024-01-15",
  "symbol": "SPY",
  "recommendation": {
    "action": "减仓",
    "target_position": 0.3,
    "stop_loss": "$465",
    "exit_condition": "价格跌破$470立即执行"
  },
  "risk_alerts": [
    {"type": "volatility_up", "severity": "medium", "message": "VIX升至22，波动上升"}
  ],
  "explanation": {
    "technical": "SPY跌破20日均线，短期趋势转弱",
    "regime": "风险状态从Risk-On转为Transition",
    "news": "美联储会议纪要显示可能维持高利率更长时间"
  },
  "confidence": 0.72
}
```

---

##  风险档位配置

| 档位 | 最大回撤容忍 | 目标年化波动 | 单标的上限 | 现金下限 |
|------|-------------|-------------|-----------|---------|
| 保守 | 5% | 6% | 25% | 30% |
| 平衡 | 10% | 10% | 35% | 15% |
| 进取 | 20% | 15% | 50% | 5% |

---

##  环境配置

### 硬件要求
- **最低**：8GB内存，无GPU
- **推荐**：16GB内存，8GB显存
- **当前配置**：RTX 4090 + AMD 7950X3D（支持LLM微调）

### 安装

```bash
git clone https://github.com/C0k11/stock-share-Helper.git
cd stock-share-Helper

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_backtest.py
```

---

## 每日新闻流水线（US + CN）

本仓库包含一套偏生产化的日更流水线：

- **抓取** 新闻（US RSS + CN RSS，并带 CN JSON fallback 兜底）
- **推理** 结构化 signals（Qwen2.5 + LoRA）
- **生成** Markdown 日报
- **评测** 信号（可选）

### 一键运行（Windows）

运行：

```powershell
run_pipeline.bat
```

产物（位于 `data/daily/`）：

- `news_YYYY-MM-DD.json`
- `signals_YYYY-MM-DD.json`
- `report_YYYY-MM-DD.md`
- `health_YYYY-MM-DD.json`（抓取健康检查）
- `etf_features_YYYY-MM-DD.json`（ETF/指数特征快照）

日报中会额外包含 **Risk Watch** 专栏（CN `regulation_crackdown`）。

### 手动运行脚本

```powershell
.\venv311\Scripts\python.exe scripts\fetch_daily_rss.py --date 2025-12-14 --health-out auto
.\venv311\Scripts\python.exe scripts\build_daily_etf_features.py --date 2025-12-14
.\venv311\Scripts\python.exe scripts\run_daily_inference.py --date 2025-12-14 --use-lora --load-in-4bit --batch-size 4 --max-input-chars 6000
.\venv311\Scripts\python.exe scripts\generate_daily_report.py --date 2025-12-14
```

### 评测（T+1 对齐）

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --signals data/daily/signals_2025-12-14.json --date 2025-12-14 --align-mode run_date --sample 20
```

多日聚合评测（扫描 `data/daily/signals_*.json`，包含 `signals_full_*.json`）：

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --scan-daily --daily-dir data/daily --start-date 2025-12-01 --end-date 2025-12-31 --align-mode run_date --auto-fetch
```

脚本会自动打印按 `event_type` 的分组统计；也支持用 `--types` 只评测特定类型（如风控专场）：

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --signals data/daily/signals_2025-12-14.json --date 2025-12-14 --align-mode run_date --types regulation_crackdown
```

### 定时运行（Windows 任务计划程序）

为了自动积累历史数据，建议用任务计划程序每天定时运行 `run_pipeline.bat`。

- Program/script：填写 `run_pipeline.bat` 的绝对路径
- Start in：仓库根目录

可选命令行示例（管理员权限运行 PowerShell，按需修改路径）：

```powershell
schtasks /Create /TN "QuantAI_DailyPipeline" /SC DAILY /ST 07:30 /RL HIGHEST /F /TR "\"D:\\Project\\Stock\\run_pipeline.bat\""
```

##  LLM微调（快速开始）

本仓库提供从公开RSS源构建弱标注数据集，并使用LoRA/QLoRA微调Qwen2.5模型的可用流水线。

### 1) 配置新闻源

编辑 `config/sources.yaml`，按需开启/关闭来源并调整权重/类别。

### 2) 构建微调数据集（追加+去重+质量过滤）

输出在 `data/`（默认不提交到git）：

```bash
.\venv311\Scripts\python.exe scripts\build_finetune_dataset.py --limit 800 --add-explain --append --dedup --split-val --val-ratio 0.05
```

会生成：

- `data/finetune/train.json`
- `data/finetune/val.json`

### 3) 训练（14B QLoRA + checkpoints + 断点续训）

单卡4090建议使用4bit QLoRA + gradient checkpointing：

```bash
.\venv311\Scripts\python.exe scripts\finetune_llm.py --model Qwen/Qwen2.5-14B-Instruct --data data/finetune/train.json --qlora --grad-ckpt --max-seq-len 1024 --batch-size 1 --grad-acc 16 --lr 1e-4 --epochs 10 --save-steps 20 --save-total-limit 10 --outdir models/llm_qwen14b_overnight --resume auto
```

### 4) 推理（base vs LoRA）

14B+LoRA 推理建议加 `--load-in-4bit`，避免CPU/磁盘offload导致失败：

```bash
.\venv311\Scripts\python.exe scripts\infer_llm.py --model Qwen/Qwen2.5-14B-Instruct --task news --load-in-4bit
.\venv311\Scripts\python.exe scripts\infer_llm.py --model Qwen/Qwen2.5-14B-Instruct --task news --use-lora --lora models\llm_qwen14b_overnight\lora_weights --load-in-4bit
```

### 5) Teacher 蒸馏数据集（DeepSeek，ETF）

支持从每日 ETF 特征快照生成高质量 teacher 数据集（OpenAI-compat API，例如 DeepSeek），用于后续 LoRA 蒸馏训练。

双模型架构（参谋长 vs 指挥官）：

- 新闻 LoRA（情报参谋）：把海量新闻噪音结构化成 `signals_*.json`（每日情报简报）。
- 交易模型/LoRA（现场指挥官）：同时读取情报简报（`signals_*.json` / `risk_watch`）与战场地图（`etf_features_*.json`），输出最终动作与目标仓位。

未来架构蓝图（planned）：

- 证据/检索层（RAG）：检索新闻原文片段、宏观数据、历史相似案例，降低幻觉。
- 风控裁决层（确定性）：硬约束最大仓位/回撤/杠杆/切换风格等，把交易 LoRA 产出当作“建议”再裁决。
- 执行规划层：把目标仓位转成下单计划（阈值再平衡、分批、交易限制）。
- 评估与监控：回测、A/B、漂移监控、每日质量看板。
- 多 Adapter 管理：尽量维持一个 base，多套 LoRA（新闻/交易等），按任务路由。

环境变量：

- `TEACHER_API_KEY`
- `TEACHER_BASE_URL`（例如 `https://api.deepseek.com`）
- `TEACHER_MODEL`（例如 `deepseek-chat`）

示例（多角色辩论 + 长推演，支持断点续跑）：

```bash
.\venv311\Scripts\python.exe scripts\generate_etf_teacher_dataset.py --daily-dir data/daily --start-date 2025-12-01 --end-date 2025-12-31 --include-cot --resume
```

---

##  免责声明

 **本项目仅供学习研究，不构成任何投资建议。**

- 历史业绩不代表未来表现
- 所有策略建议仅供参考，投资决策需自行判断
- 使用本系统产生的任何损失由用户自行承担

---


MIT License
