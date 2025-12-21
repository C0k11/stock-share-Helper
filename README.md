# QuantAI - 智能量化投顾助手

> 基于多模态 AI 的个人证券投资助手，提供 ETF 组合策略建议、风险预警与可解释决策。

[English Version](#quantai---intelligent-quantitative-investment-assistant) | [工程日志](docs/engineering_log_phase2_cn_infusion.md)

---

## 里程碑总览

| Phase | 名称 | 状态 | 完成时间 |
| :--- | :--- | :--- | :--- |
| 1 | Bulletproof JSON Pipeline | 完成 | 2025-12 |
| 2 | Teacher 数据生成 (CN/US) | 完成 | 2025-12 |
| 3 | LLM 微调 (News LoRA) | 完成 | 2025-12 |
| 4 | 生产流水线 (日更自动化) | 完成 | 2025-12 |
| 5 | ETF Trader + RAG + RiskGate | 完成 | 2025-12-17 |
| 6 | Stock Trader v1.1（Tech + News）+ 双塔接线 | 完成 | 2025-12-18 |
| 7 | Simulation / Backtest（NAV Backtest + Execution Tuning） | 完成 | 2025-12-19 |
| 8 | Paper Trading Automation | 完成 | 2025-12-19 |
| 9 | Dashboard（Streamlit Cockpit） | 完成 | 2025-12-19 |
| 10 | CoT 蒸馏 / Reasoning 升级（Trader v2） | 进行中 | 2025-12 |
| 11 | Adapter-MoE / Multi-Agent | 进行中 | 2025-12 |
| 12 | DPO / GRPO | 完成 | 2025-12 |
| 13 | 黄金运行（严格风险 + 规划器 + DPO Analyst） | 完成 | 2025-12 |
| 14 | 评测平台（Protocol Freeze + Walk-forward + Stratified Report） | 进行中 | 2025-12 |

---

## 项目概述

**非“保证收益”的黑箱系统，而是在风险约束下提供可解释建议的智能投资助手。**

核心能力：
- **策略建议**：每日目标仓位、入场/离场条件、止损/止盈
- **风险预警**：波动率飙升、回撤预警、相关性激增检测
- **新闻理解**：基于 LLM 的新闻结构化处理，解释“为何建议减仓”
- **历史分析**：分析历史趋势与新闻，识别关键市场驱动因素
- **个性化**：根据用户风险偏好（保守/平衡/进取）调整仓位建议

---

## MVP 定义（v1）

| 维度 | 决定 |
|------|------|
| **市场** | 美股 ETF 优先，港股 ETF 第二阶段 |
| **标的池** | TLT, IEF, GLD, SPY, QQQ, SHY |
| **频率** | 日频（收盘后），盘中作为 Phase 2 |
| **输出** | 策略建议 + 模拟组合曲线 |
| **风险档位** | 保守(5% 回撤) / 平衡(10%) / 进取(20%) |
| **技术栈** | Python + 本地 LLM 微调（RTX 4090） |
| **数据源** | 优先使用免费源（yfinance/Stooq/NewsAPI） |

---

## 系统架构

![System Architecture](System%20Architecture.png)

---

## 项目结构

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
│   ├── run_walkforward_eval.py       # Phase 14: walk-forward evaluation runner
│   ├── report_compare.py             # Phase 14: stratified comparison report
│   ├── build_dpo_pairs.py            # Phase 12: mine DPO preference pairs
│   ├── train_dpo.py                  # Phase 12: TRL DPO training script
│   ├── analyze_moe_results.py         # Phase 11/12: MoE vs Baseline analysis
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

## 迭代路线图

### Phase 1：数据与回测基础 
- [x] 数据抓取模块（yfinance）
- [x] 交易日历与时区处理
- [x] 技术因子（MA/动量/波动率）
- [x] 回测引擎（成本/滑点/再平衡）
- [x] 绩效指标

### Phase 2：策略 MVP 
- [x] 风险状态检测（Risk-On/Off/Transition）
- [x] 目标波动仓位计算
- [x] 趋势/动量信号
- [x] 回撤保护规则
- [ ] 滚动验证（Walk-forward validation）

### Phase 3：LLM 微调 
- [x] 新闻数据收集（多源 RSS）
- [x] 微调数据集构建（追加/去重/置信度 + 训练/验证分割）
- [x] LoRA/QLoRA 微调（支持 Qwen2.5-14B）
- [x] 新闻结构化推理（base vs LoRA）
- [x] 决策解释生成（base vs LoRA）

### Phase 4：生产流水线 
- [x] 每日新闻流水线（抓取 → 推理 → 日报）
- [x] 信号评测（T+1 对齐，按事件类型分析）
- [x] 健康监控与兜底（CN RSS → JSON API）
- [x] Windows 任务计划程序集成

### Phase 5：ETF 交易模型 + RAG + RiskGate（完成）
- [x] ETF 特征快照（`build_daily_etf_features.py`）
- [x] Teacher 蒸馏数据集（DeepSeek，25k 样本）
- [x] RAG 检索（基于 FAISS 的相似历史）
- [x] RiskGate 确定性约束
- [x] 交易推理流水线（`run_trading_inference.py`）
- [x] CN concept_hype 后处理与降噪
- [x] Model D 训练（Qwen2.5-7B）
- [x] Model D 回测报告（详见工程日志）

### Phase 6：Stock Trader v1.1（Tech + News）（完成）
- [x] News Tower 3B LoRA（noise killer）训练与验收
- [x] 历史 signals 回放（`signals_YYYY-MM-DD.json`）
- [x] Stock SFT 训练集新闻注入
- [x] Stock Trader v1.1（Qwen2.5-7B LoRA）训练完成
- [x] 推理侧 stock 新闻注入（支持 A/B 对照）
- [x] 最终测试（news-conditioning / ablation）验收通过

### Phase 7：Simulation / Backtest（完成）
- [x] 2025-12 stock 特征回放（每日 `stock_features_YYYY-MM-DD.json`）
- [x] Stock 信号质量回测（单基础模型上 LoRA 适配器轮换）
- [x] 升级为 NAV 曲线回测（仓位/成本/回撤）
- [ ] 在 horizon=T+5 重新运行并按强新闻日/平静日分层评估（需要后续数据覆盖）

### Phase 11：Adapter-MoE / Multi-Agent（进行中）
- [x] 推理侧 MoE 路由（`--moe-mode`，scalper vs analyst）
- [x] System 1 基线提示模式（`--use-fast-prompt`）
- [x] RiskGate 阈值通过 CLI 参数化（`--risk-max-drawdown`，`--risk-vol-limit`）

### Phase 12：DPO（少样本偏好对齐）（完成）

1) 从决策日志中挖掘 DPO 偏好对：

```powershell
.\venv311\Scripts\python.exe scripts\build_dpo_pairs.py `
  --inputs data\daily\moe_planner_dec2025.json `
  --daily-dir data\daily `
  --out data\dpo\pairs_moe_planner_dec2025_h5_x002.jsonl `
  --horizon 5 `
  --x 0.02 `
  --target-expert analyst
```

2) 使用 TRL 训练 DPO 适配器：

```powershell
.\venv311\Scripts\python.exe scripts\train_dpo.py `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --sft-adapter models\trader_v2_cot_scaleup\lora_weights `
  --data-path data\dpo\pairs_moe_planner_dec2025_h5_x002.jsonl `
  --output-dir models\trader_v3_dpo_analyst `
  --epochs 3 `
  --batch-size 1 `
  --grad-accum 4 `
  --lr 1e-6
```

3) 在单模型模式下验证特定（日期, ticker）案例：

注意：使用 `--allow-clear` 以便推理 schema 允许 DPO 训练生成的 `CLEAR` 输出。

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py `
  --date 2025-12-05 `
  --universe stock `
  --tickers NFLX,TCOM,RIOT `
  --model Qwen/Qwen2.5-7B-Instruct `
  --load-in-4bit `
  --adapter models\trader_v3_dpo_analyst `
  --allow-clear `
  --risk-watch-market NONE `
  --risk-max-drawdown 1 `
  --risk-vol-limit 1 `
  --out data\daily\dpo_verification_example.json
```

4) 使用 DPO Analyst 运行全月 MoE 推理（宽松风险，2025 年 12 月）：

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py `
  --date-range 2025-12-01 2025-12-31 `
  --universe stock `
  --moe-mode `
  --planner-mode rule `
  --moe-analyst models\trader_v3_dpo_analyst_ultimate\lora_weights `
  --allow-clear `
  --risk-watch-market NONE `
  --risk-max-drawdown 1 `
  --risk-vol-limit 1 `
  --out data\daily\moe_dpo_ultimate_dec2025.json
```

5) 总体分析（基线快速 vs 旧 MoE 宽松 vs 新 MoE DPO 终极，H=5）：

```powershell
.\venv311\Scripts\python.exe scripts\analyze_moe_results.py `
  --moe data\daily\moe_dpo_ultimate_dec2025.json `
  --baseline data\daily\scalper_baseline_fast_dec2025.json `
  --report data\backtest\report_2025_12_final.json `
  --strategy v1_1_news `
  --horizon 5
```

### Phase 12.3：总体分析与最终结论（2025 年 12 月）

| Metric (H=5) | Baseline Fast | Old MoE Loose | New MoE DPO Ultimate | Delta (New - Old) |
| :--- | :--- | :--- | :--- | :--- |
| **Global PnL** | **+0.4518** | -1.4581 | -1.3443 | +0.1138 |
| **Global Trades**| 205 | 689 | 672 | -17 |
| **Analyst Trades**| N/A | **23** | **6** | **-17 (74% Drop)** |
| **Analyst PnL** | N/A | -0.1285 | -0.0147 | +0.1138 |

关键结论：
1. **DPO 成功**："终极冲击"疗法（Beta=1.0 + 增强负样本）成功抑制了 Analyst 的激进性。评分交易从 23 降至 6，有效消除"噪音"，并将 Analyst 特定回撤降低约 90%。
2. **系统噪音**：全局 PnL 仍为负，因为**Scalper 专家**（负责约 660 笔交易）在"宽松风险"模式下运行以促进此次测试。
3. **前进路径**：理想系统为**严格 Scalper（基线）+ 规划器门控 + DPO Analyst**。

### Phase 14：评测平台（Protocol Freeze + Walk-forward + Stratified Report）（进行中）

目标：将评测协议与报告产物标准化，形成可复现的对比基准。

核心产物：

- `configs/baseline_fast_v1.yaml` / `configs/golden_strict_v1.yaml`
- `results/<run_id>/{baseline_fast,golden_strict}/daily.csv`
- `results/<run_id>/metrics.json`
- `results/<run_id>/report.md`

12 月窗口（Sanity Check，可对齐 Phase 13 毕业分析）：

```powershell
.\venv311\Scripts\python.exe scripts\run_walkforward_eval.py --run-id phase14_smoke_3w --windows 2025-12-01 2025-12-31
.\venv311\Scripts\python.exe scripts\report_compare.py --metrics results\phase14_smoke_3w\metrics.json
```

Q4（10-12 月）全量评测（可能耗时较长，默认会打印 `[date i/N] YYYY-MM-DD` 进度）：

```powershell
.\venv311\Scripts\python.exe scripts\run_walkforward_eval.py --run-id phase14_q4_full --baseline-config configs\baseline_fast_v1.yaml --golden-config configs\golden_strict_v1.yaml --windows 2025-10-01 2025-12-31
.\venv311\Scripts\python.exe scripts\report_compare.py --metrics results\phase14_q4_full\metrics.json
```

### Phase 8：全市场扩张 / RL（远期）
- [ ] A 股支持（CN_Trader LoRA）
- [ ] 加股（CA ETFs）
- [ ] 黄金/大宗商品（Macro_Gold LoRA）

### Phase 10：CoT 蒸馏 / 推理升级（Trader v2）
- [x] 从回测报告生成错题本（PoC）
- [x] Teacher 推理生成（严格 JSON）
- [x] 通过 subject_assets 归因改善 ticker-news 相关性
- [x] 使用 reasoning_trace 微调 Trader v2（冒烟测试）并验证端到端接线

#### Phase 10 命令（原型）

环境变量（密钥不入库）：

```bash
TEACHER_API_KEY=sk-...
TEACHER_BASE_URL=https://api.deepseek.com
TEACHER_MODEL=deepseek-reasoner
```

1) 为每日信号回填 ticker 归因：

```powershell
.\venv311\Scripts\python.exe scripts\backfill_signal_assets.py --report data\backtest\report_2025_12_extended.json --strategy v1_1_news --overwrite
```

2) 采样 ticker 专属错题：

```powershell
.\venv311\Scripts\python.exe scripts\sample_mistakes.py --report data\backtest\report_2025_12_extended.json --out data\finetune\mistakes_100_v4.jsonl --strategy v1_1_news --top-k 100 --min-abs-move 0.003 --news-score-threshold 0.0 --news-topk 3
```

3) 生成 Teacher reasoning_trace 数据集：

```powershell
.\venv311\Scripts\python.exe scripts\generate_cot_teacher.py --in data\finetune\mistakes_100_v4.jsonl --out data\finetune\cot_mistakes_100_v4.jsonl --model deepseek-reasoner --delay 1.0 --json-mode --overwrite
```

4) 构建 Trader v2 SFT 数据集（CoT + 回放缓冲区）：

```powershell
.\venv311\Scripts\python.exe scripts\build_trader_v2_dataset.py --cot data\finetune\cot_mistakes_100_v4.jsonl --replay data\finetune\trader_stock_sft_v1_plus_news.json --replay-ratio 1.0 --out-dir data\finetune --val-ratio 0.2
```

5) 微调 Trader v2（混合重训练）：

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\trader_v2_train.json --eval-data data\finetune\trader_v2_val.json --model Qwen/Qwen2.5-7B-Instruct --outdir models\trader_v2_cot --epochs 3 --batch-size 1 --grad-acc 4 --lr 2e-4 --save-steps 10 --eval-steps 10 --max-seq-len 1024 --qlora --grad-ckpt
```

6) 微调 Trader v2（从 v1.1 LoRA 增量持续微调）：

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\trader_v2_train.json --eval-data data\finetune\trader_v2_val.json --model Qwen/Qwen2.5-7B-Instruct --init-adapter models\trader_stock_v1_1_tech_plus_news\lora_weights --outdir models\trader_v2_incremental --epochs 3 --batch-size 1 --grad-acc 4 --lr 2e-4 --save-steps 10 --eval-steps 10 --max-seq-len 1024 --qlora --grad-ckpt
```

#### Phase 10.5 命令（Scale-up / Superset）

1) 采样 news cases（任何有严格 ticker-news 的样本，不限 PnL）：

```powershell
.\venv311\Scripts\python.exe scripts\sample_mistakes.py --report data\backtest\report_2025_12_final.json --strategy v1_1_news --mode all_news --out data\finetune\news_cases_scaleup_min20.jsonl --top-k 200 --min-abs-move 0.003 --news-score-threshold 0.0 --news-topk 3 --min-news-chars 20
```

2) 生成 Teacher CoT（news_case 模式，不假设“原决策一定错”）：

```powershell
.\venv311\Scripts\python.exe scripts\generate_cot_teacher.py --mode news_case --in data\finetune\news_cases_scaleup_min20.jsonl --out data\finetune\cot_news_cases_scaleup.jsonl --model deepseek-reasoner --delay 1.0 --json-mode --overwrite
```

3) 构建 v2 混合训练集（CoT + replay buffer，限制 replay 规模防止失控）：

```powershell
.\venv311\Scripts\python.exe scripts\build_trader_v2_dataset.py --cot-data data\finetune\cot_news_cases_scaleup.jsonl --replay-data data\finetune\trader_stock_sft_v1_plus_news.json --max-replay-samples 1000 --out data\finetune\train_trader_v2_scaleup.json --val-ratio 0.1 --seed 42
```

4) 微调 Trader v2 Superset：

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\train_trader_v2_scaleup.json --eval-data data\finetune\train_trader_v2_scaleup_val.json --model Qwen/Qwen2.5-7B-Instruct --outdir models\trader_v2_cot_scaleup --epochs 3 --batch-size 1 --grad-acc 16 --lr 1e-4 --max-seq-len 1024 --qlora --grad-ckpt --save-steps 100 --eval-steps 100 --eval-batch-size 1
```

5) 实战推理验收（TSLA 例）：

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --lora models\trader_v2_cot_scaleup\lora_weights --use-lora --load-4bit --tickers TSLA --out data\daily\trading_decision_2025-12-08.json
```

#### Phase 11 命令（Adapter-MoE 路由 / 多 LoRA 动态切换）

目标：平静日走“快专家”(scalper)，新闻/剧烈波动走“慢专家”(analyst)。

默认路由（只要该 ticker 命中新闻上下文就走 analyst）：

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --load-4bit --tickers TSLA,NVDA,AAPL --moe-mode --out data\daily\moe_decision_2025-12-08.json
```

显式指定两位专家 adapter 路径：

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --load-4bit --tickers TSLA,NVDA,AAPL --moe-mode --moe-scalper models\trader_stock_v1_1_tech_plus_news\lora_weights --moe-analyst models\trader_v2_cot_scaleup\lora_weights --out data\daily\moe_decision_2025-12-08.json
```

关闭 any-news（仅阈值路由，便于后续接入自定义 news_score）：

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --load-4bit --tickers TSLA,NVDA,AAPL --moe-mode --no-moe-any-news --moe-news-threshold 0.8 --out data\daily\moe_decision_2025-12-08.json
```

---

## 输出协议

每日为每个 ETF 输出：

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

## 风险档位配置

| 档位 | 最大回撤容忍 | 目标年化波动 | 单标的上限 | 现金下限 |
|------|-------------|-------------|-----------|---------|
| 保守 | 5% | 6% | 25% | 30% |
| 平衡 | 10% | 10% | 35% | 15% |
| 进取 | 20% | 15% | 50% | 5% |

---

## 环境配置

### 硬件要求
- **最低**：8GB 内存，无 GPU
- **推荐**：16GB 内存，8GB 显存
- **当前配置**：RTX 4090 + AMD 7950X3D（支持 LLM 微调）

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

- **抓取**新闻（US RSS + CN RSS，并带 CN JSON fallback 兜底）
- **推理**结构化 signals（Qwen2.5 + LoRA）
- **生成**Markdown 日报
- **评测**信号（可选）

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

---

## 可视化（仪表板）

使用 Streamlit 驾驶舱监控模拟盘状态、NAV 曲线、订单流和风险事件。

1. **安装依赖**：

```bash
pip install streamlit pandas plotly
```

2. **启动仪表板**：

```bash
streamlit run scripts/dashboard.py
```

3. **选择数据目录**：

在侧边栏中，将 `Paper Dir` 设置为：

- `data/paper_rolltest`（推荐，包含滚动演练：持有 / 强制平仓 / 待反转 / 反转）
- `data/paper`（您当前实时模拟盘目录）

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

---

## LLM 微调（快速开始）

本仓库提供从公开 RSS 源构建弱标注数据集，并使用 LoRA/QLoRA 微调 Qwen2.5 模型的可用流水线。

### 1) 配置新闻源

编辑 `config/sources.yaml`，按需开启/关闭来源并调整权重/类别。

### 2) 构建微调数据集（追加+去重+质量过滤）

输出在 `data/`（默认不提交到 git）：

```bash
.\venv311\Scripts\python.exe scripts\build_finetune_dataset.py --limit 800 --add-explain --append --dedup --split-val --val-ratio 0.05
```

会生成：

- `data/finetune/train.json`
- `data/finetune/val.json`

### 3) 训练（推荐：3B/7B QLoRA）

推荐本地配置为 Qwen2.5-3B（News）+ Qwen2.5-7B（Trader），使用 QLoRA 4 位量化。

注意：14B 实验视为历史版本，可归档（详见工程日志）。

```bash
.\venv311\Scripts\python.exe scripts\finetune_llm.py --model Qwen/Qwen2.5-3B-Instruct --data data/finetune/train.json --qlora --grad-ckpt --max-seq-len 1024 --batch-size 1 --grad-acc 16 --lr 1e-4 --epochs 3 --save-steps 50 --save-total-limit 5 --outdir models/news_final_3b_v1
```

### 4) 推理（base vs LoRA）

对于 3B/7B + LoRA 推理，建议使用 4 位加载以避免 CPU/磁盘 offload 导致的问题：

```bash
.\venv311\Scripts\python.exe scripts\infer_llm.py --model Qwen/Qwen2.5-3B-Instruct --task news --load-in-4bit
.\venv311\Scripts\python.exe scripts\infer_llm.py --model Qwen/Qwen2.5-3B-Instruct --task news --use-lora --lora models\news_final_3b_v1\lora_weights --load-in-4bit
```

---

## Stock Trader（Phase 6/7）

本仓库包含 Stock Trader 流水线，支持可选的新闻上下文注入。

- 特征生成：`scripts/build_daily_features_universal.py` -> `data/daily/stock_features_YYYY-MM-DD.json`
- 新闻回放：`scripts/backfill_news_signals.py` -> `data/daily/signals_YYYY-MM-DD.json`
- Stock 回测：`scripts/backtest_trader.py` -> `data/backtest/report_*.json`

### 5) Teacher 蒸馏数据集（DeepSeek，ETF）

支持从每日 ETF 特征快照生成高质量 teacher 数据集（OpenAI-compat API，例如 DeepSeek），用于后续 LoRA 蒸馏训练。

双模型架构（参谋长 vs 指挥官）：

- 新闻 LoRA（情报参谋）：把海量新闻噪音结构化成 `signals_*.json`（每日情报简报）。
- 交易模型/LoRA（现场指挥官）：同时读取情报简报（`signals_*.json` / `risk_watch`）与战场地图（`etf_features_*.json`），输出最终动作与目标仓位。

未来架构蓝图（planned）：

- 证据/检索层（RAG）：检索新闻原文片段、宏观数据、历史相似案例，降低幻觉。
- 风控裁决层（确定性）：硬约束最大仓位/回撤/杠杆/切换风格等，把交易 LoRA 产出当作"建议"再裁决。
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

## 免责声明

**本项目仅供学习研究，不构成任何投资建议。**

- 历史业绩不代表未来表现
- 所有策略建议仅供参考，投资决策需自行判断
- 使用本系统产生的任何损失由用户自行承担

---

##  License

MIT License

---
---

# QuantAI - Intelligent Quantitative Investment Assistant

> A multimodal AI-powered personal securities investment assistant providing ETF portfolio strategy recommendations, risk alerts, and explainable decisions.

[中文版](#quantai---智能量化投顾助手) | [Engineering Log](docs/engineering_log_phase2_cn_infusion.md)

---

## Milestone Overview

| Phase | Topic | Status | Description |
| :--- | :--- | :--- | :--- |
| 1 | Bulletproof JSON Pipeline | Done | Strict JSON repair + schema validation. |
| 2 | Teacher Data Generation (CN/US) | Done | Multi-market teacher dataset generation. |
| 3 | LLM Fine-tuning (News LoRA) | Done | News LoRA training + inference. |
| 4 | Production Pipeline (Daily Automation) | Done | Daily pipeline automation + health/fallback. |
| 5 | ETF Trader + RAG + RiskGate | Done | RAG retrieval + deterministic risk gate. |
| 6 | Stock Trader v1.1 (Tech + News) + Dual Tower | Done | News Tower + Trader Tower integration. |
| 7 | Backtest & Execution | Done | NAV curve backtest + `Hold=Keep` + `Confirm=2` execution filter. |
| 8 | Paper Trading | Done | Rolling daily simulation with state persistence + RiskGate CLEAR. |
| 9 | Dashboard | Done | Streamlit cockpit for NAV, orders, and risk monitoring. |
| 10 | CoT Distillation (Reasoning, Trader v2) | In Progress | Mistake book + teacher reasoning_trace for explainable trading. |
| 11 | Adapter-MoE / Multi-Agent | In Progress | LoRA experts + router (MoE) + tunable RiskGate thresholds for A/B. |
| 12 | RL (DPO/GRPO) | Done | DPO preference surgery successfully reduced Analyst noise; full-month MoE run + grand analysis complete (Dec 2025). |
| 13 | Golden Run (Strict Risk + Planner + DPO Analyst) | Done | Full-month Dec 2025 run with strict risk controls and planner gating. |
| 14 | Evaluation Platform (Protocol Freeze + Walk-forward + Stratified Report) | In Progress | Frozen configs + walk-forward runner + date-aligned stratified report. |

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
│   ├── build_dpo_pairs.py            # Phase 12: mine DPO preference pairs
│   ├── train_dpo.py                  # Phase 12: TRL DPO training script
│   ├── analyze_moe_results.py         # Phase 11/12: MoE vs Baseline analysis
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

### Phase 12.3: Grand Analysis & Final Verdict (Dec 2025)

| Metric (H=5) | Baseline Fast | Old MoE Loose | New MoE DPO Ultimate | Delta (New - Old) |
| :--- | :--- | :--- | :--- | :--- |
| **Global PnL** | **+0.4518** | -1.4581 | -1.3443 | +0.1138 |
| **Global Trades**| 205 | 689 | 672 | -17 |
| **Analyst Trades**| N/A | **23** | **6** | **-17 (74% Drop)** |
| **Analyst PnL** | N/A | -0.1285 | -0.0147 | +0.1138 |

Key conclusions:
1. **DPO Success**: The "Ultimate Shock" therapy (Beta=1.0 + Augmented Negatives) successfully curbed the Analyst's aggression. Scored trades dropped from 23 to 6, effectively silencing the "noise" and reducing Analyst-specific drawdowns by ~90%.
2. **System Noise**: The Global PnL remains negative because the **Scalper expert** (responsible for ~660 trades) was running in "Loose Risk" mode to facilitate this test.
3. **Path Forward**: The ideal system is **Strict Scalper (Baseline) + Planner Gating + DPO Analyst**.

### Phase 10: CoT Distillation (Reasoning, Trader v2)
- [x] Generate mistake book from backtest reports (PoC)
- [x] Teacher reasoning generation (strict JSON)
- [x] Improve ticker-news relevance via subject_assets attribution
- [x] Fine-tune Trader v2 with reasoning_trace (smoke) and validate end-to-end wiring
- [x] Phase 10.5: Scale-up to news cases (not only mistakes), mix replay buffer, train Superset Trader v2
- [x] Phase 11 (start): Adapter-MoE routing (multi-LoRA dynamic switching, heuristic routing by news/volatility)

#### Phase 10 Commands (Prototype)

Environment variables (keep secrets out of git):

```bash
TEACHER_API_KEY=sk-...
TEACHER_BASE_URL=https://api.deepseek.com
TEACHER_MODEL=deepseek-reasoner
```

1) Backfill ticker attribution into daily signals:

```powershell
.\venv311\Scripts\python.exe scripts\backfill_signal_assets.py --report data\backtest\report_2025_12_extended.json --strategy v1_1_news --overwrite
```

2) Sample ticker-specific mistakes:

```powershell
.\venv311\Scripts\python.exe scripts\sample_mistakes.py --report data\backtest\report_2025_12_extended.json --out data\finetune\mistakes_100_v4.jsonl --strategy v1_1_news --top-k 100 --min-abs-move 0.003 --news-score-threshold 0.0 --news-topk 3
```

3) Generate teacher reasoning_trace dataset:

```powershell
.\venv311\Scripts\python.exe scripts\generate_cot_teacher.py --in data\finetune\mistakes_100_v4.jsonl --out data\finetune\cot_mistakes_100_v4.jsonl --model deepseek-reasoner --delay 1.0 --json-mode --overwrite
```

4) Build Trader v2 SFT dataset (CoT + replay buffer):

```powershell
.\venv311\Scripts\python.exe scripts\build_trader_v2_dataset.py --cot data\finetune\cot_mistakes_100_v4.jsonl --replay data\finetune\trader_stock_sft_v1_plus_news.json --replay-ratio 1.0 --out-dir data\finetune --val-ratio 0.2
```

5) Fine-tune Trader v2 (mixed re-training):

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\trader_v2_train.json --eval-data data\finetune\trader_v2_val.json --model Qwen/Qwen2.5-7B-Instruct --outdir models\trader_v2_cot --epochs 3 --batch-size 1 --grad-acc 4 --lr 2e-4 --save-steps 10 --eval-steps 10 --max-seq-len 1024 --qlora --grad-ckpt
```

6) Fine-tune Trader v2 (incremental continued fine-tuning from v1.1 LoRA):

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\trader_v2_train.json --eval-data data\finetune\trader_v2_val.json --model Qwen/Qwen2.5-7B-Instruct --init-adapter models\trader_stock_v1_1_tech_plus_news\lora_weights --outdir models\trader_v2_incremental --epochs 3 --batch-size 1 --grad-acc 4 --lr 2e-4 --save-steps 10 --eval-steps 10 --max-seq-len 1024 --qlora --grad-ckpt
```

#### Phase 10.5 Commands (Scale-up / Superset)

1) Sample news cases (any sample with strict ticker-news, regardless of PnL):

```powershell
.\venv311\Scripts\python.exe scripts\sample_mistakes.py --report data\backtest\report_2025_12_final.json --strategy v1_1_news --mode all_news --out data\finetune\news_cases_scaleup_min20.jsonl --top-k 200 --min-abs-move 0.003 --news-score-threshold 0.0 --news-topk 3 --min-news-chars 20
```

2) Generate Teacher CoT (news_case mode, doesn't assume "original decision is wrong"):

```powershell
.\venv311\Scripts\python.exe scripts\generate_cot_teacher.py --mode news_case --in data\finetune\news_cases_scaleup_min20.jsonl --out data\finetune\cot_news_cases_scaleup.jsonl --model deepseek-reasoner --delay 1.0 --json-mode --overwrite
```

3) Build v2 mixed training set (CoT + replay buffer, limit replay size to prevent overflow):

```powershell
.\venv311\Scripts\python.exe scripts\build_trader_v2_dataset.py --cot-data data\finetune\cot_news_cases_scaleup.jsonl --replay-data data\finetune\trader_stock_sft_v1_plus_news.json --max-replay-samples 1000 --out data\finetune\train_trader_v2_scaleup.json --val-ratio 0.1 --seed 42
```

4) Fine-tune Trader v2 Superset:

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\train_trader_v2_scaleup.json --eval-data data\finetune\train_trader_v2_scaleup_val.json --model Qwen/Qwen2.5-7B-Instruct --outdir models\trader_v2_cot_scaleup --epochs 3 --batch-size 1 --grad-acc 16 --lr 1e-4 --max-seq-len 1024 --qlora --grad-ckpt --save-steps 100 --eval-steps 100 --eval-batch-size 1
```

5) Production inference validation (TSLA example):

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --lora models\trader_v2_cot_scaleup\lora_weights --use-lora --load-4bit --tickers TSLA --out data\daily\trading_decision_2025-12-08.json
```

#### Phase 11 Commands (Adapter-MoE Routing / Multi-LoRA Dynamic Switching)

Goal: Use "fast expert" (scalper) on quiet days, "slow expert" (analyst) on news/volatile days.

Default routing (route to analyst if ticker has news context):

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --load-4bit --tickers TSLA,NVDA,AAPL --moe-mode --out data\daily\moe_decision_2025-12-08.json
```

Explicitly specify expert adapter paths:

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --load-4bit --tickers TSLA,NVDA,AAPL --moe-mode --moe-scalper models\trader_stock_v1_1_tech_plus_news\lora_weights --moe-analyst models\trader_v2_cot_scaleup\lora_weights --out data\daily\moe_decision_2025-12-08.json
```

Disable any-news routing (threshold-only, for custom news_score integration):

```powershell
.\venv311\Scripts\python.exe scripts\run_trading_inference.py --date 2025-12-08 --model Qwen/Qwen2.5-7B-Instruct --load-4bit --tickers TSLA,NVDA,AAPL --moe-mode --no-moe-any-news --moe-news-threshold 0.8 --out data\daily\moe_decision_2025-12-08.json
```

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


MIT License
