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
| 10 | CoT 蒸馏 / Reasoning 升级（Trader v2） | 完成 | 2025-12 |
| 11 | Adapter-MoE / Multi-Agent | 完成 | 2025-12 |
| 12 | DPO / GRPO | 完成 | 2025-12 |
| 13 | 黄金运行（严格风险 + 规划器 + DPO Analyst） | 完成 | 2025-12 |
| 14 | 评测平台（Protocol Freeze + Walk-forward + Stratified Report） | 进行中 | 2025-12 |
| 15 | Q4 Walk-forward 报告 + Alpha Mining + Surgical DPO（Analyst Alpha Hunter / Alpha Max V3） | 完成 | 2025-12-25 |
| 16 | 日报生成器 + Paper Trading（产品化：Daily Job / Ledger / NAV / Charts） | 完成 | 2025-12-25 |
| 17 | Planner Dataset / SFT + MRI / Showdown | 完成 | 2025-12-25 |
| 18 | MoE Router 修复 + signals_assets 回填 + Take 5 验证（2022-06） | 完成 | 2025-12-26 |
| 19 | Planner Offline RL / Bandit（19.2 Gatekeeper v2：Showdown + 阈值 Sweep，默认阈值 0.05） | 完成 | 2025-12-27 |
| 20 | 数据飞轮（Unified Data Harvester）+ 模拟实盘压力测试 | 进行中 | 2025-12 |
| 21 | 多模态视觉之眼（Visual Alpha / Chartist） | 进行中 | 2025-12 |
| 22 | 宏观指挥官（Macro-Agent Hierarchy） | 规划中 | - |
| 23 | 深度思考与辩论（System-2 Debate） | 规划中 | - |
| 24 | 精细化执行（Execution Algorithms） | 规划中 | - |

### Phase 15 最新进展（Alpha Mining / Alpha Days Compass）

- **Hell-month 压力测试（2022-06）**：完成 Baseline / Golden Strict V1 / Experiment Alpha V4（`alpha_max_v3`）对照回测。
- **关键结论**：在 Analyst 未触发（news signals 缺失、`moe_vol_threshold=-1`）的窗口内，V4 与 V1 指标一致，用于验证系统接线的幂等与安全。
- **News Injection 验证（2022-06 部分窗口）**：注入 news signals 后，确认 Analyst 可被唤醒（`analyst_coverage > 0`），且 6/10 极端下跌日 Analyst reasoning 明确引用 CPI/通胀语境；同时 Baseline 在 6/10 的表现与 Golden 相同，证明基础风控（Risk Manager / Drawdown Gate）具备“保命”能力。
- **差异日（Alpha）线索**：在 2022-06-06 出现显著差异（Golden - Baseline ≈ +2.0%），Phase 15.2 的 alpha pair 挖掘将优先围绕该日期展开。
- **Alpha Days 罗盘（Rich Alpha Compass）**：已从 `daily.csv` 生成增强版 `alpha_days.csv`，包含 `total_news_vol/max_news_impact/avg_vol/suggest_upsize` 并支持 `DEFENSIVE_ALPHA`

### Phase 17：Planner SFT（Tabular MLP）Quickstart

目标：把 Planner 的日级 `planner_strategy`（如 `defensive/aggressive_long/cash_preservation`）从 rule 可学习化，并支持在日报中输出 strategy + 置信度（MRI）。

1) 构建训练集（按日聚合 -> CSV）：

```powershell
.\venv311\Scripts\python.exe scripts\training\build_planner_dataset.py --run-dir results\phase15_5_SHOWDOWN_v4_WITH_NEWS_2022-06-01_2022-06-22 --system golden_strict --out data\training\planner_dataset_v1.csv --start 2022-06-01 --end 2022-06-22
```

2) 训练 MLP（输出模型 bundle）：

```powershell
.\venv311\Scripts\python.exe scripts\training\train_planner_sft.py --data data\training\planner_dataset_v1.csv --out models\planner_sft_v1.pt
```

3) 生成带 MRI 的日报（即使历史 run 是 rule，也可离线补齐 probs）：

```powershell
.\venv311\Scripts\python.exe scripts\product\generate_daily_report.py --run-dir results\phase15_5_SHOWDOWN_v4_WITH_NEWS_2022-06-01_2022-06-22 --system golden_strict --date 2022-06-06 --planner-policy sft --planner-sft-model models\planner_sft_v1.pt --out reports\daily\2022-06-06_planner_ai_insight.md
```

产物路径：

- `data/training/planner_dataset_v1.csv`（不进 git）
- `models/planner_sft_v1.pt`（不进 git）

### Phase 15.3：回马枪（Analyst DPO / Forward-Return Pairs）Quickstart

目标：用未来收益（forward return）自动生成大规模 preference pairs，训练 Analyst 的 DPO adapter。

1) 生成 DPO pairs（h=5, x=0.005；含 NEG 样本）：

```powershell
.\venv311\Scripts\python.exe scripts\build_dpo_pairs.py --inputs "results\phase15_5_SHOWDOWN_v4_WITH_NEWS_2022-06-01_2022-06-22\golden_strict\decisions_2022-06-01_2022-06-22.json" --daily-dir "data\daily" --out "data\dpo\phase15_3_pairs_h5_x0005.jsonl" --horizon 5 --x 0.005 --target-expert analyst --min-abs-impact 0.5 --max-news-signals 3 --include-nonbuy
```

2) 启动 DPO 训练（以 v4 Analyst LoRA 为起点）：

```powershell
.\venv311\Scripts\python.exe scripts\train_dpo.py --base-model "Qwen/Qwen2.5-7B-Instruct" --sft-adapter "models\trader_v4_dpo_analyst_alpha" --data-path "data\dpo\phase15_3_pairs_h5_x0005.jsonl" --output-dir "models\phase15_3_analyst_dpo_h5_x0005" --epochs 1 --batch-size 1 --grad-accum 8 --lr 5e-6 --beta 0.1
```

产物路径：

- `data/dpo/phase15_3_pairs_h5_x0005.jsonl`（不进 git）
- `models/phase15_3_analyst_dpo_h5_x0005`（不进 git）

提取命令（示例）：

```powershell
.\venv311\Scripts\python.exe scripts\mining\extract_alpha_days.py --run-dir results\phase15_5_showdown_alpha_v4_jun2022\golden_strict
```

产物路径：

- `results/phase15_5_showdown_alpha_v4_jun2022/golden_strict/alpha_days.csv`

复现要点（新闻数据源与信号生成）：

```powershell
\venv311\Scripts\python.exe scripts\fetch_historical_news_gdelt.py --start 2022-06-01 --end 2022-06-30 --output data\raw\news_us_raw_2022_06.jsonl
\venv311\Scripts\python.exe scripts\backfill_news_signals.py --news-data data\raw\news_us_raw_2022_06.jsonl --out-dir data\daily --start 2022-06-01 --end 2022-06-30 --overwrite
```

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

## Mari Agent Hub（新闻分发 + 多分析员）

新增后端能力：从聊天或 API 提交一段新闻（可附 URL），系统自动分发给多个分析员（macro/risk/factcheck/sentiment），回收结构化结果并生成可执行总结；全流程写入审计日志，并保存到 Mari 长期记忆。

### API

- `POST /api/v1/news/submit`：提交新闻（text/url/source）并返回 `news_id`
- `GET /api/v1/news/{news_id}`：查询分析进度与最终总结
- `GET /api/v1/agents/audit?limit=200`：查看审计日志（JSONL）
- `GET /api/v1/tools/fetch_url?url=...`：网页抓取工具（支持全网，默认阻断内网/localhost）
- `POST /api/v1/actions/execute`：统一动作执行器（按钮/系统动作）

### Chat 使用

在聊天框里直接输入：

- `/news 这里粘贴新闻正文或链接`

完成后会返回 `news_id`，你也可以把 `news_id` 再发给 Mari 获取总结。

### ACTION（让 Mari “点按钮”）

当 Mari 需要触发按钮/系统动作时，会输出一个或多个：

```action
{"action":"start_rl","params":{}}
```

桌面端可配置为自动执行（见 `configs/secretary.yaml`）。

支持的 action：

- `start_rl` / `stop_rl`
- `set_mode`（params: `{ "mode": "online" | "offline" }`）
- `submit_news`（params: `{text, url?, source?}`）
- `fetch_url`（params: `{url, timeout_sec?}`）
- `remember`（params: `{content, category?, importance?}`）

### 配置开关（configs/secretary.yaml）

- `network.allow_all`: 是否允许全网抓取
- `network.block_private`: 是否阻断内网/localhost（SSRF 防护）
- `actions.auto_execute`: 桌面端是否自动执行 `action` 指令
- `rl.auto_start`: Live runner 启动时是否自动开启在线 RL

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

1) 采样 ticker 专属错题：

```powershell
.\venv311\Scripts\python.exe scripts\sample_mistakes.py --report data\backtest\report_2025_12_extended.json --strategy v1_1_news --out data\finetune\mistakes_100_v4.jsonl --top-k 100 --min-abs-move 0.003 --news-score-threshold 0.0 --news-topk 3
```

2) 生成 Teacher CoT（严格 JSON）：

```powershell
.\venv311\Scripts\python.exe scripts\generate_cot_teacher.py --in data\finetune\mistakes_100_v4.jsonl --out data\finetune\cot_mistakes_100_v4.jsonl --model deepseek-reasoner --delay 1.0 --json-mode --overwrite
```

3) 构建 Trader v2 SFT 数据集（CoT + 回放缓冲区）：

```powershell
.\venv311\Scripts\python.exe scripts\build_trader_v2_dataset.py --cot data\finetune\cot_mistakes_100_v4.jsonl --replay data\finetune\trader_stock_sft_v1_plus_news.json --replay-ratio 1.0 --out-dir data\finetune --val-ratio 0.2
```

4) 微调 Trader v2（混合重训练）：

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py --data data\finetune\trader_v2_train.json --eval-data data\finetune\trader_v2_val.json --model Qwen/Qwen2.5-7B-Instruct --outdir models\trader_v2_cot --epochs 3 --batch-size 1 --grad-acc 4 --lr 2e-4 --save-steps 10 --eval-steps 10 --max-seq-len 1024 --qlora --grad-ckpt
```

5) 微调 Trader v2（从 v1.1 LoRA 增量持续微调）：

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
.\venv311\Scripts\python.exe scripts\build_trader_v2_dataset.py --cot-data data\finetune\cot_news_cases_scaleup.jsonl --replay-data data\finetune\trader_stock_sft_v1_plus_news.json --max-replay-samples 1000 --out-dir data\finetune --val-ratio 0.1
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

### Phase 16：日报生成器 + Paper Trading（产品化）Quickstart

Phase 16 把“抓取 → signals → 特征 → 交易推理 → 日报 → 纸上交易 + 账本/NAV/图表”收敛为一个入口脚本：

```powershell
.\venv311\Scripts\python.exe scripts\product\run_daily_job.py --date 2025-12-14
```

---

## Phase 21：Visual Alpha（Chartist）Quickstart

目标：把 OHLCV 变成“可被视觉模型理解的图像”（Retina），再用 VLM 作为独立专家输出结构化技术面信号。

### 21.1 生成 K 线图（Retina）

默认使用项目标准原始数据源：`data/raw/<TICKER>.parquet`。

```powershell
.\venv311\Scripts\python.exe scripts\data\generate_charts.py `
  --asof 2024-01-31 `
  --tickers AAPL,MSFT,NVDA `
  --lookback 60 `
  --out-jsonl data\charts\2024-01-31\charts_base64.jsonl
```

产物：

- `data/charts/2024-01-31/AAPL.png`（以及其它 ticker）
- `data/charts/2024-01-31/charts_base64.jsonl`（供 VLM API 调用）

### 21.2 Chartist（图表专家）推理

Chartist 采用 **OpenAI-compatible API**（支持本地 vLLM/Ollama 或云端），脚本只做 HTTP 调用，解耦显存与推理进程。

#### Ollama（Windows）本地 VLM（OpenAI-compatible）

安装（命令行，可复现）：

```powershell
winget install -e --id Ollama.Ollama
```

注意：安装完成后需要 **重启终端**（让 PATH 生效）。

拉取视觉模型：

```powershell
ollama pull llama3.2-vision
```

可选：把 Ollama 模型缓存统一放到指定目录（推荐）：

```powershell
setx OLLAMA_MODELS "D:\\Project\\ml_cache\\ollama\\models"
```

注意：设置后需要 **重启终端**（并建议重启 Ollama 进程）使其生效。

可选：验证 OpenAI-compatible 端点：

```powershell
powershell -NoProfile -Command "irm http://localhost:11434/v1/models"
```

环境变量（密钥不入库）：

- `OPENAI_BASE_URL`（示例：`http://127.0.0.1:8000/v1`）
- `OPENAI_API_KEY`（本地可用 `EMPTY`）
- `VLM_MODEL`（示例：`Qwen2-VL-7B-Instruct`）

Dry-run（不需要启动任何 VLM 服务，用于验证 I/O 与输出格式）：

```powershell
.\venv311\Scripts\python.exe scripts\inference\run_chart_expert.py `
  --asof 2024-01-31 `
  --dry-run `
  --limit 3 `
  --out-jsonl results\phase21_chartist\chart_signals_dryrun.jsonl
```

真实推理（需要你已启动本地 VLM 服务）：

```powershell
.\venv311\Scripts\python.exe scripts\inference\run_chart_expert.py `
  --asof 2024-01-31 `
  --out-jsonl results\phase21_chartist\chart_signals.jsonl
```

Ollama 示例（llama3.2-vision）：

```powershell
.\venv311\Scripts\python.exe scripts\inference\run_chart_expert.py `
  --asof 2024-01-31 `
  --limit 3 `
  --model "llama3.2-vision" `
  --api-base "http://localhost:11434/v1" `
  --api-key "ollama" `
  --out-jsonl results\phase21_chartist\chart_signals_real_smoke.jsonl
```

### 21.3 Chartist Overlay（接入交易推理 / Walk-forward Eval）

当你已经生成了 Jan 2024 的 chartist signals（示例：`results\phase21_chartist\jan2024_signals.jsonl`），可以通过 `run_walkforward_eval.py` 将其注入 Golden Strict（Baseline 作为对照不注入）。

关键参数：

- `--chart-signals-file <jsonl>`：Chartist signals（`ticker/asof/signal/confidence`）
- `--chart-confidence <float>`：置信度阈值
- `--chart-mode {standard,conservative}`：
  - `standard`：允许 UPGRADE + BLOCK
  - `conservative`：仅 BLOCK

阈值注意事项（Ceiling Effect）：

- 在 Jan 2024 的一批 signals 中观测到 `confidence` 上限为 `0.8`
- overlay 打分规则为 `if conf <= threshold: score=0`
- 因此当 `--chart-confidence >= 0.8`（含 0.80）时可能导致 overlay **0 次触发**

#### Phase 21.6：Smoke Run（wakeup_test, 7 天窗口）

目标：先验证 overlay “确实触发”并观察收益/回撤趋势。

```powershell
.\venv311\Scripts\python.exe scripts\run_walkforward_eval.py `
  --run-id phase21_6_wakeup_test `
  --baseline-config configs\baseline_fast_v1.yaml `
  --golden-config configs\experiment_alpha_v4.yaml `
  --windows 2024-01-03 2024-01-10 `
  --planner-mode sft `
  --override-planner-sft-model models\planner_sft_v1.pt `
  --override-planner-rl-model models\rl_gatekeeper_v2.pt `
  --override-planner-rl-threshold 0.05 `
  --chart-signals-file results\phase21_chartist\jan2024_signals.jsonl `
  --chart-confidence 0.75 `
  --chart-mode standard `
  --force-rerun
```

输出：

- `results\phase21_6_wakeup_test\metrics.json`
- `results\phase21_6_wakeup_test\golden_strict\decisions_2024-01-03_2024-01-10.json`

该窗口观测（用于 sanity check）：

- overlay 触发：`UPGRADE=165`，`BLOCK=0`
- `pnl_sum_net['1']`：Golden `0.2053` vs Baseline `0.1812`
- `max_drawdown`：Golden `-0.0496` vs Baseline `-0.0630`

#### Phase 21.7：Jan 2024 Full Month（0.75 Standard）

```powershell
.\venv311\Scripts\python.exe scripts\run_walkforward_eval.py `
  --run-id phase21_7_jan2024_full_standard_075 `
  --baseline-config configs\baseline_fast_v1.yaml `
  --golden-config configs\experiment_alpha_v4.yaml `
  --windows 2024-01-03 2024-01-31 `
  --planner-mode sft `
  --override-planner-sft-model models\planner_sft_v1.pt `
  --override-planner-rl-model models\rl_gatekeeper_v2.pt `
  --override-planner-rl-threshold 0.05 `
  --chart-signals-file results\phase21_chartist\jan2024_signals.jsonl `
  --chart-confidence 0.75 `
  --chart-mode standard `
  --force-rerun
```

### Phase 16：主要产物

- `reports/daily/2025-12-14.md`
- `reports/daily/assets/2025-12-14_nav.png` / `2025-12-14_allocation.png`
- `data/paper/orders/orders_2025-12-14.csv`（以及 `data/paper/state.json` / `data/paper/portfolio.json`）
- `paper_trading/live_ledger.csv`（实盘账本，append-only）
- `paper_trading/nav.csv`（NAV time-series，append-only）
- `paper_trading/daily_signals.csv`（daily snapshot，append-only）
- `paper_trading/log.csv`（compat export）

回放模式（从历史 results 抽取某一天 decisions 生成日报 + 更新账本）：

```powershell
.\venv311\Scripts\python.exe scripts\product\run_daily_job.py --date 2022-06-06 --skip-fetch --skip-news-parse --skip-features --skip-trading-inference --results-run-dir results\phase15_5_SHOWDOWN_v4_WITH_NEWS_2022-06-01_2022-06-22 --results-system golden_strict
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
