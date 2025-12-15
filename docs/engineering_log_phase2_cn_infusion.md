# A-Share LLM Logic Infusion 工程日志（Phase 1-2）

> 本文用于记录：我们在“注入 A 股灵魂”过程中遇到的问题、讨论结论、工程实现方案、产物清单与后续规划。
> 重要原则：任何 API Key/Token **不得**写入仓库；只允许通过环境变量或本地 `.env.local`（已在 `.gitignore` 忽略）管理。

## 1. 目标与范围

- 目标：让 LLM 在 **A 股新闻理解与交易语义** 上具备“中国特色金融逻辑”，能正确解读诸如“新质生产力 / 化债 / 国家队 / 立案调查 / ST / 重大资产重组 / 概念炒作”等语境，并稳定输出结构化标签。
- 当前阶段聚焦：
  - Phase 1：工程侧“止血”，保证 **严格 JSON 输出的可解析率**（批处理不中断）。
  - Phase 2：用 teacher 模型生成高质量训练数据，“重训大脑”，注入 A 股 event_type 与方向约束。

## 2. Phase 1：Bulletproof JSON Pipeline（止血）

### 2.1 典型故障模式

- LLM 输出 JSON 时，字符串里夹带未转义的 `"`，导致整体 JSON 解析失败。
- 批量评测/生成时，单条失败会中断 batch，影响迭代效率。

### 2.2 结论与工程策略

- 输入侧清洗：把输入文本中的双引号 `"` 替换成单引号 `'`，从源头降低 JSON 被破坏概率。
- 系统提示强约束：
  - 明确要求输出 **STRICT JSON only**。
  - 明确要求字符串值内部禁止出现双引号（使用单引号替代）。
- 输出侧兜底：
  - 从输出中提取第一个 JSON 对象。
  - 解析失败时尝试 `json_repair`/正则修复。
  - Fail-safe：单条失败记录错误，不中断 batch。

### 2.3 产物与脚本

- `scripts/infer_llm.py`
  - 新闻任务 prompt 强化 + 输入引号清洗。
- `scripts/eval_news_json.py`
  - 单条评测：输入清洗 + 强 prompt + 输出修复。
- `scripts/eval_news_json_batch.py`
  - 批量评测：模型只加载一次；单条 fail-safe；统计解析率。
- `.gitignore`
  - 已忽略 `/scripts/eval_outputs/` 与 `/scripts/eval_results/`，防止评测产物污染仓库。

## 3. A 股专属标签体系（粗粒度 event_type）

为了快速对齐“能用”的中国市场结构化理解，先采用粗粒度枚举：

- `policy_stimulus`
- `regulation_crackdown`
- `market_intervention`
- `concept_hype`
- `corporate_restructuring`

设计原则：

- 先把 **事件类型识别** 与 **方向（impact）** 做稳定，再逐步细化子类。
- A 股常见“黑话”优先：国家队、救市、窗口指导、立案调查/行政处罚、ST、化债、并购重组/借壳、题材炒作。

## 4. Phase 2：Teacher 选型与数据生成

### 4.1 Teacher 选型结论

- 最终选型：**DeepSeek-V3（`deepseek-chat`）**
- 理由：
  - 对 A 股政策隐喻/黑话理解更原生（如“化债”更贴近地方政府隐性债务置换语境）。
  - 成本低，适合批量生成。
  - 指令遵循好，配合 JSON 兜底策略更稳。
- 不选：`deepseek-reasoner`（R1）
  - 可能输出思考过程，干扰 strict JSON。

### 4.2 Teacher 生成脚本

- `scripts/generate_phase2_teacher_dataset.py`
  - OpenAI-compatible `/v1/chat/completions` 调用方式。
  - 已支持 `TEACHER_BASE_URL=https://api.deepseek.com`（会自动补 `/v1`）。
  - 输入：新闻 list（每条包含 `title/content/published_at`）。
  - 输出：可直接用于 LoRA 微调的 `conversations` JSON list。

### 4.3 A 股强制规则（生成后再二次 enforce）

为保证“方向不跑偏”，对 teacher 结果做硬约束：

- `policy_stimulus` => `impact_equity=+1` 且 `impact_bond=+1`
- `regulation_crackdown` => `impact_equity=-1`
- `market_intervention` => `impact_equity=+1`
- `concept_hype` => `impact_equity=+1`（summary 强制提到“炒作/短期/投机”语义）
- `corporate_restructuring` => `impact_equity=+1`

说明：这是一种“先保方向正确”的工程折中；后续可引入 `impact_confidence` 或更细分的结构来缓解过强先验。

## 5. 数据与评测体系

### 5.1 数据配比

- 目标：CN 400 + US 600（US:CN = 60:40）。

### 5.2 固定评测集与指标（规划）

- 建议固定评测集：50-200 条（覆盖：政策/概念/监管/国家队/重组/黑天鹅）。
- 指标：
  - JSON 可解析率（含修复后）
  - 字段缺失率/额外字段率
  - event_type 命中率（人工 spot check）
  - impact 方向一致性（对政策/监管/救市等强信号）

## 6. 安全与密钥管理

- 任何 API Key 不进入仓库、不进入脚本参数默认值。
- 推荐使用：
  - 环境变量：`TEACHER_API_KEY / TEACHER_BASE_URL / TEACHER_MODEL`
  - 或本地 `.env.local`（已加入 `.gitignore`）
- 如果密钥出现在聊天/日志/截图中，应视为泄露并立即作废。

## 7. 后续路线图（高优先级）

- 生成 CN 400 teacher 数据并做 QC：统计 OK/FAIL、解析率、缺失/额外字段分布。
- 补齐 US 600 teacher 或现有 US 数据源，形成 LoRA-C 训练集。
- 去重与评测 harness：避免模板化导致 hash 碰撞；建立稳定评测与一致性指标。
- 扩展新闻数据入口：RSS + 公开历史数据集（避免版权风险）。
- 回测/策略最小闭环：信号→仓位→交易规则→绩效，确保无未来函数。
- 存储预算：控制 checkpoints 数量，仅保留必要 LoRA 权重。

## 8. 每日工程记录（2025-12-14）

### 8.1 当日目标

- 目标 1：生成 CN400 teacher 数据集（DeepSeek-V3 / `deepseek-chat`），确保 strict JSON 可解析与字段稳定。
- 目标 2：补齐“数据管道可靠性”证据：QC 指标、后处理清洗、可复现脚本与工程日志。
- 目标 3：为 LoRA-C（US+CN 混合）训练准备可训练数据文件与训练入口命令。

### 8.2 关键决策与原因

- 决策：使用本地 `.env.local` 管理 `TEACHER_*` 配置（并保持 `.gitignore` 忽略）。
  - 原因：避免在命令行参数/仓库文件中暴露密钥；同时支持 IDE 内直接复现运行。
  - 备注：IDE 对被 `.gitignore` 命中的文件存在访问限制，因此采用“仓库内 `env.local.example` 模板 + 本地复制为 `.env.local`”的双文件策略。

- 决策：teacher 生成完成后，增加“可训练化后处理”步骤。
  - 原因：teacher 输出存在 `sentiment` 类型混用（数字与字符串并存）与少量 `impact_bond` 异常值，会降低微调稳定性。
  - 策略：不重跑 teacher，直接对输出做本地后处理，生成 clean 版本。

- 决策：混合数据集构建使用可复现采样（seed 固定）。
  - 原因：当前 US 训练集规模不足预期（本地现存 `train.json` 为 284），为了先启动 LoRA-C 验证 Phase 2 假设，采用有放回采样补足到 target=1000，同时保留可追溯性。

### 8.3 当日实现与产物

- CN 新闻抓取扩展：
  - `scripts/fetch_cn_news.py` 增加分页抓取能力，最终生成 `data/cn_news_400.json`（400 条）。

- CN400 teacher 生成：
  - `scripts/generate_phase2_teacher_dataset.py` 支持 DeepSeek base_url 规范化（自动补 `/v1`）。
  - 运行结果：`data/finetune/phase2_cn_teacher.json` items=400。

- QC（原始 teacher 输出）：
  - `parse_ok=400, parse_fail=0`
  - `missing={}`，`extra={}`
  - `event_type` 分布：concept_hype 137 / corporate_restructuring 100 / policy_stimulus 78 / regulation_crackdown 60 / market_intervention 25
  - 发现：`sentiment` 混用（如 `negative/neutral/positive` 与 `-1/0/1` 并存）；`impact_bond` 存在少量非预期值。

- 后处理 clean 版本：
  - 产出：`data/finetune/phase2_cn_teacher_clean.json`
  - 处理：
    - `sentiment` 归一到 `-1/0/1`
    - `impact_bond` 规则化：仅 `policy_stimulus=1`，其余置 `0`

- 脚本入库（可复现证据）：
  - `scripts/qc_phase2_cn_teacher.py`
  - `scripts/postprocess_phase2_cn_teacher.py`
  - `scripts/build_phase2_hybrid_dataset.py`（US+CN 目标 1000、US:CN=60:40、seed 固定、可选 shuffle）

### 8.4 当前状态与下一步

- 训练数据：
  - 推荐训练输入使用 `data/finetune/phase2_hybrid_1000.json`（由脚本生成，target=1000，US:CN=60:40）。
  - CN 侧使用 clean 版本：`data/finetune/phase2_cn_teacher_clean.json`。

- 下一步（进入“炼丹”）：
  - 使用 `scripts/finetune_llm.py` 启动 LoRA-C（Qwen2.5-14B + QLoRA + grad checkpointing）。
  - 训练后用固定 A 股 5-case 回归评测：检查 `event_type` 与 impact 强绑定是否符合预期。

- 安全：
  - 若密钥曾出现在聊天/文本中，应视为泄露并尽快作废，后续仅使用本地 `.env.local` 或环境变量。

### 8.5 LoRA-C 微调实测记录（Hybrid 1000, 2025-12-14）

- 训练输入：`data/finetune/phase2_hybrid_1000.json`（1000 条，US:CN=60:40）
- 训练命令（不含密钥）：

```powershell
.\venv311\Scripts\python.exe scripts\finetune_llm.py `
  --model Qwen/Qwen2.5-14B-Instruct `
  --data data\finetune\phase2_hybrid_1000.json `
  --qlora --grad-ckpt --max-seq-len 1024 `
  --batch-size 1 --grad-acc 16 `
  --lr 1e-4 --epochs 3 `
  --save-steps 50 --save-total-limit 5 `
  --outdir models\llm_qwen14b_lora_c_hybrid `
  --resume auto
```

- 训练结果摘要：
  - `train_runtime`: 3477.8323s（约 58min）
  - steps: 189（`18.40s/it`）
  - 末段 loss（log）：0.5997 → 0.5967
  - `train_loss`（全程均值）：0.9075121551594406
  - 保存路径：`models\llm_qwen14b_lora_c_hybrid\lora_weights`

### 8.6 训练后回归评测：CN5 + US5（Base vs LoRA, 2025-12-14）

回归集（合成文本，避免版权风险）：`docs/phase2_regression_cases_cn5_us5.json`。

评测工具：`scripts/infer_llm.py`（支持 `--cases` 批量推理与 `--compare-lora` Base vs LoRA 对比，并打印 `JSON_PARSE_OK_RATE`）。

#### 8.6.1 JSON 可解析率

- CN 5-case：
  - Base：5/5 OK
  - LoRA：5/5 OK
- US 5-case：
  - Base：5/5 OK
  - LoRA：5/5 OK

结论：Phase 1 的“strict JSON + 输入清洗”链路在 LoRA 后仍保持稳定。

#### 8.6.2 CN 侧对齐效果（方向与枚举）

- `policy_stimulus`（稳增长/专项债）：
  - LoRA 输出更贴近硬规则（`impact_bond=1`），方向一致。
- `regulation_crackdown`（立案调查/监管措施）：
  - Base 与 LoRA 均稳定输出 `impact_equity=-1`。
- `market_intervention`（维稳/长期资金入市）：
  - Base 倾向输出 `market_intervention`（符合预期）。
  - LoRA 有一次将其归类为 `policy_stimulus`，并给出 `impact_bond=1`。

结论：LoRA 在“政策刺激”与“维稳干预”的边界上存在混淆；需要通过更明确的提示词约束/更多对比样本来区分。

#### 8.6.3 US 侧漂移（最重要发现）

观察：LoRA 在多条 US 宏观 case 上开始输出中文侧 event_type（如 `policy_stimulus` / `policy_stable`），并倾向给出 `impact_equity=1`、`impact_bond=1` 的模式化结果。

风险：这意味着混合训练在当前规模下（US 284 + CN 400，且 CN 标签体系硬约束较强）可能造成“标签空间被 CN 主导”，从而影响 US case 的结构化语义保持。

#### 8.6.4 下一步动作（建议优先级）

- A（高）：推理侧提示词“分市场枚举”
  - 对 US 明确 event_type enum（US 宏观/公司/风险等），避免模型自由迁移到 CN enum。
  - 对 CN 明确 `market_intervention` 与 `policy_stimulus` 的判别准则（例如：是否包含财政/货币具体工具 vs 仅稳定预期/资金入市表态）。

- B（高）：训练侧补充“反漂移锚点”
  - 增加 US 样本量或提升 US 训练权重（例如 US:CN=70:30 或扩增 US 到 600+）。
  - 或引入一个显式字段（如 `market`）并在训练样本中固定输出（让模型先判别市场，再判别 event_type）。

- C（中）：把回归集扩展为 20-case（CN10+US10），并纳入 nightly 回归。

#### 8.6.5 方向 A 验证结果：US 漂移已被推理侧隔离显著缓解

我们在 `scripts/infer_llm.py` 实施 Split Brain Strategy：

- 对 US：强制 US-only event_type enum（并显式禁止 CN enum）。
- 同时将 system prompt 强化为“标准 JSON（键/字符串必须使用双引号）”，避免单引号字符串导致 `json.loads` 失败。

验证（US 5-case，Base vs LoRA）：

- JSON 可解析率：
  - Base：从 4/5 修复到 5/5
  - LoRA：保持 5/5
- event_type 漂移：
  - LoRA 输出已回到 US enum（`fomc_decision` / `inflation_data` / `jobs_report` / `gdp_data` / `fiscal_tariff`），未再出现 `policy_stimulus` 等 CN enum。

剩余问题（待后续精修）：

- CN 侧 `market_intervention` vs `policy_stimulus` 边界仍存在一次混淆，需要在提示词中加入更明确判别准则或补充对比样本。

## 9. Phase 3：实战化 Production Pipeline（Night Ops, 2025-12-14）

目标：搭起“每日自动选股/风控”的骨架（自动抓新闻 → 自动推理 → 自动出日报）。

### 9.1 Step 1 - The Eyes（RSS 数据获取）

工程决策：

- 不覆盖现有 `config/sources.yaml` 结构。
  - 原因：仓库已有 `version/defaults/sources`（Fed/BEA/BLS/Yahoo 等），直接覆盖会破坏历史脚本与兼容性。
  - 方案：在文件末尾追加 Phase 3 专用字段 `us_sources` / `cn_sources`，让新脚本优先读取新字段，旧脚本仍可继续使用 `sources:`。

新增/更新产物：

- 配置扩展：`config/sources.yaml`
  - 追加 `us_sources`：CNBC、Investing、Yahoo Finance news index
  - 追加 `cn_sources`：Sina Finance、WallstreetCN

- 抓取脚本：`scripts/fetch_daily_rss.py`
  - 行为：抓取近 `--hours`（默认 26h）新闻，做去重与容错
  - 输出：`data/daily/news_YYYY-MM-DD.json`
  - 去重策略：`market + source + url + title` hash
  - 时间处理：尽量解析 `published_parsed/updated_parsed`，否则使用 RFC822 字符串，缺失时间戳的条目默认保留

运行方式：

```powershell
.\venv311\Scripts\python.exe scripts\fetch_daily_rss.py
```

### 9.2 下一步（Step 2/3 规划）

- Step 2 - The Brain：`scripts/run_daily_inference.py`
  - 输入：`data/daily/news_YYYY-MM-DD.json`
  - 加载：LoRA-C 权重（`models\llm_qwen14b_lora_c_hybrid\lora_weights`）
  - 输出：逐条结构化信号 JSON（包含 market + event_type + impacts + summary + url/source），并统计 JSON 解析成功率

- Step 3 - The Mouth：`scripts/generate_daily_report.py`
  - 输入：Step 2 的信号文件
  - 输出：Markdown/HTML 日报（分市场汇总、Top 正负面事件、风险提示、关注标的/板块）

### 9.3 Step 1 修复：CN RSS 被拦截（HTML/404/invalid token）→ API fallback

现象：

- Sina RSS / WallstreetCN RSS 在本地环境下返回 `text/html` / 404 / bozo invalid token，导致 CN entries=0。

处置：

- 在 `scripts/fetch_daily_rss.py` 增加 CN fallback（当 CN RSS 为 0 时自动启用）：
  - Sina roll JSON
  - Eastmoney 快讯
  - CLS 电报

结果：

- 近 26h 抓取总量：234（US=11, CN=223；CN 以 fallback 为主）

### 9.4 Step 2 实测：全量推理 + Split Brain 生效

运行：

```powershell
.\venv311\Scripts\python.exe scripts\run_daily_inference.py --date 2025-12-14 --use-lora --load-in-4bit
```

结果：

- parse_ok：234/234
- market counts：US=11, CN=223
- event_type（top）：
  - concept_hype: 91
  - regulation_crackdown: 63
  - corporate_restructuring: 32
  - policy_stimulus: 23
  - market_intervention: 14

备注：US/CN split-brain 枚举隔离在冒烟（US5+CN5）中验证有效，CN 分支可以稳定输出 CN enum。

### 9.5 Step 3 实测：Markdown 日报生成

运行：

```powershell
.\venv311\Scripts\python.exe scripts\generate_daily_report.py --date 2025-12-14
```

产物：

- `data/daily/report_2025-12-14.md`

### 9.6 工程问题：CN 文本乱码（mojibake）

现象：

- report 中出现类似 `鍗...` / `China鈥檚` 的错码字符（常见于 UTF-8 字节被错误按 GBK/GB18030 解码或反向）。

处置：

- 在抓取端 `scripts/fetch_daily_rss.py` 增加：
  - `resp.content` 显式解码（utf-8/utf-8-sig/gb18030 fallback）
  - 基于启发式 penalty 的 `mojibake repair`（gbk/gb18030 -> utf-8 回转）
- 在日报端 `scripts/generate_daily_report.py` 增加二次兜底 repair（防止上游残留错码）。

### 9.7 Step 2B：Batch 推理的“防爆阀”（OOM/尾部延迟防护）

问题背景：

- Batch 推理时，最长样本会决定整个 batch 的 token 长度与计算成本。
- 如果 RSS 抓入超长正文（例如财报全文/长文 3 万字），会带来：
  - 显存飙升（甚至 OOM）
  - tail latency（短新闻也被长新闻拖慢）

处置（工业级防护）：

- 在 `scripts/run_daily_inference.py` 增加输入侧字符级硬截断：
  - 参数：`--max-input-chars`（默认 6000）
  - 行为：在构建 prompt 前对 `content` 执行 `content[:max] + "...(truncated)"`
- 修复 batch decode 的边界：
  - 使用 `attention_mask.sum()` 得到 per-sample prompt 长度
  - decode 时按每条样本的 `prompt_len` 截断（而不是按 padding 后的统一长度），避免混入 prompt 残留影响 JSON 提取。

附加增强：

- 支持 `--batch-size` 提升吞吐
- 支持 `--resume` + `--save-every` 断点续跑与增量写盘（生产更稳）

### 9.8 压力测试（Stress Test）：核弹样本验证防爆阀生效

目的：

- 验证 `--max-input-chars` 能在极端长文本（5 万字级）下防止 batch OOM 与 tail latency。

步骤：

1) 生成压力测试数据（1 条 50k 字 monster + 7 条正常）：

```powershell
.\venv311\Scripts\python.exe -c "import json; normal='Short news.'; huge='CRASH ' * 10000; data=[{'title':f'News {i}', 'content': huge if i==0 else normal, 'source':'Test', 'market':'US', 'url':'http://test', 'published_at':'2025-12-14'} for i in range(8)]; json.dump(data, open('data/daily/stress_test.json','w'), indent=2); print('Created stress_test.json with 1 monster entry (50k chars).')"
```

2) Batch 推理点火（batch-size=8 + max-input-chars=6000）：

```powershell
.\venv311\Scripts\python.exe scripts/run_daily_inference.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --lora models/llm_qwen14b_lora_c_hybrid/lora_weights \
  --use-lora \
  --data data/daily/stress_test.json \
  --load-in-4bit \
  --batch-size 8 \
  --max-input-chars 6000 \
  --out data/daily/stress_result.json
```

验收结果（关键日志）：

- 截断证据：`Truncated content ... from 59999 to 6000 chars`
- 无 OOM / 无卡顿：推理顺利完成
- parse_ok：8/8
- 输出：`data/daily/stress_result.json`

## 10. Phase 4：对答案（Backtesting/Evaluation, 2025-12-14）

目标：

- 将 LLM 的结构化信号（AI 预测方向）与行情数据（Ground Truth）对齐，计算胜率/覆盖率。

### 10.1 判卷脚本：`scripts/evaluate_signal.py`

输入：

- `data/daily/signals_YYYY-MM-DD.json`

Ground Truth 数据源（本地已有）：

- US：`data/raw/SPY.parquet`
- CN：`data/raw/510300.parquet`（沪深300 ETF 代理）

对齐策略（粗粒度、先跑通）：

- 默认使用 T+1（从新闻 `published_at` 的日期开始，向后找下一交易日的日收益）
- 为避免周末/节假日，向后查找窗口默认 7 天（可调）

对齐口径选项：

- `--align-mode published_at`（默认）：按每条新闻的 `published_at` 日期对齐。
  - 风险：若抓取来源混杂、或时区导致 `published_at` 跨日（例如同一批 signals 里出现 12-14/12-15），会导致样本分散且部分市场无法对齐。
- `--align-mode run_date`：强制把所有条目的对齐基准设为 signals 文件对应的 `--date`。
  - 用途：快速“对答案”首跑（把同一天信号集中对齐到 T+1），避免被 `published_at` 跨日影响样本量。

输出：

- 总体胜率（Accuracy）
- 分市场胜率（US/CN）
- 覆盖率/跳过数（因行情缺失导致无法对齐）

### 10.2 常见坑：行情数据滞后导致“全跳过”

现象：

- 若本地 parquet 的最后交易日早于 signals 的 pub_date+1，所有样本会被判定为无行情数据，从而全部 skipped。

脚本已增强诊断输出：

- 打印 US/CN 行情数据的 `last_return_date`
- 打印 signals 的 `pub_date range`

### 10.3 可选增强：自动补齐行情（需要联网）

为了让“判卷子”一键跑通，脚本提供 `--auto-fetch`：

- 会用现有 `DataFetcher` 拉取缺失日期区间并合并回 `data/raw/*.parquet`

示例（直接对 2025-12-14 signals 计算胜率）：

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --date 2025-12-14 --auto-fetch --fetch-source yfinance --sample 20
```

示例（使用 run_date 口径集中对齐）：

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --date 2025-12-14 --align-mode run_date --auto-fetch --fetch-source yfinance --sample 20
```

备注：

- `--auto-fetch` 会发起外部请求（yfinance/akshare），默认关闭。

### 10.4 工程问题：Beta 污染与按事件类型评估（Event-based Alpha Analysis）

背景：

- 若当日指数本身单边下跌/上涨（强 Beta），用“涨跌方向命中率”直接给新闻信号判卷，会把大量偏多/偏空的事件类型一刀切判错。

处置：

- 在 `scripts/evaluate_signal.py` 增加按 `event_type` 的分组统计（Total/Wins/Acc%），用来定位“哪类事件更接近 Alpha、哪类更像噪音”。

示例（2025-12-14 signals，以 `--align-mode run_date` 对齐到 2025-12-15；510300 当日 -0.15%）：

- `concept_hype`: 60 条，Acc 0.00%
- `policy_stimulus`: 11 条，Acc 0.00%
- `regulation_crackdown`: 15 条，Acc 100.00%
- `market_intervention`: 11 条，Acc 9.09%

### 10.5 评测“滤网”：按事件类型过滤（Risk-only 专场）

新增：

- `scripts/evaluate_signal.py` 增加 `--types`（逗号分隔），仅评测指定 `event_type`。

示例（仅评风控事件）：

```powershell
.\venv311\Scripts\python.exe scripts\evaluate_signal.py --signals data/daily/signals_full_2025-12-14.json --date 2025-12-14 --align-mode run_date --types regulation_crackdown
```

### 10.6 自动化积累：每日一键流水线（Windows）

背景：

- RSS 通常仅保留近 24h，无法天然回溯历史；因此需要从今天开始日更积累 signals + 评测数据。

新增：

- 根目录新增 `run_pipeline.bat`：一键执行 `fetch_daily_rss.py` → `run_daily_inference.py` → `generate_daily_report.py`。
- `TODAY` 由 Python 生成（避免 Windows 区域设置导致 `%date%` 切片不一致）。

## 11. 新闻模块运维：每日抓取健康检查（2025-12-15）

新增：

- `scripts/fetch_daily_rss.py` 增加 `--health-out`：输出抓取健康检查 JSON。
  - `--health-out auto`：写入 `data/daily/health_YYYY-MM-DD.json`

健康指标包含：

- `by_source[market:source]`：rss_entries / rss_in_window / rss_added / rss_errors / fallback_added
- `cn`：rss_items / fallback_used / fallback_candidates / fallback_added
- `totals`：rss_added / rss_errors / fallback_added / final_total

本地验证（2h 窗口）：

- CN RSS 0 条时自动启用 fallback，并写出 health 文件用于后续监控。

## 12. Phase 4 增强：多日聚合评测 + 风控专栏（2025-12-15）

新增：

- `scripts/evaluate_signal.py` 支持 `--scan-daily`：扫描 `data/daily/signals_*.json`（含 `signals_full_*.json`）并输出跨日汇总统计。
- `scripts/generate_daily_report.py` 增加 `Risk Watch (CN: regulation_crackdown)` 专栏：统计数量/占比并列出 Top 条目。

## 13. Phase 5（A/A）：ETF 特征快照 + 日更接入（2025-12-15）

目标：

- 让后续 LLM 学会在 ETF/指数层面结合技术面与风险状态输出仓位/风控建议（先做结构化特征与 teacher 产物，再蒸馏微调）。

新增：

- `scripts/build_daily_etf_features.py`
  - 输入：`data/raw/*.parquet`
  - 输出：`data/daily/etf_features_YYYY-MM-DD.json`
  - 内容：技术指标（MA/动量/波动/回撤等）+ 市场 Regime（SPY/VIX 可选）+ teacher 目标仓位（含 risk_profile 约束）。

接入：

- `run_pipeline.bat` 增加一步生成 ETF 特征快照（在 fetch 与 LLM 推理之间）。
- `scripts/generate_daily_report.py` 增加 `ETF Feature Summary`（若特征文件存在则展示）。

## 14. Phase 5.2：豪华版 Teacher 蒸馏（DeepSeek → Qwen，2025-12-15）

目标：

- 用 DeepSeek 作为 Teacher 对 ETF 特征快照进行“多角色推演 + 综合”，生成高质量蒸馏样本（适配后续 LoRA 训练）。

新增：

- `scripts/generate_etf_teacher_dataset.py`
  - 输入：`data/daily/etf_features_YYYY-MM-DD.json`（按日期区间扫描）
  - 可选：读取同日 `data/daily/signals_YYYY-MM-DD.json`，提取 `regulation_crackdown` 风控摘要作为上下文
  - 输出：`data/finetune/teacher_etf/teacher_etf.jsonl`（JSONL，支持追加）

输出格式（每行一个样本）：

- `output` 为单个 JSON 对象，包含：
  - `role_aggressive` / `role_risk` / `role_quant`
  - `synthesis`
  - `label`（action/target_position/risk_notes/rationale）

运维与安全阀：

- `--resume`：按样本 id 去重断点续跑
- `--max-output-tokens`：限制 teacher 输出长度（避免无穷烧钱）
- `--max-retries` + `--timeout`：失败重试与超时控制

配置：

- 通过 env 读取：`TEACHER_API_KEY` / `TEACHER_BASE_URL` / `TEACHER_MODEL`
  - DeepSeek OpenAI-compat 示例：`TEACHER_BASE_URL=https://api.deepseek.com`，`TEACHER_MODEL=deepseek-chat`

运行记录：

- 通过 `--variants` 支持同一（date,symbol）多视角多样本生成（用于“烧钱换质量”）。
- 以 `2025-12-14` 为例：7 个 ETF × 6 variants = 42 行 JSONL（`failed=0`）。
- Windows PowerShell 抽样解析注意：建议 `Get-Content ... -Encoding UTF8 | ConvertFrom-Json`，否则可能出现乱码导致解析失败。

## 15. Phase 5.3：Teacher 数据扩量（20k 级别，2025-12-15）

目标：

- 为过夜 QLoRA 蒸馏准备 20k 级别 teacher 样本（仅靠单日数据远远不够）。

新增/增强：

- `scripts/build_daily_etf_features.py`
  - 增加 `--start-date/--end-date`：按交易日批量生成 `etf_features_YYYY-MM-DD.json`
  - 批量模式以 SPY 交易日为基准日历（缺失则退化为全标的日期并集）
- `scripts/generate_etf_teacher_dataset.py`
  - 增加 `--workers`：并发调用 teacher API（否则 20k 顺序调用会超时）
  - 增加 `--cot-ratio`：抽样启用长推演（在成本与质量之间折中）

建议口径：

- 以 7 个 ETF、`variants=6` 为例：约 500 个交易日 ≈ 21k 样本（满足 20k 目标）。

样本配比（大A:美股=60:40）：

- 通过按市场分配 `variants` 实现：CN ETF（510xxx）生成更多 variants，US ETF（SPY/QQQ/TLT/GLD）生成较少 variants。
