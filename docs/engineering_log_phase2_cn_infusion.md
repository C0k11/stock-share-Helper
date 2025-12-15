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
