"""quantai.distill — 教师-学生蒸馏管线（DeepSeek 教师 → 本地 Qwen QLoRA/DPO 学生资料）。

- `scenarios`：真实行情 + analysis/ 指标 → 分析任务 prompt（纯逻辑）。
- `client`：DeepSeek OpenAI 兼容客户端（key 只从 env；Mock 供测试零调用）。
- `generator`：场景 × 教师 → SFT(`conversations`) / DPO(`prompt/chosen/rejected`) JSONL，
  格式与 `quantai.llm.{finetune,dpo}` 严格对接。

成本闸：真实生成只走 `scripts/distill.py --run --confirm-spend`（用户手动），
CI/测试零真实调用。
"""

from quantai.distill.client import DeepSeekClient, MissingApiKeyError, MockDeepSeekClient
from quantai.distill.generator import (
    DistillGenerator,
    scenario_to_dpo_record,
    scenario_to_sft_record,
    write_jsonl,
)
from quantai.distill.journal import (
    REVIEW_SYSTEM_PROMPT,
    backfill_outcomes,
    build_review_messages,
    journal_to_dpo_records,
    journal_to_sft_records,
    load_journal,
    parse_review_score,
    rule_student_answer,
    run_daily_journal,
)
from quantai.distill.scenarios import (
    Scenario,
    ScenarioBuilder,
    build_indicator_brief,
    weak_baseline_answer,
)

__all__ = [
    "Scenario",
    "ScenarioBuilder",
    "build_indicator_brief",
    "weak_baseline_answer",
    "DeepSeekClient",
    "MockDeepSeekClient",
    "MissingApiKeyError",
    "DistillGenerator",
    "scenario_to_sft_record",
    "scenario_to_dpo_record",
    "write_jsonl",
    "REVIEW_SYSTEM_PROMPT",
    "rule_student_answer",
    "build_review_messages",
    "parse_review_score",
    "run_daily_journal",
    "backfill_outcomes",
    "load_journal",
    "journal_to_sft_records",
    "journal_to_dpo_records",
]
