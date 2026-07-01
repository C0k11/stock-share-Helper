"""蒸馏数据生成：场景 × 教师客户端 → SFT / DPO 训练 JSONL。

输出格式**严格对接**现有训练器（有测试锁）：
- SFT（`quantai.llm.finetune.QLoRAFineTuner`）：每行
  `{"conversations": [{"role": "system"|"user"|"assistant", "content": ...}, ...]}`
  ——`build_training_texts` 直接套 chat 模板。
- DPO（`quantai.llm.dpo.DPORunner`）：每行
  `{"prompt": <消息列表>, "chosen": str, "rejected": str}`
  ——`REQUIRED_DPO_COLUMNS` 校验 + `format_prompt_messages` 兼容消息列表。

诚实口径：这是**离线蒸馏**（教师批量生成资料 → 学生离线 QLoRA/DPO），与项目里
lookahead 自查、online-learning 诚实占位是同一种 integrity 叙事——不宣称在线学习。
DPO 的 rejected 侧当前是确定性弱基线（见 `weak_baseline_answer` 的诚实说明）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

from quantai.distill.scenarios import Scenario, weak_baseline_answer


def scenario_to_sft_record(scenario: Scenario, teacher_answer: str) -> dict:
    """场景 + 教师回答 → SFT 行（conversations 格式）。"""
    return {
        "conversations": scenario.messages + [{"role": "assistant", "content": teacher_answer}],
        "meta": {"scenario_id": scenario.scenario_id, "symbol": scenario.symbol, "task": scenario.task},
    }


def scenario_to_dpo_record(
    scenario: Scenario, chosen: str, rejected: Optional[str] = None
) -> dict:
    """场景 + 教师回答（chosen）→ DPO 偏好对行。rejected 默认弱基线。"""
    return {
        "prompt": scenario.messages,
        "chosen": chosen,
        "rejected": rejected if rejected is not None else weak_baseline_answer(scenario),
        "meta": {"scenario_id": scenario.scenario_id, "symbol": scenario.symbol, "task": scenario.task},
    }


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    """逐行写 JSONL（UTF-8，无 BOM）。返回行数。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


class DistillGenerator:
    """把场景流喂给教师客户端，落 SFT+DPO JSONL。

    客户端注入（`DeepSeekClient` 或 `MockDeepSeekClient`）——本类不碰网络配置、
    不读密钥；成本闸在 CLI 层。`on_progress` 可注入进度回调（CLI 打印）。
    """

    def __init__(
        self,
        client,
        on_progress: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self.client = client
        self.on_progress = on_progress or (lambda i, sid: None)

    def run(
        self,
        scenarios: Iterable[Scenario],
        sft_path: str | Path,
        dpo_path: str | Path,
        limit: Optional[int] = None,
    ) -> dict:
        """逐场景调教师 → 同时产 SFT 与 DPO 行。返回统计摘要。

        单场景失败不炸整批（记录进 `failures`，最后如实汇报）。
        `limit` 是硬上限（额度保险丝）：达到即停。
        """
        sft_records: list[dict] = []
        dpo_records: list[dict] = []
        failures: list[dict] = []
        for i, sc in enumerate(scenarios):
            if limit is not None and i >= limit:
                break
            try:
                answer = self.client.chat(sc.messages)
                sft_records.append(scenario_to_sft_record(sc, answer))
                dpo_records.append(scenario_to_dpo_record(sc, chosen=answer))
                self.on_progress(i, sc.scenario_id)
            except Exception as exc:  # noqa: BLE001 - 单场景失败不应中断整批
                failures.append({"scenario_id": sc.scenario_id, "error": str(exc)})
        n_sft = write_jsonl(sft_path, sft_records)
        n_dpo = write_jsonl(dpo_path, dpo_records)
        return {
            "sft_written": n_sft,
            "dpo_written": n_dpo,
            "failures": failures,
            "usage": dict(getattr(self.client, "usage_totals", {})),
            "model": getattr(self.client, "model", "unknown"),
        }
