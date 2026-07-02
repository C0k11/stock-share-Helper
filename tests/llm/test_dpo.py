"""quantai.llm.dpo 测试（纯逻辑接缝，不下真模型/不跑 trl）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantai.config import AppConfig
from quantai.llm import dpo
from quantai.llm.dpo import (
    DPORunner,
    build_dpo_config_dict,
    format_prompt_messages,
    has_chat_template,
    validate_dpo_columns,
)


# --------------------------------------------------------------------------- #
# 列校验
# --------------------------------------------------------------------------- #
def test_validate_dpo_columns_ok() -> None:
    validate_dpo_columns(["prompt", "chosen", "rejected", "extra"])  # 不抛


def test_validate_dpo_columns_missing_raises() -> None:
    with pytest.raises(ValueError) as ei:
        validate_dpo_columns(["prompt", "chosen"])
    assert "rejected" in str(ei.value)


# --------------------------------------------------------------------------- #
# chat_template 检测 + prompt 格式化
# --------------------------------------------------------------------------- #
class _TokWithTemplate:
    chat_template = "{{ x }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "TEMPLATED:" + "|".join(m["content"] for m in messages)


class _TokNoTemplate:
    chat_template = None


class _TokRaises:
    @property
    def chat_template(self):
        raise RuntimeError("boom")


def test_has_chat_template_true() -> None:
    assert has_chat_template(_TokWithTemplate()) is True


def test_has_chat_template_false() -> None:
    assert has_chat_template(_TokNoTemplate()) is False


def test_has_chat_template_exception_is_false() -> None:
    assert has_chat_template(_TokRaises()) is False


def test_format_prompt_messages_uses_template_when_available() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    assert format_prompt_messages(_TokWithTemplate(), msgs) == "TEMPLATED:hi"


def test_format_prompt_messages_fallback_without_template() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    out = format_prompt_messages(_TokNoTemplate(), msgs)
    assert out == "[user]\nhi\n\n[assistant]\n"


def test_format_prompt_messages_passthrough_non_list() -> None:
    assert format_prompt_messages(_TokNoTemplate(), "already a string") == "already a string"


# --------------------------------------------------------------------------- #
# DPOConfig kwargs
# --------------------------------------------------------------------------- #
def test_build_dpo_config_dict_reference_free_and_beta() -> None:
    cfg = AppConfig().llm.dpo
    out = build_dpo_config_dict(cfg, output_dir="out", use_bf16=True)
    assert out["beta"] == 0.1
    assert out["reference_free"] is False
    assert out["learning_rate"] == pytest.approx(5e-6)
    assert out["bf16"] is True and out["fp16"] is False
    assert out["save_total_limit"] == 2


def test_build_dpo_config_dict_fp16_branch() -> None:
    out = build_dpo_config_dict(AppConfig().llm.dpo, output_dir="out", use_bf16=False)
    assert out["fp16"] is True and out["bf16"] is False


# --------------------------------------------------------------------------- #
# 构造
# --------------------------------------------------------------------------- #
def test_from_config_maps_fields() -> None:
    runner = DPORunner.from_config(
        AppConfig().llm, sft_adapter="models/sft", output_dir="models/dpo_out"
    )
    assert runner.model_name == "Qwen/Qwen3-8B"
    assert runner.sft_adapter == "models/sft"
    assert runner.cfg.beta == 0.1
    assert runner.cache_dir == "models/hf_cache"


def test_dpo_config_dict_uses_output_dir() -> None:
    runner = DPORunner.from_config(AppConfig().llm, sft_adapter="x", output_dir="models/dpo_out")
    out = runner.dpo_config_dict(use_bf16=False)
    assert Path(out["output_dir"]) == Path("models/dpo_out")


def test_import_does_not_require_trl() -> None:
    assert hasattr(dpo, "DPORunner")
