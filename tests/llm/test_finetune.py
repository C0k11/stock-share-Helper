"""quantai.llm.finetune 测试（纯逻辑接缝 + save/attach，不下真模型）。"""

from __future__ import annotations

from pathlib import Path

from quantai.config import AppConfig
from quantai.llm import finetune
from quantai.llm.finetune import (
    QLoRAFineTuner,
    build_lora_config_dict,
    build_training_arguments_dict,
    build_training_texts,
)


# --------------------------------------------------------------------------- #
# 纯逻辑接缝
# --------------------------------------------------------------------------- #
def test_build_lora_config_dict() -> None:
    cfg = AppConfig().llm.finetune
    out = build_lora_config_dict(cfg)
    assert out["r"] == 16
    assert out["lora_alpha"] == 32
    assert out["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert out["task_type"] == "CAUSAL_LM"
    assert out["bias"] == "none"


def test_build_training_arguments_bf16_branch() -> None:
    cfg = AppConfig().llm.finetune
    out = build_training_arguments_dict(cfg, output_dir="out", use_bf16=True)
    assert out["bf16"] is True and out["fp16"] is False
    assert out["num_train_epochs"] == 3
    assert out["report_to"] == "none"
    assert out["optim"] == "adamw_torch"  # load_in_4bit 默认 False


def test_build_training_arguments_fp16_branch() -> None:
    cfg = AppConfig().llm.finetune
    out = build_training_arguments_dict(cfg, output_dir="out", use_bf16=False)
    assert out["fp16"] is True and out["bf16"] is False


def test_build_training_arguments_paged_optim_when_4bit() -> None:
    cfg = AppConfig().llm.finetune
    cfg.load_in_4bit = True
    out = build_training_arguments_dict(cfg, output_dir="out", use_bf16=True)
    assert out["optim"] == "paged_adamw_8bit"  # QLoRA 用 paged 8bit 优化器


class _FakeTokenizer:
    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False):
        return "T:" + "|".join(m["content"] for m in conv)


def test_build_training_texts() -> None:
    tok = _FakeTokenizer()
    convs = [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        [{"role": "user", "content": "c"}],
    ]
    assert build_training_texts(tok, convs) == ["T:a|b", "T:c"]


# --------------------------------------------------------------------------- #
# 训练器：from_config / save / attach
# --------------------------------------------------------------------------- #
def test_from_config_maps_fields() -> None:
    ft = QLoRAFineTuner.from_config(AppConfig().llm, output_dir="models/x")
    assert ft.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert ft.cfg.lora_r == 16
    assert ft.cache_dir == "models/hf_cache"


class _SaveSpy:
    def __init__(self) -> None:
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = str(path)


def test_save_writes_model_and_tokenizer(tmp_path: Path) -> None:
    model, tok = _SaveSpy(), _SaveSpy()
    ft = QLoRAFineTuner(cfg=AppConfig().llm.finetune, output_dir=str(tmp_path))
    ft.attach(model, tok)
    out = ft.save()
    assert Path(out) == tmp_path / "lora_weights"
    assert model.saved_to == str(tmp_path / "lora_weights")
    assert tok.saved_to == str(tmp_path / "lora_weights")
    assert (tmp_path / "lora_weights").exists()


def test_save_custom_path(tmp_path: Path) -> None:
    model, tok = _SaveSpy(), _SaveSpy()
    ft = QLoRAFineTuner(cfg=AppConfig().llm.finetune, output_dir=str(tmp_path))
    ft.attach(model, tok)
    custom = tmp_path / "custom"
    out = ft.save(str(custom))
    assert Path(out) == custom
    assert model.saved_to == str(custom)


def test_import_does_not_require_torch() -> None:
    assert hasattr(finetune, "QLoRAFineTuner")
