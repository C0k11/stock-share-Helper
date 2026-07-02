"""quantai.llm.dpo —— DPO 偏好对齐训练（lazy trl）。

忠实迁移 `scripts/train_dpo.py`：在 4-bit 基座 + SFT/Analyst LoRA 之上，用偏好对
（prompt / chosen / rejected）做 DPO 对齐。torch/transformers/peft/trl/datasets 全部懒导入。

名词卡（见 modules/dpo.md）：
- **DPO（Direct Preference Optimization）** = 用「同一 prompt 下 chosen 优于 rejected」的偏好对
  直接优化策略，使 chosen 概率相对参考模型升高、rejected 降低。
- **相比 PPO/RLHF**：DPO **不需要单独训练 reward model、也不在线采样**，把对齐变成一个
  监督式的成对损失，工程上更省、更稳。
- **reference_free**：跳过参考模型项（dry-run/冒烟用；正式对齐应保留参考模型）。

诚实说明（接线现状）：在线自进化闭环 `src/rl/online_learning.py` 目前**只收集**经验并
用 `PreferenceLogger` 落地 chosen/rejected 偏好对，其 `OnlineLearningManager._trigger_update`
带着 `# TODO: Implement actual gradient updates`——**真正的 DPO 训练是离线的本模块**
（旧 `scripts/train_dpo.py`），由 API/夜间任务以子进程触发，而非在线梯度更新。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompts import format_messages_fallback

#: DPO 数据集必需列。
REQUIRED_DPO_COLUMNS = ("prompt", "chosen", "rejected")
#: 旧码硬编码：DPO 检查点最多保留 2 份。
_SAVE_TOTAL_LIMIT = 2


# --------------------------------------------------------------------------- #
# 纯逻辑接缝（可单测，不需要 torch/trl）
# --------------------------------------------------------------------------- #
def has_chat_template(tokenizer: Any) -> bool:
    """tokenizer 是否带 chat_template（忠实迁移 train_dpo.py::_has_chat_template）。"""
    try:
        return bool(getattr(tokenizer, "chat_template", None))
    except Exception:
        return False


def format_prompt_messages(tokenizer: Any, prompt_messages: Any) -> Any:
    """把 prompt（消息列表）格式化成字符串：有 chat_template 用之，否则纯文本回退。

    忠实迁移 train_dpo.py::_format_prompt_messages；非 list 原样返回。
    """
    if not isinstance(prompt_messages, list):
        return prompt_messages
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    return format_messages_fallback(prompt_messages)


def validate_dpo_columns(columns: Any) -> None:
    """校验数据集含 prompt/chosen/rejected，否则抛 ValueError（忠实迁移 train_dpo.py）。"""
    missing = set(REQUIRED_DPO_COLUMNS).difference(set(columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")


def build_dpo_config_dict(cfg: Any, *, output_dir: str, use_bf16: bool) -> Dict[str, Any]:
    """构造 trl.DPOConfig 的 kwargs（`use_bf16` 由调用方传入，便于单测脱离 torch）。"""
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(cfg.batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "num_train_epochs": int(cfg.num_epochs),
        "learning_rate": float(cfg.learning_rate),
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": int(cfg.logging_steps),
        "save_steps": int(cfg.save_steps),
        "save_total_limit": _SAVE_TOTAL_LIMIT,
        "beta": float(cfg.beta),
        "max_prompt_length": int(cfg.max_prompt_length),
        "max_length": int(cfg.max_length),
        "reference_free": bool(cfg.reference_free),
        "report_to": "none",
    }


# --------------------------------------------------------------------------- #
# DPO 训练器
# --------------------------------------------------------------------------- #
class DPORunner:
    """在 4bit 基座 + SFT adapter 上跑 DPO 对齐。重活懒加载；纯逻辑接缝可单测。"""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-8B",
        sft_adapter: str,
        output_dir: str,
        cfg: Any,
        cache_dir: str = "models/hf_cache",
    ) -> None:
        self.model_name = str(model_name)
        self.sft_adapter = str(sft_adapter)
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.cache_dir = str(cache_dir)
        self.model: Any = None
        self.tokenizer: Any = None

    @classmethod
    def from_config(cls, app_cfg: Any, *, sft_adapter: str, output_dir: str) -> "DPORunner":
        return cls(
            model_name=app_cfg.model_name,
            sft_adapter=sft_adapter,
            output_dir=output_dir,
            cfg=app_cfg.dpo,
            cache_dir=app_cfg.cache_dir,
        )

    def attach(self, model: Any, tokenizer: Any) -> "DPORunner":
        """注入已构造好的 model/tokenizer（绕过 torch 加载）。供测试与高级用法。"""
        self.model = model
        self.tokenizer = tokenizer
        return self

    def dpo_config_dict(self, *, use_bf16: bool) -> Dict[str, Any]:
        return build_dpo_config_dict(self.cfg, output_dir=str(self.output_dir), use_bf16=use_bf16)

    # ----------------------------------------------------------------- #
    # 重活（lazy torch/transformers/peft）
    # ----------------------------------------------------------------- #
    def setup(self) -> None:
        """加载 4-bit 基座 + SFT adapter（is_trainable=True）。忠实迁移 train_dpo.py。"""
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = PeftModel.from_pretrained(self.model, self.sft_adapter, is_trainable=True)

    def train(self, data_path: str) -> str:
        """跑 DPO 训练并保存 adapter。忠实迁移 train_dpo.py。返回输出目录。"""
        if self.model is None:
            self.setup()

        import torch
        from datasets import load_dataset
        from trl import DPOConfig, DPOTrainer

        dataset = load_dataset("json", data_files=data_path, split="train")
        validate_dpo_columns(dataset.column_names)

        def _map_row(ex: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "prompt": format_prompt_messages(self.tokenizer, ex["prompt"]),
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            }

        keep = set(REQUIRED_DPO_COLUMNS)
        dataset = dataset.map(
            _map_row, remove_columns=[c for c in dataset.column_names if c not in keep]
        )

        use_bf16 = bool(torch.cuda.is_available())
        trainer = DPOTrainer(
            model=self.model,
            args=DPOConfig(**self.dpo_config_dict(use_bf16=use_bf16)),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        return str(self.output_dir)
