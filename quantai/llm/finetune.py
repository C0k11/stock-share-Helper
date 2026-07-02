"""quantai.llm.finetune —— QLoRA 监督微调（lazy torch/peft）。

忠实迁移 `src/llm/finetune/train.py::FineTuner`：4-bit NF4 量化基座 + LoRA 适配器训练。
torch/transformers/peft/datasets 全部方法内懒导入。可单测的纯逻辑接缝：
- `build_lora_config_dict` / `build_training_arguments_dict`：构造配置字典（不碰 torch）；
- `build_training_texts`：把 conversations 套 chat 模板成训练文本（用假 tokenizer 可测）。

名词卡（见 modules/finetune.md）：
- **QLoRA** = 4-bit NF4 量化基座（省显存）+ 双量化 + LoRA 低秩适配器 + paged 8bit 优化器。
- **LoRA** = 冻结基座，只训练注入到注意力投影（q/k/v/o）的低秩矩阵 A·B，参数量 <1%。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

#: 训练阶段的 logging_steps（旧码硬编码 10，这里集中一处）。
_LOGGING_STEPS = 10


# --------------------------------------------------------------------------- #
# 纯逻辑接缝（可单测，不需要 torch）
# --------------------------------------------------------------------------- #
def build_lora_config_dict(cfg: Any) -> Dict[str, Any]:
    """从 `LLMFinetuneConfig` 构造 peft.LoraConfig 的 kwargs。"""
    return {
        "r": int(cfg.lora_r),
        "lora_alpha": int(cfg.lora_alpha),
        "lora_dropout": float(cfg.lora_dropout),
        "target_modules": list(cfg.target_modules),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def build_training_arguments_dict(cfg: Any, *, output_dir: str, use_bf16: bool) -> Dict[str, Any]:
    """构造 transformers.TrainingArguments 的 kwargs。

    `use_bf16` 由调用方按 `torch.cuda.is_available()` 传入，便于单测脱离 torch。
    QLoRA(4bit) 用 paged 8bit 优化器省显存；否则普通 adamw_torch。
    """
    optim = "paged_adamw_8bit" if bool(cfg.load_in_4bit) else "adamw_torch"
    return {
        "output_dir": output_dir,
        "num_train_epochs": int(cfg.num_epochs),
        "per_device_train_batch_size": int(cfg.batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "learning_rate": float(cfg.learning_rate),
        "warmup_ratio": float(cfg.warmup_ratio),
        "logging_steps": _LOGGING_STEPS,
        "save_steps": int(cfg.save_steps),
        "save_total_limit": int(cfg.save_total_limit),
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "optim": optim,
        "gradient_checkpointing": bool(cfg.gradient_checkpointing),
        "report_to": "none",
    }


def build_training_texts(tokenizer: Any, conversations: List[Any]) -> List[str]:
    """把每条 conversation 套 chat 模板成训练文本（忠实迁移 train.py::preprocess）。"""
    texts: List[str] = []
    for conv in conversations:
        texts.append(
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        )
    return texts


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class QLoRAFineTuner:
    """QLoRA/LoRA 监督微调器。重活懒加载；纯逻辑接缝可单测。"""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-8B",
        output_dir: str = "models/llm",
        cfg: Any,
        cache_dir: str = "models/hf_cache",
        init_adapter_path: Optional[str] = None,
    ) -> None:
        self.model_name = str(model_name)
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.cache_dir = str(cache_dir)
        self.init_adapter_path = str(init_adapter_path or "").strip()
        self.model: Any = None
        self.tokenizer: Any = None

    @classmethod
    def from_config(
        cls, app_cfg: Any, *, output_dir: str = "models/llm", init_adapter_path: Optional[str] = None
    ) -> "QLoRAFineTuner":
        return cls(
            model_name=app_cfg.model_name,
            output_dir=output_dir,
            cfg=app_cfg.finetune,
            cache_dir=app_cfg.cache_dir,
            init_adapter_path=init_adapter_path,
        )

    def attach(self, model: Any, tokenizer: Any) -> "QLoRAFineTuner":
        """注入已构造好的 model/tokenizer（绕过 torch 加载）。供测试与高级用法。"""
        self.model = model
        self.tokenizer = tokenizer
        return self

    def lora_config_dict(self) -> Dict[str, Any]:
        return build_lora_config_dict(self.cfg)

    # ----------------------------------------------------------------- #
    # 重活（lazy torch/transformers/peft）
    # ----------------------------------------------------------------- #
    def setup(self) -> None:
        """加载基座（可选 4bit）+ 应用/加载 LoRA。忠实迁移 train.py::setup。"""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["low_cpu_mem_usage"] = True

        if bool(self.cfg.load_in_4bit):
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if bool(self.cfg.load_in_4bit):
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=bool(self.cfg.gradient_checkpointing)
            )

        if self.cfg.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if self.init_adapter_path:
            self.model = PeftModel.from_pretrained(self.model, self.init_adapter_path, is_trainable=True)
        else:
            self.model = get_peft_model(self.model, LoraConfig(**self.lora_config_dict()))

    def train(
        self,
        train_data_path: str,
        *,
        eval_data_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> str:
        """执行 SFT 训练并保存 LoRA 权重。忠实迁移 train.py::train。返回保存路径。"""
        if self.model is None:
            self.setup()

        import torch
        from datasets import load_dataset
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        if eval_data_path:
            ds = load_dataset("json", data_files={"train": train_data_path, "validation": eval_data_path})
            train_dataset, eval_dataset = ds["train"], ds["validation"]
        else:
            train_dataset = load_dataset("json", data_files=train_data_path)["train"]
            eval_dataset = None

        def _preprocess(examples: Dict[str, Any]) -> Any:
            texts = build_training_texts(self.tokenizer, examples["conversations"])
            return self.tokenizer(
                texts, truncation=True, max_length=int(self.cfg.max_seq_length), padding=False
            )

        tokenized_train = train_dataset.map(
            _preprocess, batched=True, remove_columns=train_dataset.column_names
        )
        tokenized_eval = (
            eval_dataset.map(_preprocess, batched=True, remove_columns=eval_dataset.column_names)
            if eval_dataset is not None
            else None
        )

        use_bf16 = bool(torch.cuda.is_available())
        args_dict = build_training_arguments_dict(
            self.cfg, output_dir=str(self.output_dir / "checkpoints"), use_bf16=use_bf16
        )
        args_dict["eval_strategy"] = "steps" if tokenized_eval is not None else "no"
        args_dict["eval_steps"] = int(self.cfg.save_steps) if tokenized_eval is not None else None

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**args_dict),
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return self.save()

    def save(self, path: Optional[str] = None) -> str:
        """保存 LoRA 权重 + tokenizer。返回保存目录。"""
        save_path = Path(path) if path else self.output_dir / "lora_weights"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        return str(save_path)

    def load(self, path: str) -> None:
        """在已 setup 的基座上加载 LoRA 权重。"""
        from peft import PeftModel

        if self.model is None:
            self.setup()
        self.model = PeftModel.from_pretrained(self.model, path)
