"""
LLM微调训练
"""

import os
from typing import Optional, Dict
from pathlib import Path
from loguru import logger


class FineTuner:
    """LLM微调训练器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "models/llm",
        init_adapter_path: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_seq_length: int = 2048,
        gradient_checkpointing: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            lora_r: LoRA秩
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.init_adapter_path = str(init_adapter_path or "").strip() if init_adapter_path else ""

        self.max_seq_length = max_seq_length
        self.gradient_checkpointing = gradient_checkpointing
        self.load_in_4bit = load_in_4bit
        
        self.lora_config = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        self.model = None
        self.tokenizer = None
    
    def setup(self):
        """初始化模型和tokenizer"""
        logger.info(f"Setting up fine-tuner with model: {self.model_name}")
        
        try:
            import torch
            import inspect
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, PeftModel, get_peft_model
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["low_cpu_mem_usage"] = True

            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

            if self.load_in_4bit:
                from peft import prepare_model_for_kbit_training

                if hasattr(self.model, "config"):
                    self.model.config.use_cache = False

                try:
                    sig = inspect.signature(prepare_model_for_kbit_training)
                    if "use_gradient_checkpointing" in sig.parameters:
                        self.model = prepare_model_for_kbit_training(
                            self.model, use_gradient_checkpointing=self.gradient_checkpointing
                        )
                    else:
                        self.model = prepare_model_for_kbit_training(self.model)
                except Exception:
                    self.model = prepare_model_for_kbit_training(self.model)

            if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            
            # 应用/加载 LoRA
            if self.init_adapter_path:
                logger.info(f"Warm-start: loading init adapter: {self.init_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.init_adapter_path,
                    is_trainable=True,
                )
            else:
                lora_config = LoraConfig(**self.lora_config)
                self.model = get_peft_model(self.model, lora_config)
            
            # 打印可训练参数
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} = {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to setup: {e}")
            raise
    
    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        eval_batch_size: int = 0,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        save_steps: int = 100,
        eval_steps: int = 0,
        eval_max_samples: int = 0,
        save_total_limit: int = 3,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        执行微调训练
        
        Args:
            train_data_path: 训练数据路径
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            gradient_accumulation_steps: 梯度累积步数
            warmup_ratio: 预热比例
            save_steps: 保存间隔
        """
        if self.model is None:
            self.setup()
        
        logger.info(f"Starting fine-tuning with {train_data_path}")
        
        try:
            import torch
            from datasets import load_dataset
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            # 加载数据集
            eval_path = str(eval_data_path) if eval_data_path else None
            if eval_path:
                ds_dict = load_dataset(
                    "json",
                    data_files={"train": train_data_path, "validation": eval_path},
                )
                train_dataset = ds_dict["train"]
                eval_dataset = ds_dict["validation"]
            else:
                train_dataset = load_dataset("json", data_files=train_data_path)["train"]
                eval_dataset = None
            
            # 预处理函数
            def preprocess(examples):
                texts = []
                for conv in examples["conversations"]:
                    text = self.tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                
                return self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False
                )
            
            tokenized_train = train_dataset.map(
                preprocess,
                batched=True,
                remove_columns=train_dataset.column_names
            )

            tokenized_eval = None
            if eval_dataset is not None:
                tokenized_eval = eval_dataset.map(
                    preprocess,
                    batched=True,
                    remove_columns=eval_dataset.column_names
                )

                if int(eval_max_samples) > 0:
                    try:
                        n = len(tokenized_eval)
                        k = min(int(eval_max_samples), int(n))
                        tokenized_eval = tokenized_eval.select(range(k))
                    except Exception:
                        pass
            
            # 训练参数
            use_bf16 = bool(torch.cuda.is_available())
            optim = "paged_adamw_8bit" if self.load_in_4bit else "adamw_torch"

            if int(eval_steps) < 0:
                enable_eval = False
            else:
                enable_eval = tokenized_eval is not None

            if int(eval_steps) == 0:
                effective_eval_steps = save_steps
            else:
                effective_eval_steps = int(eval_steps)

            per_device_eval_bs = int(eval_batch_size) if int(eval_batch_size) > 0 else int(batch_size)
            try:
                training_args = TrainingArguments(
                    output_dir=str(self.output_dir / "checkpoints"),
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=per_device_eval_bs,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    warmup_ratio=warmup_ratio,
                    logging_steps=10,
                    save_steps=save_steps,
                    save_total_limit=save_total_limit,
                    fp16=not use_bf16,
                    bf16=use_bf16,
                    optim=optim,
                    gradient_checkpointing=self.gradient_checkpointing,
                    eval_strategy="steps" if enable_eval else "no",
                    eval_steps=effective_eval_steps if enable_eval else None,
                    report_to="none",
                )
            except ValueError:
                training_args = TrainingArguments(
                    output_dir=str(self.output_dir / "checkpoints"),
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=per_device_eval_bs,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    learning_rate=learning_rate,
                    warmup_ratio=warmup_ratio,
                    logging_steps=10,
                    save_steps=save_steps,
                    save_total_limit=save_total_limit,
                    fp16=not use_bf16,
                    bf16=use_bf16,
                    optim="adamw_torch",
                    gradient_checkpointing=self.gradient_checkpointing,
                    eval_strategy="steps" if enable_eval else "no",
                    eval_steps=effective_eval_steps if enable_eval else None,
                    report_to="none",
                )
            
            # 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # 训练器
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=data_collator,
            )
            
            # 开始训练
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # 保存模型
            self.save()
            
            logger.info("Fine-tuning completed!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save(self, path: Optional[str] = None):
        """保存LoRA权重"""
        save_path = Path(path) if path else self.output_dir / "lora_weights"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str):
        """加载LoRA权重"""
        from peft import PeftModel
        
        if self.model is None:
            self.setup()
        
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"Loaded LoRA weights from {path}")


def run_finetune(
    train_data: str = "data/finetune/train.json",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 3
):
    """运行微调的便捷函数"""
    trainer = FineTuner(model_name=model_name)
    trainer.train(train_data, num_epochs=epochs)
