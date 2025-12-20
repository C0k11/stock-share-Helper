import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


def _ensure_project_root_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _has_chat_template(tokenizer) -> bool:
    try:
        return bool(getattr(tokenizer, "chat_template", None))
    except Exception:
        return False


def _format_prompt_messages(tokenizer, prompt_messages):
    if not isinstance(prompt_messages, list):
        return prompt_messages

    if _has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    chunks = []
    for m in prompt_messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        chunks.append(f"[{role}]\n{content}")
    chunks.append("[assistant]\n")
    return "\n\n".join(chunks)


def train_dpo() -> None:
    _ensure_project_root_on_path()

    parser = argparse.ArgumentParser(description="Phase 12: Train DPO Adapter for Trader")
    parser.add_argument("--base-model", type=str, required=True, help="Path/name to base model (e.g. Qwen)")
    parser.add_argument("--sft-adapter", type=str, required=True, help="Path to existing SFT/Analyst LoRA adapter")
    parser.add_argument("--data-path", type=str, required=True, help="Path to .jsonl dataset from build_dpo_pairs.py")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the new DPO adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6, help="Lower LR for DPO stability")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--reference-free", action="store_true", help="Skip reference model (useful for dry-run/smoke)")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)

    args = parser.parse_args()

    print(f"Loading Base Model: {args.base_model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading SFT Adapter from: {args.sft_adapter}...")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()

    print(f"Loading Dataset: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    required_cols = {"prompt", "chosen", "rejected"}
    missing = required_cols.difference(set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    def _map_row(ex):
        return {
            "prompt": _format_prompt_messages(tokenizer, ex["prompt"]),
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    dataset = dataset.map(_map_row, remove_columns=[c for c in dataset.column_names if c not in required_cols])

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        reference_free=args.reference_free,
        report_to="none",
    )

    print(f"Initializing DPOTrainer (reference_free={args.reference_free})...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print("DPOTrainer initialized.")

    print("Starting DPO Training...")
    trainer.train()

    print(f"Saving DPO Adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    train_dpo()
