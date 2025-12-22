import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
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


def main() -> None:
    _ensure_project_root_on_path()

    p = argparse.ArgumentParser(description="Phase 15.3: Surgical DPO Training (Merge V3 + Polish V4 LoRA)")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--prev-adapter", default="models/trader_v3_dpo_analyst_ultimate")
    p.add_argument("--data-path", default="data/dpo/v4_train.jsonl")
    p.add_argument("--output-dir", default="models/trader_v4_dpo_analyst_alpha")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-length", type=int, default=3072)

    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    p.add_argument("--reference-free", action="store_true", default=False)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=1000000)

    args = p.parse_args()

    base_model = str(args.base_model)
    prev_adapter = str(args.prev_adapter)
    data_path = str(args.data_path)
    output_dir = str(args.output_dir)

    if not os.path.exists(prev_adapter):
        raise SystemExit(f"prev adapter not found: {prev_adapter}")
    if not os.path.exists(data_path):
        raise SystemExit(f"data not found: {data_path}")

    # --- Merge (V3 -> Base) ---
    print(f"Loading Base Model (16-bit) for merge: {base_model}")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading previous adapter to merge: {prev_adapter}")
    model = PeftModel.from_pretrained(model, prev_adapter)
    model = model.merge_and_unload()
    print("Previous adapter merged successfully. Model is now V3 Full.")

    # --- Polish (new V4 LoRA) ---
    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=[str(x) for x in args.lora_target_modules],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    print(f"Loading Dataset: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

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

    # DPOConfig matches the existing project script (scripts/train_dpo.py)
    fp16 = bool(torch.cuda.is_available())
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        fp16=fp16,
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=1,
        beta=float(args.beta),
        max_prompt_length=int(args.max_prompt_length),
        max_length=int(args.max_length),
        reference_free=bool(args.reference_free),
        report_to="none",
    )

    print(f"Initializing DPOTrainer (reference_free={args.reference_free})...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Surgical DPO Training...")
    trainer.train()

    print(f"Saving V4 adapter to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
