#!/usr/bin/env python
"""LLM推理脚本：基础模型 + LoRA权重

示例：
  .\\venv311\\Scripts\\python.exe scripts\\infer_llm.py --task news --use-lora
  .\\venv311\\Scripts\\python.exe scripts\\infer_llm.py --task explain --use-lora

注意：LoRA权重默认读取 models/llm/lora_weights（该目录在.gitignore中，不会被提交）。
"""

import sys
from pathlib import Path
import argparse
from loguru import logger


# 添加项目根目录到路径（确保可直接运行脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_messages(task: str) -> list:
    if task == "news":
        system = "You are a professional financial news analyst. Read the news and output a structured JSON."
        user = (
            "Title: Fed signals rates may stay higher for longer\n"
            "Content: In the latest minutes, policymakers emphasized inflation risks and kept a restrictive stance.\n\n"
            "Output JSON with fields: event_type, sentiment, impact_equity(-1/0/1), impact_bond(-1/0/1), impact_gold(-1/0/1), summary."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if task == "explain":
        system = "You are an investment assistant. Explain the recommendation clearly and concisely."
        user = (
            "Market Regime: Risk-Off, VIX: 26, SPY below 200DMA.\n"
            "Signals: trend=-1, momentum=-1.\n"
            "Recommendation: reduce SPY allocation to 30%, increase TLT/GLD.\n\n"
            "Please explain the reasoning and risk notes for a retail investor."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # generic chat
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


def main():
    parser = argparse.ArgumentParser(description="Infer with base model and optional LoRA")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--lora", default="models/llm/lora_weights", help="LoRA weights path")
    parser.add_argument("--use-lora", action="store_true", help="Load LoRA weights")
    parser.add_argument("--task", choices=["news", "explain", "chat"], default="news")
    parser.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.use_lora:
        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        logger.info(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    messages = build_messages(args.task)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=" * 80)
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()
