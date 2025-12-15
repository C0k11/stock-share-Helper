#!/usr/bin/env python
"""LLM推理脚本：基础模型 + LoRA权重

示例：
  .\\venv311\\Scripts\\python.exe scripts\\infer_llm.py --task news --use-lora
  .\\venv311\\Scripts\\python.exe scripts\\infer_llm.py --task explain --use-lora

注意：LoRA权重默认读取 models/llm/lora_weights（该目录在.gitignore中，不会被提交）。
"""

import sys
import json
from pathlib import Path
import argparse
from loguru import logger


# 添加项目根目录到路径（确保可直接运行脚本）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def sanitize_input_text(s: str) -> str:
    return (s or "").replace('"', "'")


CN_EVENT_TYPES = [
    "policy_stimulus",
    "regulation_crackdown",
    "market_intervention",
    "concept_hype",
    "corporate_restructuring",
]


US_EVENT_TYPES = [
    "fomc_decision",
    "inflation_data",
    "jobs_report",
    "gdp_data",
    "fiscal_tariff",
    "corporate_earnings",
    "geopolitical_risk",
    "financial_stability",
    "commodity_shock",
    "other_us",
]


def build_messages(
    task: str,
    title: str | None = None,
    content: str | None = None,
    *,
    market: str = "us",
) -> list:
    if task == "news":
        market = (market or "us").strip().lower()
        if market == "cn":
            system = (
                "You are an expert A-Share (Chinese Stock Market) trader. You understand hidden signals and A-share jargon. "
                "You must output STRICT JSON only (no markdown, no prose outside JSON). "
                "Output valid JSON only. Do not use double quotes (\") within string values; use single quotes (') instead. "
                "You MUST choose event_type from this enum: "
                f"{CN_EVENT_TYPES}. "
                "You MUST follow these hard rules: "
                "- If event_type is regulation_crackdown, impact_equity MUST be -1. "
                "- If event_type is market_intervention, impact_equity MUST be 1. "
                "- If event_type is corporate_restructuring, impact_equity MUST be 1. "
                "- If event_type is policy_stimulus, impact_equity MUST be 1 and impact_bond MUST be 1. "
                "- If event_type is NOT policy_stimulus, impact_bond MUST be 0. "
                "- If event_type is concept_hype, impact_equity MUST be 1 and summary MUST mention speculative/short-term nature."
            )
        else:
            system = (
                "You are a professional US financial news analyst. "
                "You must output STRICT JSON only (no markdown, no prose outside JSON). "
                "Output valid JSON only. Do not use double quotes (\") within string values; use single quotes (') instead. "
                "This is a US market task. "
                "You MUST choose event_type from this US-only enum: "
                f"{US_EVENT_TYPES}. "
                "Do NOT output any China/A-share event types (forbidden examples: policy_stimulus, regulation_crackdown, market_intervention, concept_hype, corporate_restructuring). "
                "If you are unsure, use other_us."
            )
        if title is None:
            title = "Fed signals rates may stay higher for longer"
        if content is None:
            content = "In the latest minutes, policymakers emphasized inflation risks and kept a restrictive stance."
        user = (
            f"Title: {title}\n"
            f"Content: {content}\n\n"
            "Output JSON with fields: event_type, sentiment, impact_equity(-1/0/1), impact_bond(-1/0/1), impact_gold(-1/0/1), summary."
        )
        user = sanitize_input_text(user)
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


def _extract_first_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _try_parse_json(text: str) -> tuple[bool, str | None]:
    js = _extract_first_json_object(text)
    if js is None:
        return False, None
    try:
        json.loads(js)
        return True, js
    except Exception:
        return False, js


def _generate_once(model, tokenizer, messages: list, max_new_tokens: int, deterministic: bool):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not deterministic,
    }
    if not deterministic:
        gen_kwargs.update({"temperature": 0.7, "top_p": 0.9})

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    out_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return out_text


def main():
    parser = argparse.ArgumentParser(description="Infer with base model and optional LoRA")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--lora", default="models/llm/lora_weights", help="LoRA weights path")
    parser.add_argument("--use-lora", action="store_true", help="Load LoRA weights")
    parser.add_argument("--compare-lora", action="store_true", help="Run base vs LoRA comparison in one process")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit (recommended for 14B + LoRA)")
    parser.add_argument("--task", choices=["news", "explain", "chat"], default="news")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--title", default=None)
    parser.add_argument("--content", default=None)
    parser.add_argument("--cases", default=None, help="Path to JSON list: [{title, content, id?, tag?}, ...]")
    parser.add_argument("--offset", type=int, default=0, help="Offset into cases list loaded from --cases")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases loaded from --cases (0 = no limit)")

    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    def run_cases(current_model, label: str):
        if args.cases:
            cases_path = Path(args.cases)
            if not cases_path.exists():
                raise FileNotFoundError(f"Cases file not found: {cases_path}")
            cases = json.load(open(cases_path, "r", encoding="utf-8"))
            if not isinstance(cases, list):
                raise ValueError("--cases must be a JSON list")
            if args.offset and args.offset > 0:
                cases = cases[args.offset :]
            if args.limit and args.limit > 0:
                cases = cases[: args.limit]
        else:
            cases = [{"title": args.title, "content": args.content, "id": "single"}]

        print("=" * 80)
        print(f"RUN={label} task={args.task} deterministic={args.deterministic} cases={len(cases)}")

        ok_cnt = 0
        for idx, c in enumerate(cases, start=1):
            title = c.get("title")
            content = c.get("content")
            case_id = c.get("id", str(idx))
            tag = c.get("tag", "")
            market = c.get("market") or "us"
            messages = build_messages(args.task, title=title, content=content, market=market)
            out_text = _generate_once(
                current_model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                deterministic=args.deterministic,
            )

            ok, js = _try_parse_json(out_text)
            ok_cnt += 1 if ok else 0

            print("-" * 80)
            print(f"CASE {idx}/{len(cases)} id={case_id} market={market} tag={tag}")
            if title:
                print(f"TITLE: {title}")
            print(f"JSON_PARSE_OK={ok}")
            if js is not None:
                print(js)
            else:
                print(out_text.strip())

        print("-" * 80)
        print(f"JSON_PARSE_OK_RATE={ok_cnt}/{len(cases)}")
        print("=" * 80)

    if args.compare_lora:
        run_cases(model, label="BASE")

        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        logger.info(f"Loading LoRA weights: {lora_path}")
        lora_model = PeftModel.from_pretrained(model, str(lora_path))
        run_cases(lora_model, label="LORA")
        return

    if args.use_lora:
        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        logger.info(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    run_cases(model, label="SINGLE")


if __name__ == "__main__":
    main()
