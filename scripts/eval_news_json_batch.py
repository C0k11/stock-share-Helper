#!/usr/bin/env python
"""Batch eval for news->strict JSON.

This script loads the base model once and evaluates multiple news items from a JSON list.

Example:
  .\venv311\Scripts\python.exe scripts\eval_news_json_batch.py \
    --model Qwen/Qwen2.5-14B-Instruct --use-lora --lora models\\xxx\\lora_weights --load-in-4bit --pod \
    --cases scripts\\eval_inputs\\news_test_cases_5.json --out scripts\\eval_outputs\\A.json
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_messages(title: str, content: str, pod_style: bool) -> list:
    if pod_style:
        system = (
            "You are Pod 042 from NieR:Automata: calm, concise, logically rigorous. "
            "You must output STRICT JSON only (no markdown, no prose outside JSON). "
            "The JSON values must still be plain text."
        )
    else:
        system = (
            "You are a professional financial news analyst. "
            "You must output STRICT JSON only (no markdown, no prose outside JSON)."
        )

    schema = (
        "Output exactly ONE JSON object with these fields: "
        "event_type (string), sentiment (string), impact_equity (-1/0/1), impact_bond (-1/0/1), impact_gold (-1/0/1), summary (string)."
    )

    user = (
        f"Title: {title}\n"
        f"Content: {content}\n\n"
        f"{schema}\n"
        "Do not include any additional keys."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_first_json(text: str) -> str:
    if not text:
        raise ValueError("Empty model output")

    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' found in output")

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
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces in output")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("cases file must be a JSON list")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "title" not in item or "content" not in item:
            raise ValueError(f"case[{i}] must be an object with title/content")
    return data


def validate_schema(obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    required = ["event_type", "sentiment", "impact_equity", "impact_bond", "impact_gold", "summary"]
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj.keys() if k not in required]
    return missing, extra


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate news->strict JSON output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--lora", default="models/llm/lora_weights", help="LoRA weights path")
    parser.add_argument("--use-lora", action="store_true", help="Load LoRA weights")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--pod", action="store_true", help="Use Pod 042 style system prompt (still JSON-only)")
    parser.add_argument("--cases", required=True, help="Path to JSON list of cases")
    parser.add_argument("--out", default="", help="Optional output path to save per-case results")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    cases = load_cases(cases_path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
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

    if args.use_lora:
        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        logger.info(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    results: List[Dict[str, Any]] = []
    parse_ok = 0
    schema_ok = 0

    for idx, item in enumerate(cases):
        cid = item.get("id") or f"case_{idx+1}"
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()

        messages = build_messages(title=title, content=content, pod_style=bool(args.pod))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        decoded = ""
        parsed_obj = None
        missing: List[str] = []
        extra: List[str] = []
        error = ""

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt) :]
            decoded = decoded.strip()

            json_text = extract_first_json(decoded)
            parsed_obj = json.loads(json_text)
            parse_ok += 1

            if isinstance(parsed_obj, dict):
                missing, extra = validate_schema(parsed_obj)
                if not missing and not extra:
                    schema_ok += 1
        except Exception as e:
            error = str(e)

        results.append(
            {
                "id": cid,
                "title": title,
                "parsed": parsed_obj,
                "missing": missing,
                "extra": extra,
                "error": error,
                "raw": decoded,
            }
        )

        status = "OK" if parsed_obj is not None else "FAIL"
        logger.info(f"[{idx+1}/{len(cases)}] {cid}: {status} missing={missing} extra={extra}")

    summary = {
        "cases": len(cases),
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "parse_rate": round(parse_ok / max(1, len(cases)), 3),
        "schema_rate": round(schema_ok / max(1, len(cases)), 3),
        "model": args.model,
        "lora": str(args.lora) if args.use_lora else "",
        "pod": bool(args.pod),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
    }

    payload = {"summary": summary, "results": results}

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {out_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
