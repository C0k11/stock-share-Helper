#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.llm_tools import extract_json_text, repair_and_parse_json


def resolve_adapter_path(p: str) -> str:
    base = Path(p)
    if base.is_file():
        base = base.parent

    direct = base
    if (direct / "adapter_config.json").exists():
        return str(direct)

    lw = base / "lora_weights"
    if (lw / "adapter_config.json").exists():
        return str(lw)

    ckpt_root = base / "checkpoints"
    if ckpt_root.exists():
        candidates: List[Tuple[int, Path]] = []
        for cp in ckpt_root.glob("checkpoint-*"):
            name = cp.name
            if not name.startswith("checkpoint-"):
                continue
            try:
                step = int(name.split("checkpoint-")[-1])
            except Exception:
                continue
            if (cp / "adapter_config.json").exists():
                candidates.append((step, cp))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return str(candidates[-1][1])

    raise SystemExit(
        f"LoRA adapter not found under: {p}. Expected adapter_config.json in <path>, <path>/lora_weights, or <path>/checkpoints/checkpoint-*"
    )


def build_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a professional quantitative trader and risk manager. "
        "You will be given a JSON input containing market features and optional RAG context. "
        "Return STRICT JSON only (no markdown, no extra text). "
        "The JSON must contain keys: action, target_position, confidence, risk_notes, rationale. "
        "action must be one of: buy, add, hold, reduce, clear. "
        "target_position must be a number between 0 and 1. "
        "confidence must be a number between 0 and 1. "
        "risk_notes must be an array of short strings."
    )

    user = json.dumps(payload, ensure_ascii=False, indent=2)
    user = f"Input JSON:\n{user}\n\nOutput JSON:" 

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_model(base_model_id: str, adapter_path: str, load_4bit: bool) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["low_cpu_mem_usage"] = True

    if load_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    adapter_path = resolve_adapter_path(adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def run_once(model, tokenizer, payload: Dict[str, Any], max_new_tokens: int, temperature: float) -> Tuple[str, Optional[Dict[str, Any]], str]:
    messages = build_messages(payload)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    do_sample = float(temperature) > 0
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=do_sample,
            top_p=0.9,
            repetition_penalty=1.05,
        )

    gen_ids = out_ids[0][inputs.input_ids.shape[1] :]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    parsed: Optional[Dict[str, Any]] = None
    parse_error = ""
    try:
        js = extract_json_text(raw_text.strip())
        if js is None:
            raise ValueError("no json found in model output")
        obj = repair_and_parse_json(raw_text.strip())
        if not isinstance(obj, dict):
            raise ValueError("model output json is not object")
        parsed = obj
    except Exception as e:
        parsed = None
        parse_error = str(e)

    return raw_text, parsed, parse_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: load Trader LoRA and run a single inference")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora", default="models/llm_etf_trading_qwen25_7b_rag_final")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    parser.set_defaults(load_4bit=True)
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "date": "2024-12-18",
        "ticker": "SPY",
        "features": {
            "trend_score": 0.8,
            "volatility": "high",
            "news_sentiment": 0.6,
        },
        "rag_context": "Historical Setup: Similar high volatility bullish breakout occurred on 2023-10-25...",
    }

    base_model = str(args.base).replace("\\", "/")
    lora_path = str(args.lora)

    print(f"Loading Base Model: {base_model}...")
    print(f"Loading LoRA Adapter: {lora_path}...")
    model, tokenizer = load_model(base_model, lora_path, bool(args.load_4bit))

    print("Generating...")
    raw_text, parsed, parse_error = run_once(
        model,
        tokenizer,
        payload,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
    )

    print("=" * 80)
    print("RAW_OUTPUT:")
    print(raw_text)
    print("=" * 80)

    if parsed is not None:
        print("JSON_PARSE_OK: True")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        raise SystemExit(0)

    print("JSON_PARSE_OK: False")
    print("JSON_ERROR: " + str(parse_error))
    raise SystemExit(1)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
