#!/usr/bin/env python

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_tools import extract_json_text, repair_and_parse_json


def _as_role_content(msg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(msg, dict):
        return None

    if "role" in msg and "content" in msg:
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "")
        if not role:
            return None
        return {"role": role, "content": content}

    if "from" in msg and "value" in msg:
        role = str(msg.get("from") or "").strip()
        content = str(msg.get("value") or "")
        if not role:
            return None
        if role == "human":
            role = "user"
        return {"role": role, "content": content}

    return None


def _extract_ground_truth_decision(item: Dict[str, Any]) -> str:
    conv = item.get("conversations")
    if not isinstance(conv, list) or len(conv) < 3:
        return "UNKNOWN"

    msg = _as_role_content(conv[-1])
    if msg is None:
        return "UNKNOWN"

    raw = str(msg.get("content") or "")
    obj = repair_and_parse_json(raw)
    if isinstance(obj, dict):
        dec = str(obj.get("decision") or "").strip().upper()
        if dec in ("BUY", "SELL", "HOLD"):
            return dec
    return "UNKNOWN"


def _build_prompt(tokenizer: Any, item: Dict[str, Any]) -> str:
    force_json = bool(item.get("_force_json"))

    conv = item.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        raise ValueError("Invalid conversations")

    sys_msg = _as_role_content(conv[0])
    user_msg = _as_role_content(conv[1])

    if sys_msg is None or user_msg is None:
        raise ValueError("Unsupported conversation message format")

    system_text = sys_msg["content"]
    if force_json:
        system_text = (
            f"{system_text}\n\n"
            "Hard requirement: output STRICT JSON only (no markdown, no prose outside JSON). "
            "Output exactly ONE JSON object. "
            "The JSON MUST include key decision with value BUY/SELL/HOLD. "
            "The JSON SHOULD include key analysis (short). "
            "ticker is optional."
        )

    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_msg["content"]}]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse_pred_decision(text: str) -> Tuple[bool, str]:
    raw = extract_json_text(text)
    if raw is None:
        t = text.upper()
        patterns = [
            r"\bDECISION\b\s*[:=\-]\s*(BUY|SELL|HOLD)\b",
            r"\bTRADING\s+DECISION\b\s*[:=\-]\s*(BUY|SELL|HOLD)\b",
            r"\bSIGNAL\b\s*[:=\-]\s*(BUY|SELL|HOLD)\b",
            r"\b(BUY|SELL|HOLD)\b",
        ]
        for p in patterns:
            ms = re.findall(p, t)
            if ms:
                return False, ms[-1]
        return False, "ERROR"

    obj = repair_and_parse_json(raw)
    if not isinstance(obj, dict):
        t = text.upper()
        ms = re.findall(r"\b(BUY|SELL|HOLD)\b", t)
        if ms:
            return False, ms[-1]
        return False, "ERROR"

    dec = str(obj.get("decision") or "").strip().upper()
    if dec in ("BUY", "SELL", "HOLD"):
        return True, dec

    return True, "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Trader Stock LoRA: JSON validity, accuracy, and prediction distribution")
    parser.add_argument("--data", default="data/finetune/trader_stock_sft_v1.json")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora", default="models/trader_stock_v1_tech_only/lora_weights")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--force-json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-invalid", type=int, default=3)

    args = parser.parse_args()

    data_path = Path(args.data)
    lora_path = Path(args.lora)

    print(f"Loading dataset: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Data must be a JSON list")

    rng = random.Random(int(args.seed))
    n = min(int(args.samples), len(data))
    sample_items = rng.sample(data, n)

    if args.force_json:
        for it in sample_items:
            if isinstance(it, dict):
                it["_force_json"] = True

    print(f"Loading model: base={args.base_model} lora={lora_path}")
    if not lora_path.exists():
        raise SystemExit(f"LoRA weights not found: {lora_path} (training may not be finished)")

    # Prefer tokenizer/chat_template saved together with LoRA weights.
    # This avoids mismatches where the base model's default chat template causes the model
    # to answer in prose instead of the SFT JSON format.
    tokenizer_src = str(lora_path) if (lora_path / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    tokenizer.padding_side = "left"
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

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()

    valid_json = 0
    correct = 0
    correct_json_only = 0
    decision_found = 0

    dist: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0, "OTHER": 0, "ERROR": 0}

    gt_hold = 0
    gt_known = 0

    shown = 0
    shown_invalid = 0

    for idx, item in enumerate(sample_items, start=1):
        gt = _extract_ground_truth_decision(item)
        if gt in ("BUY", "SELL", "HOLD"):
            gt_known += 1
            if gt == "HOLD":
                gt_hold += 1

        prompt = _build_prompt(tokenizer, item)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        out_ids = outputs[0]
        gen_ids = out_ids[prompt_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

        ok_json, pred = _parse_pred_decision(decoded)
        if ok_json:
            valid_json += 1

        if pred != "ERROR":
            decision_found += 1

        if pred not in dist:
            dist["OTHER"] += 1
        else:
            dist[pred] += 1

        if pred == gt:
            correct += 1
            if ok_json:
                correct_json_only += 1

        if shown < 3:
            shown += 1
            print("-" * 40)
            print(f"Case {shown}")
            print(f"GT={gt} PRED={pred} json_ok={ok_json}")

        if (not ok_json) and shown_invalid < int(args.debug_invalid):
            shown_invalid += 1
            print("-" * 40)
            print(f"Invalid JSON sample {shown_invalid}")
            print(f"GT={gt} PRED={pred}")
            print(f"RAW_OUTPUT_HEAD={decoded[:240].replace(chr(10), ' ')}")

    acc = (correct / n) if n > 0 else 0.0
    acc_json_only = (correct_json_only / valid_json) if valid_json > 0 else 0.0
    json_rate = (valid_json / n) if n > 0 else 0.0
    decision_found_rate = (decision_found / n) if n > 0 else 0.0

    hold_baseline = (gt_hold / gt_known) if gt_known > 0 else 0.0

    print("=" * 40)
    print(f"Evaluation Report")
    print(f"Samples: {n}")
    print(f"JSON valid: {valid_json}/{n} ({json_rate:.1%})")
    print(f"Decision extracted: {decision_found}/{n} ({decision_found_rate:.1%})")
    print(f"Accuracy: {correct}/{n} ({acc:.1%})")
    print(f"Accuracy (JSON-only): {correct_json_only}/{valid_json} ({acc_json_only:.1%})")
    print(f"GT HOLD baseline (if always predict HOLD): {hold_baseline:.1%}")
    print(f"Prediction distribution: {dist}")


if __name__ == "__main__":
    main()
