#!/usr/bin/env python

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.risk.gate import RiskGate


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


def try_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        s = extract_first_json(text.strip())
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "JSON is not an object"
    except Exception as e:
        return None, str(e)


def parse_case_features(prompt: str) -> Dict[str, Any]:
    vol = 0.0
    dd = 0.0

    m = re.search(r"Vol\s*=\s*([-+]?\d+(?:\.\d+)?)%", prompt, flags=re.IGNORECASE)
    if m:
        try:
            vol = float(m.group(1))
        except Exception:
            vol = 0.0

    m = re.search(r"DD\s*=\s*([-+]?\d+(?:\.\d+)?)%", prompt, flags=re.IGNORECASE)
    if m:
        try:
            dd = float(m.group(1))
        except Exception:
            dd = 0.0

    return {
        "volatility_ann_pct": vol,
        "drawdown_20d_pct": dd,
    }


def parse_case_news_signals(prompt: str) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    if "News Risk" not in prompt:
        return signals

    m = re.search(
        r"News\s*Risk\s*:\s*.*?\(\s*Event\s*:\s*([\w\-]+)\s*,\s*Impact\s*:\s*([\w\-]+)\s*\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not m:
        return signals

    event_type = str(m.group(1)).strip()
    impact_txt = str(m.group(2)).strip().lower()
    impact_equity = -1.0 if impact_txt in {"negative", "neg", "bearish"} else 1.0
    signals.append({"event_type": event_type, "impact_equity": impact_equity})
    return signals


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


def build_messages(prompt: str) -> List[Dict[str, str]]:
    system = (
        "You are a professional quantitative trader and risk manager. "
        "Analyze the market data and provide a trading decision. "
        "You must output STRICT JSON only (no markdown, no extra text). "
        "The JSON must contain keys: action, target_position, risk_notes, rationale. "
        "action must be one of: buy, add, hold, reduce, clear. "
        "target_position must be a number between 0 and 1. "
        "risk_notes must be an array of short strings."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = build_messages(prompt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    do_sample = float(temperature) > 0
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=do_sample,
            repetition_penalty=1.05,
        )

    out_ids = generated[0][model_inputs.input_ids.shape[1] :]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--lora", type=str, default="models/llm_etf_trading_qwen25_14b")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    base_model = str(args.base).replace("\\", "/")
    adapter_path = resolve_adapter_path(str(args.lora))

    print(f"Loading Base Model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_kwargs["torch_dtype"] = torch.float16
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    print(f"Loading Trading LoRA Adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    test_cases = [
        {
            "desc": "Case 1: 强牛市 (均线多头排列 + 动量强)",
            "input": """
Asset: SPY
Price: 480.50
Trend: Bullish (Price > MA20 > MA60)
Momentum: Strong (5D=+3.2%, 20D=+5.5%)
Risk: Low (Vol=11%, DD=-0.5%)
Regime: Risk_On
News Risk: None
Decision:
""".strip(),
        },
        {
            "desc": "Case 2: 暴跌破位 (监管打击 + 跌破均线)",
            "input": """
Asset: 510300 (CSI 300)
Price: 3.20
Trend: Bearish (Price < MA20 < MA60)
Momentum: Weak (5D=-4.1%, 20D=-8.2%)
Risk: High (Vol=28%, DD=-12.5%)
Regime: Risk_Off
News Risk: Critical (Event: regulation_crackdown, Impact: Negative)
Decision:
""".strip(),
        },
        {
            "desc": "Case 3: 高位震荡 (动量减弱)",
            "input": """
Asset: QQQ
Price: 405.00
Trend: Sideways (Price ~ MA20 > MA60)
Momentum: Neutral (5D=-0.5%, 20D=+1.2%)
Risk: Moderate (Vol=18%, DD=-2.0%)
Regime: Transition
News Risk: Moderate (Event: concept_hype, Impact: Positive)
Decision:
""".strip(),
        },
    ]

    print("\n=== Starting Inference Test ===")
    gate = RiskGate()
    for case in test_cases:
        print(f"\n>> {case['desc']}")
        print("Input:\n" + case["input"])
        print("-" * 30)
        out = generate_response(
            model,
            tokenizer,
            case["input"],
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
        print("Model Output:\n" + out)
        obj, err = try_parse_json(out)
        if obj is not None:
            print("JSON_OK: True")
            print(json.dumps(obj, ensure_ascii=False, indent=2))

            proposed_action = str(obj.get("action") or "hold")
            try:
                proposed_pos = float(obj.get("target_position", 0.0))
            except Exception:
                proposed_pos = 0.0

            features = parse_case_features(case["input"])
            news_signals = parse_case_news_signals(case["input"])
            final_action, final_pos, trace = gate.adjudicate(
                features,
                news_signals,
                proposed_action,
                proposed_pos,
            )

            print("FINAL_DECISION:")
            print(json.dumps({"action": final_action.lower(), "target_position": final_pos}, ensure_ascii=False))
            print("RISK_TRACE:")
            for t in trace:
                print("- " + str(t))
        else:
            print("JSON_OK: False")
            print("JSON_ERROR: " + str(err))
        print("=" * 50)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
