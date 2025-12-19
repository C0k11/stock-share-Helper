#!/usr/bin/env python

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_tools import extract_json_text, repair_and_parse_json


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# NOTE: kept as the pre-retry2 path so your PowerShell replace command can patch it.
LORA_PATH = "models/news_final_3b_v1_1_noise_killer_retry2/lora_weights"

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
    "noise",
]


def sanitize_input_text(s: str) -> str:
    return (s or "").replace('"', "'")


def try_parse_json(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    raw_json = extract_json_text(text)
    if raw_json is None:
        return False, None, None

    obj = repair_and_parse_json(text)
    if isinstance(obj, dict):
        return True, obj, raw_json
    return False, None, raw_json


def build_messages_news(*, market: str, title: str, content: str) -> List[Dict[str, str]]:
    market = (market or "US").strip().upper()
    title = sanitize_input_text(title)
    content = sanitize_input_text(content)

    system = (
        "You are a professional US financial news analyst. "
        "You must output STRICT JSON only (no markdown, no prose outside JSON). "
        "Output valid JSON only using standard JSON double quotes for keys and string values. "
        "Do NOT use single quotes to delimit JSON strings. Do NOT include quote characters inside string values. "
        "This is a US market task. "
        "You MUST choose event_type from this US-only enum: "
        f"{US_EVENT_TYPES}. "
        "Do NOT output any China/A-share event types (forbidden examples: policy_stimulus, regulation_crackdown, market_intervention, concept_hype, corporate_restructuring). "
        "If the content is non-financial, off-topic, or has no tradable market impact, choose event_type noise and set impacts to 0. "
        "If you are unsure, use other_us."
    )

    user = (
        f"Market: {market}\n"
        f"Title: {title}\n"
        f"Content: {content}\n\n"
        "Output exactly ONE JSON object with these fields: "
        "event_type, sentiment, impact_equity(-1/0/1), impact_bond(-1/0/1), impact_gold(-1/0/1), summary. "
        "Do not include any additional keys."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class TestCase:
    name: str
    title: str
    content: str
    expect_noise: bool


TEST_CASES: List[TestCase] = [
    TestCase(
        name="Case 1 (Dyson)",
        title="Top 10 best vacuum cleaners for 2024",
        content=(
            "The top 10 best vacuum cleaners for 2024 have been released. Dyson and Shark continue to lead the market with new cordless models. "
            "Here is a breakdown of the battery life and suction power."
        ),
        expect_noise=True,
    ),
    TestCase(
        name="Case 2 (LeBron)",
        title="LeBron James signs new endorsement deal",
        content=(
            "LeBron James has reportedly signed a new endorsement deal with a major sportswear brand. Fans reacted on social media and analysts discussed the marketing impact."
        ),
        expect_noise=True,
    ),
    TestCase(
        name="Case 3 (NVDA)",
        title="NVIDIA reports blowout earnings and raises guidance",
        content=(
            "NVIDIA (NVDA) reported quarterly results that beat Wall Street estimates and raised forward guidance, citing strong demand for AI data center GPUs. "
            "The stock rose in after-hours trading as analysts upgraded price targets."
        ),
        expect_noise=False,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="News Tower v1.1 Noise-Killer targeted smoke test")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--lora", default=LORA_PATH)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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

    base_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    model = base_model
    if args.use_lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.lora))

    model.eval()

    results = []

    for tc in TEST_CASES:
        messages = build_messages_news(market="US", title=tc.title, content=tc.content)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        ok, obj, raw = try_parse_json(decoded)
        event_type = None
        if ok and isinstance(obj, dict):
            event_type = str(obj.get("event_type") or "").strip()

        pass_case = (event_type == "noise") if tc.expect_noise else (event_type not in (None, "", "noise"))

        results.append(
            {
                "case": tc.name,
                "expect_noise": tc.expect_noise,
                "event_type": event_type,
                "pass": bool(pass_case),
                "raw_json": raw,
            }
        )

        print("=" * 90)
        print(tc.name)
        print(f"Title: {tc.title}")
        print(f"Expect noise: {tc.expect_noise}")
        print("-- MODEL JSON --")
        print(raw if raw else "(no json extracted)")
        print(f"RESULT: {'PASS' if pass_case else 'FAIL'}  event_type={event_type}")

    print("=" * 90)
    print(json.dumps({"summary": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
