import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "models/news_final_3b_v1_full/lora_weights"

US_EVENT_TYPES: List[str] = [
    "rate_hike",
    "rate_cut",
    "hawkish_signal",
    "dovish_signal",
    "inflation_data",
    "employment_data",
    "gdp_growth",
    "earnings_beat",
    "earnings_miss",
    "guidance_update",
    "merger_acquisition",
    "liquidity_event",
    "short_squeeze",
    "technical_breakout",
    "geopolitical_tension",
    "trade_policy",
    "noise",
    "concept_hype",
]
US_EVENT_TYPE_SET = set(US_EVENT_TYPES)


TEST_CASES = [
    {
        "type": "Bullish (Earnings)",
        "title": "NVIDIA reports blowout earnings and announces buyback",
        "content": "NVIDIA (NVDA) reported Q3 revenue of $18.12 billion, up 206% year-over-year, crushing analyst estimates of $16.1 billion. Data center revenue hit a record high. The company also announced a $25 billion buyback program.",
        "source": "mock",
        "url": "https://example.com/nvda-earnings",
        "published_at": "2024-11-21T00:00:00Z",
    },
    {
        "type": "Bearish (China Policy/Regulation)",
        "title": "China announces stricter rules on gaming time for minors",
        "content": "China's regulators announced new strict rules on gaming time for minors, causing Tencent and NetEase stocks to plummet 10% in pre-market trading. Analysts warn this could significantly impact future recurring revenue.",
        "source": "mock",
        "url": "https://example.com/china-gaming-rules",
        "published_at": "2024-11-22T00:00:00Z",
    },
    {
        "type": "Neutral/Noise",
        "title": "Top 10 best vacuum cleaners for 2024",
        "content": "The top 10 best vacuum cleaners for 2024 have been released. Dyson and Shark continue to lead the market with new cordless models. Here is a breakdown of the battery life and suction power.",
        "source": "mock",
        "url": "https://example.com/vacuum-cleaners",
        "published_at": "2024-11-23T00:00:00Z",
    },
]


@dataclass
class SmokeNewsItem:
    id: str
    market: str
    published_at: str
    source: str
    url: str
    title: str
    content: str


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
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unclosed JSON object in output")


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def validate_news_schema(obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    required = {
        "market",
        "event_type",
        "subject_assets",
        "sentiment_score",
        "confidence",
        "summary",
        "reasoning",
        "rag_evidence_ids",
    }
    missing = [k for k in sorted(required) if k not in obj]
    extra = [k for k in sorted(obj.keys()) if k not in required]

    if missing or extra:
        return missing, extra

    if str(obj.get("market") or "").strip().upper() != "US":
        missing.append("market!=US")

    et = str(obj.get("event_type") or "").strip()
    if et not in US_EVENT_TYPE_SET:
        missing.append("event_type_not_in_enum")

    sa = obj.get("subject_assets")
    if not isinstance(sa, list):
        missing.append("subject_assets_not_list")

    se = _coerce_float(obj.get("sentiment_score"))
    if se is None or not (-1.0 <= se <= 1.0):
        missing.append("sentiment_score_invalid")

    cf = _coerce_float(obj.get("confidence"))
    if cf is None or not (0.0 <= cf <= 1.0):
        missing.append("confidence_invalid")

    rag = obj.get("rag_evidence_ids")
    if not isinstance(rag, list):
        missing.append("rag_evidence_ids_not_list")

    return missing, extra


def build_messages_v1(item: SmokeNewsItem) -> List[Dict[str, str]]:
    system = (
        "You are a US financial market intelligence analyst. "
        "Output MUST be one single valid JSON object and nothing else."
        "Do not output markdown."
    )

    enum_text = ", ".join(US_EVENT_TYPES)

    user_payload = {
        "id": item.id,
        "market": "US",
        "published_at": item.published_at,
        "source": item.source,
        "url": item.url,
        "title": item.title,
        "content": item.content,
    }

    schema = (
        "Output exactly ONE JSON object with keys: market, event_type, subject_assets, sentiment_score, confidence, summary, reasoning, rag_evidence_ids. "
        "market MUST be 'US'. "
        "event_type MUST be one of the predefined enums. "
        "subject_assets MUST be an array of uppercase tickers (e.g., ['SPY','QQQ']); it MAY be an empty array if the news has no clear tradable target. "
        "sentiment_score MUST be a float between -1 and 1. "
        "confidence MUST be a float between 0 and 1. "
        "rag_evidence_ids MUST be an array of ids from the provided historical context (or empty array). "
        "Do not include any additional keys."
    )

    note = "If the news is non-financial, off-topic, or contains insufficient information, set event_type='noise' and use subject_assets=[] with low confidence."

    user = (
        f"US_EVENT_TYPE_ENUMS: {enum_text}\n\n"
        "HISTORICAL_CONTEXT_TOP3_JSONL:\n(none)\n\n"
        f"CURRENT_NEWS_JSON={json.dumps(user_payload, ensure_ascii=False)}\n\n"
        f"{schema}\n{note}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main() -> None:
    print(f"Loading Base Model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
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

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

    print(f"Loading LoRA Adapter: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("\nStarting News Tower Smoke Test...")

    for i, case in enumerate(TEST_CASES, start=1):
        item = SmokeNewsItem(
            id=f"smoke_{i}",
            market="US",
            published_at=case["published_at"],
            source=case["source"],
            url=case["url"],
            title=case["title"],
            content=case["content"],
        )

        print(f"\n[{i}] Case: {case['type']}")
        print(f"Title: {item.title}")
        print(f"Text: {item.content[:120]}...")

        messages = build_messages_v1(item)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        out_ids = outputs[0]
        gen_ids = out_ids[prompt_len:]
        decoded_gen = tokenizer.decode(gen_ids, skip_special_tokens=True)

        raw = ""
        parsed: Optional[Dict[str, Any]] = None
        err = ""
        missing: List[str] = []
        extra: List[str] = []

        try:
            raw = extract_first_json(decoded_gen)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Model output JSON is not an object")
            missing, extra = validate_news_schema(parsed)
        except Exception as e:
            err = str(e)

        print("-" * 22 + " MODEL OUTPUT (JSON) " + "-" * 22)
        print(raw.strip() if raw else "(empty)")

        if parsed is None or err or missing or extra:
            print(f"RESULT: FAIL err={err} missing={missing} extra={extra}")
            continue

        print(
            "RESULT: OK "
            f"event_type={parsed.get('event_type')} "
            f"sentiment_score={parsed.get('sentiment_score')} "
            f"confidence={parsed.get('confidence')} "
            f"subject_assets={parsed.get('subject_assets')}"
        )


if __name__ == "__main__":
    main()
