#!/usr/bin/env python

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def sanitize_input_text(s: str) -> str:
    return (s or "").replace('"', "'")


def extract_first_json(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
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
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def try_parse_json(text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    js = extract_first_json(text)
    if js is None:
        return False, None, None
    try:
        return True, json.loads(js), js
    except Exception:
        return False, None, js


def build_messages_news(*, market: str, title: str, content: str) -> List[Dict[str, str]]:
    market = (market or "US").strip().upper()
    title = sanitize_input_text(title)
    content = sanitize_input_text(content)

    if market == "CN":
        system = (
            "You are an expert A-Share (Chinese Stock Market) trader. You understand hidden signals and A-share jargon. "
            "You must output STRICT JSON only (no markdown, no prose outside JSON). "
            "Output valid JSON only using standard JSON double quotes for keys and string values. "
            "Do NOT use single quotes to delimit JSON strings. Do NOT include quote characters inside string values. "
            "You MUST choose event_type from this enum: "
            f"{CN_EVENT_TYPES}. "
            "Tagging Rules (decision logic): "
            "- If the news mentions concrete monetary/fiscal tools (e.g., RRR cut, interest rate cut, special bonds, white list, debt swap), choose policy_stimulus. "
            "- If the news is about National Team buying, stabilizing market expectations, long-term funds entering market, or curbing short-selling, choose market_intervention. "
            "- If both appear: prioritize policy_stimulus if real money/liquidity is released; prioritize market_intervention if it's mostly verbal support or direct stock buying. "
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
            "Output valid JSON only using standard JSON double quotes for keys and string values. "
            "Do NOT use single quotes to delimit JSON strings. Do NOT include quote characters inside string values. "
            "This is a US market task. "
            "You MUST choose event_type from this US-only enum: "
            f"{US_EVENT_TYPES}. "
            "Do NOT output any China/A-share event types (forbidden examples: policy_stimulus, regulation_crackdown, market_intervention, concept_hype, corporate_restructuring). "
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


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    return data


def _load_existing_signals(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return _load_json_list(path)
    except Exception:
        return []


def _today_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="Run daily inference over fetched news and save structured signals")
    parser.add_argument("--in", dest="in_path", default=None, help="Input news JSON (default: data/daily/news_YYYY-MM-DD.json)")
    parser.add_argument("--out", dest="out_path", default=None, help="Output signals JSON (default: data/daily/signals_YYYY-MM-DD.json)")
    parser.add_argument("--date", default=None, help="Override date YYYY-MM-DD used for default in/out")

    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--lora", default="models/llm_qwen14b_lora_c_hybrid/lora_weights")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--market", choices=["ALL", "US", "CN"], default="ALL")
    parser.add_argument("--sample-us", type=int, default=0, help="Take first N US items (overrides --offset/--limit selection)")
    parser.add_argument("--sample-cn", type=int, default=0, help="Take first N CN items (overrides --offset/--limit selection)")

    args = parser.parse_args()

    date_str = args.date or _today_str()
    in_path = Path(args.in_path) if args.in_path else Path("data/daily") / f"news_{date_str}.json"
    out_path = Path(args.out_path) if args.out_path else Path("data/daily") / f"signals_{date_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = _load_json_list(in_path)

    # Market filtering / sampling (for smoke tests and split-brain validation)
    market_filter = (args.market or "ALL").upper()
    if market_filter in ("US", "CN"):
        items = [it for it in items if str(it.get("market") or "US").strip().upper() == market_filter]

    if (args.sample_us and args.sample_us > 0) or (args.sample_cn and args.sample_cn > 0):
        us_items = [it for it in items if str(it.get("market") or "US").strip().upper() == "US"]
        cn_items = [it for it in items if str(it.get("market") or "US").strip().upper() == "CN"]
        picked: List[Dict[str, Any]] = []
        if args.sample_us and args.sample_us > 0:
            picked.extend(us_items[: args.sample_us])
        if args.sample_cn and args.sample_cn > 0:
            picked.extend(cn_items[: args.sample_cn])
        items = picked
    else:
        if args.offset and args.offset > 0:
            items = items[args.offset :]
        if args.limit and args.limit > 0:
            items = items[: args.limit]

    logger.info(f"Input items: {len(items)} from {in_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {args.model}")
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

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.use_lora:
        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        logger.info(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    signals: List[Dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume:
        existing = _load_existing_signals(out_path)
        for it in existing:
            sid = it.get("id")
            if isinstance(sid, str) and sid:
                done_ids.add(sid)
        if existing:
            signals = existing
        logger.info(f"Resume enabled: loaded={len(existing)} done_ids={len(done_ids)}")

    et_counts: Dict[str, int] = {}
    market_counts: Dict[str, int] = {"US": 0, "CN": 0}

    pending: List[Dict[str, Any]] = []
    for it in items:
        sid = it.get("id")
        if args.resume and isinstance(sid, str) and sid and sid in done_ids:
            continue
        pending.append(it)

    logger.info(f"Pending items: {len(pending)}")

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.deterministic,
    }
    if not args.deterministic:
        gen_kwargs.update({"temperature": 0.7, "top_p": 0.9})

    batch_size = max(1, int(args.batch_size))
    save_every = max(1, int(args.save_every))
    processed_in_run = 0

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]

        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for it in batch:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip()
            market = (it.get("market") or "US").strip().upper()
            if market not in ("US", "CN"):
                market = "US"

            messages = build_messages_news(market=market, title=title, content=content)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            metas.append(
                {
                    "id": it.get("id"),
                    "market": market,
                    "source": (it.get("source") or "").strip(),
                    "url": (it.get("url") or "").strip(),
                    "published_at": it.get("published_at"),
                    "title": title,
                    "input_chars": len(content),
                }
            )

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for i in range(len(metas)):
            out_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            parse_ok, obj, raw_json = try_parse_json(out_text)
            meta = metas[i]

            market = meta["market"]
            market_counts[market] = market_counts.get(market, 0) + 1
            if parse_ok and obj:
                et = str(obj.get("event_type") or "").strip()
                et_counts[et] = et_counts.get(et, 0) + 1

            signals.append(
                {
                    **meta,
                    "parse_ok": parse_ok,
                    "signal": obj,
                    "raw_json": raw_json,
                }
            )

            sid = meta.get("id")
            if isinstance(sid, str) and sid:
                done_ids.add(sid)

            processed_in_run += 1

        total_done = len(done_ids) if args.resume else processed_in_run
        denom = (len(items) if not args.resume else len(items))
        ok_count = sum(1 for x in signals if x.get("parse_ok"))
        if (processed_in_run % 10 == 0) or (start + batch_size >= len(pending)):
            logger.info(f"[{min(total_done, denom)}/{denom}] parse_ok={ok_count}/{len(signals)}")

        if (processed_in_run % save_every == 0) or (start + batch_size >= len(pending)):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(signals, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved signals: {out_path}")
    final_ok = sum(1 for x in signals if x.get("parse_ok"))
    logger.info(f"Parse OK rate: {final_ok}/{len(signals)}")
    logger.info(f"Market counts: {market_counts}")
    logger.info(f"Top event_type counts: {sorted(et_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")


if __name__ == "__main__":
    main()
