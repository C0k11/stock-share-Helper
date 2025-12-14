#!/usr/bin/env python

import sys
import argparse
import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


CN_EVENT_TYPES = [
    "policy_stimulus",
    "regulation_crackdown",
    "market_intervention",
    "concept_hype",
    "corporate_restructuring",
]


def sanitize_input_text(s: str) -> str:
    return (s or "").replace('"', "'")


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


def repair_json_text(s: str) -> str:
    if not s:
        return s
    s2 = s
    s2 = re.sub(r",\s*=(?=\s*\")", ",", s2)
    s2 = re.sub(r"\{\s*=(?=\s*\")", "{", s2)
    s2 = re.sub(r"=\s*,", ",", s2)
    s2 = re.sub(r"(:\s*-?\d+)\s*\"(?=\s*[},])", r"\1", s2)
    s2 = re.sub(r"\"\s*,\s*\"", r"\", \"", s2)
    return s2


def stable_id(title: str, content: str, published_at: str) -> str:
    basis = "|".join([published_at or "", title or "", content or ""]).strip()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def enforce_cn_rules(obj: Dict[str, Any]) -> Dict[str, Any]:
    et = str(obj.get("event_type") or "").strip()

    def as_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    impact_equity = as_int(obj.get("impact_equity"))
    impact_bond = as_int(obj.get("impact_bond"))
    impact_gold = as_int(obj.get("impact_gold"))

    if et == "policy_stimulus":
        impact_equity = 1
        impact_bond = 1
    elif et == "regulation_crackdown":
        impact_equity = -1
    elif et == "market_intervention":
        impact_equity = 1
    elif et == "concept_hype":
        impact_equity = 1
    elif et == "corporate_restructuring":
        impact_equity = 1

    impact_equity = max(-1, min(1, impact_equity))
    impact_bond = max(-1, min(1, impact_bond))
    impact_gold = max(-1, min(1, impact_gold))

    obj["impact_equity"] = impact_equity
    obj["impact_bond"] = impact_bond
    obj["impact_gold"] = impact_gold

    if "sentiment" in obj and isinstance(obj["sentiment"], str):
        obj["sentiment"] = obj["sentiment"].strip().lower()

    if "summary" in obj and isinstance(obj["summary"], str):
        obj["summary"] = sanitize_input_text(obj["summary"].strip())

    return obj


def validate_schema(obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    required = ["event_type", "sentiment", "impact_equity", "impact_bond", "impact_gold", "summary"]
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj.keys() if k not in required]
    return missing, extra


def build_teacher_messages_cn(title: str, content: str) -> List[Dict[str, str]]:
    title = sanitize_input_text(title)
    content = sanitize_input_text(content)

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
        "- If event_type is concept_hype, impact_equity MUST be 1 and summary MUST mention speculative/short-term nature."
    )

    user = (
        f"Title: {title}\n"
        f"Content: {content}\n\n"
        "Output exactly ONE JSON object with these fields: "
        "event_type, sentiment, impact_equity(-1/0/1), impact_bond(-1/0/1), impact_gold(-1/0/1), summary. "
        "Do not include any additional keys."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_openai_compat_base_url(base_url: str) -> str:
    """Normalize base_url to an OpenAI-compatible root.

    - Accepts both https://api.xxx.com and https://api.xxx.com/v1
    - Returns https://api.xxx.com/v1 (no trailing slash)
    """
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("Empty base_url")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
) -> str:
    base_url = normalize_openai_compat_base_url(base_url)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema: {data}") from e


def load_news_items(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input JSON list of news items")
    parser.add_argument("--out", required=True, help="Output dataset path (JSON list of conversations)")

    parser.add_argument("--market", choices=["CN"], default="CN")
    parser.add_argument("--teacher-provider", choices=["openai_compat"], default="openai_compat")
    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")

    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max", type=int, default=0, help="Limit number of items (0=all)")

    args = parser.parse_args()

    api_key = os.getenv(args.teacher_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Missing API key env var: {args.teacher_api_key_env}. "
            "Set it in your environment before running."
        )

    inp_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_news_items(inp_path)
    if args.max and args.max > 0:
        items = items[: int(args.max)]

    out_data: List[Dict[str, Any]] = []

    for i, item in enumerate(items):
        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()
        published_at = str(item.get("published_at") or item.get("meta", {}).get("published_at") or "").strip()

        cid = str(item.get("id") or stable_id(title, content, published_at))

        if not title and not content:
            continue

        messages = build_teacher_messages_cn(title=title, content=content)

        raw = ""
        parsed = None
        missing: List[str] = []
        extra: List[str] = []
        err = ""

        try:
            raw = call_openai_compatible_chat(
                base_url=args.teacher_base_url,
                api_key=api_key,
                model=args.teacher_model,
                messages=messages,
                timeout=int(args.timeout),
            )

            json_text = extract_first_json(raw.strip())
            try:
                parsed = json.loads(json_text)
            except Exception:
                parsed = json.loads(repair_json_text(json_text))

            if not isinstance(parsed, dict):
                raise ValueError("Teacher output JSON is not an object")

            if args.market == "CN":
                parsed = enforce_cn_rules(parsed)

            missing, extra = validate_schema(parsed)
        except Exception as e:
            err = str(e)

        out_item = {
            "conversations": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional financial news analyst. "
                        "You must output STRICT JSON only (no markdown, no prose outside JSON). "
                        "Do not use double quotes (\") within string values; use single quotes (') instead."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Title: {sanitize_input_text(title)}\n"
                        f"Content: {sanitize_input_text(content)}\n\n"
                        "Output exactly ONE JSON object with fields: event_type, sentiment, impact_equity(-1/0/1), impact_bond(-1/0/1), impact_gold(-1/0/1), summary. "
                        "Do not include any additional keys."
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "{}",
                },
            ],
            "meta": {
                "id": cid,
                "market": args.market,
                "published_at": published_at,
                "teacher_model": args.teacher_model,
                "teacher_base_url": args.teacher_base_url,
                "teacher_error": err,
                "teacher_missing": missing,
                "teacher_extra": extra,
            },
        }

        out_data.append(out_item)

        status = "OK" if parsed is not None and not missing and not extra else "FAIL"
        logger.info(f"[{i+1}/{len(items)}] {cid}: {status} missing={missing} extra={extra} err={err[:120]}")

        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved dataset: {out_path} items={len(out_data)}")


if __name__ == "__main__":
    main()
