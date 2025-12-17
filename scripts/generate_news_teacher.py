#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(project_root / ".env.local", override=False)
        load_dotenv(project_root / ".env", override=False)
    except Exception:
        return


def normalize_openai_compat_base_url(base_url: str) -> str:
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
    max_tokens: int,
    temperature: float,
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
        "temperature": float(temperature),
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema: {data}") from e


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


def stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


_WORD_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


@dataclass
class NewsItem:
    id: str
    title: str
    content: str
    published_at: str
    source: str
    url: str


class SimpleNewsRAG:
    def __init__(self, items: Sequence[NewsItem]):
        self.items = list(items)
        self._token_sets: List[set] = [set(tokenize(it.title + "\n" + it.content)) for it in self.items]

    def search(self, query: NewsItem, top_k: int = 3, exclude_id: Optional[str] = None) -> List[NewsItem]:
        qset = set(tokenize(query.title + "\n" + query.content))
        if not qset:
            return []

        scored: List[Tuple[float, int]] = []
        for i, it in enumerate(self.items):
            if exclude_id and it.id == exclude_id:
                continue
            aset = self._token_sets[i]
            if not aset:
                continue
            inter = len(qset & aset)
            if inter <= 0:
                continue
            union = len(qset | aset)
            score = inter / max(1, union)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[NewsItem] = []
        for score, idx in scored[: max(0, int(top_k))]:
            _ = score
            out.append(self.items[idx])
        return out


def load_news_items(path: Path, max_items: int = 0) -> List[NewsItem]:
    items: List[Dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
                if max_items and len(items) >= int(max_items):
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            items = [x for x in obj if isinstance(x, dict)]
        else:
            raise ValueError("Input must be JSONL or JSON list")

    out: List[NewsItem] = []
    for i, it in enumerate(items):
        title = str(it.get("title") or "").strip()
        content = str(it.get("content") or "").strip()
        published_at = str(it.get("published_at") or it.get("time") or it.get("datetime") or "").strip()
        source = str(it.get("source") or it.get("source_id") or "").strip()
        url = str(it.get("url") or "").strip()

        rid = str(it.get("id") or stable_id(title, content, published_at, url))
        if not title and not content:
            continue

        out.append(
            NewsItem(
                id=rid,
                title=title,
                content=content,
                published_at=published_at,
                source=source,
                url=url,
            )
        )

        if max_items and len(out) >= int(max_items):
            break

        _ = i

    return out


def validate_schema_v1(obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
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
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj.keys() if k not in required]

    if str(obj.get("market") or "").strip() != "US":
        missing.append("market(US)")

    et = str(obj.get("event_type") or "").strip()
    if not et:
        missing.append("event_type")
    elif et not in US_EVENT_TYPE_SET:
        missing.append("event_type(enum)")

    sa = obj.get("subject_assets")
    if not isinstance(sa, list):
        missing.append("subject_assets(array)")
    else:
        for x in sa[:10]:
            if not isinstance(x, str) or not x.strip():
                missing.append("subject_assets(items as string)")
                break

    def _to_float(v: Any) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    ss = _to_float(obj.get("sentiment_score"))
    if ss is None or not (-1.0 <= ss <= 1.0):
        missing.append("sentiment_score(range -1..1)")

    cf = _to_float(obj.get("confidence"))
    if cf is None or not (0.0 <= cf <= 1.0):
        missing.append("confidence(range 0..1)")

    if not isinstance(obj.get("summary"), str) or not str(obj.get("summary") or "").strip():
        missing.append("summary")

    if not isinstance(obj.get("reasoning"), str) or not str(obj.get("reasoning") or "").strip():
        missing.append("reasoning")

    rei = obj.get("rag_evidence_ids")
    if not isinstance(rei, list):
        missing.append("rag_evidence_ids(array)")
    else:
        for x in rei[:10]:
            if not isinstance(x, str):
                missing.append("rag_evidence_ids(items as string)")
                break

    return missing, extra


def build_teacher_messages(*, item: NewsItem, context_items: Sequence[NewsItem]) -> List[Dict[str, str]]:
    system = (
        "You are a US financial market intelligence analyst. "
        "Output MUST be one single valid JSON object and nothing else." 
        "Do not output markdown."
    )

    enum_text = ", ".join(US_EVENT_TYPES)

    ctx_lines: List[str] = []
    for ci in list(context_items)[:3]:
        ctx_lines.append(
            json.dumps(
                {
                    "id": ci.id,
                    "published_at": ci.published_at,
                    "source": ci.source,
                    "url": ci.url,
                    "title": ci.title[:240],
                    "content": ci.content[:800],
                },
                ensure_ascii=False,
            )
        )
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "(none)"

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

    note = (
        "If the news is non-financial, off-topic, or contains insufficient information, set event_type='noise' and use subject_assets=[] with low confidence."
    )

    user = (
        f"US_EVENT_TYPE_ENUMS: {enum_text}\n\n"
        f"HISTORICAL_CONTEXT_TOP3_JSONL:\n{ctx_block}\n\n"
        f"CURRENT_NEWS_JSON={json.dumps(user_payload, ensure_ascii=False)}\n\n"
        f"{schema}\n{note}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def iter_existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("id"):
                out.add(str(obj.get("id")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate News C teacher dataset (US-only + RAG context) via OpenAI-compatible API")
    parser.add_argument(
        "--in",
        "--in_file",
        "--in-file",
        dest="inp",
        required=True,
        help="Input raw news file (.jsonl or .json list)",
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        default="data/finetune/news_final_3b",
        help="Output directory for teacher/train/val artifacts",
    )
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--out-train", default=None)
    parser.add_argument("--out-val", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)

    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL", "deepseek-chat"))
    parser.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max", type=int, default=0)

    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--no-rag", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume by skipping ids already in out-jsonl")

    args = parser.parse_args()

    try_load_dotenv()

    api_key = os.getenv(args.teacher_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Missing API key env var: {args.teacher_api_key_env}. Set it in your environment before running."
        )

    inp_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else out_dir / "teacher_news_us_v1.jsonl"
    out_train = Path(args.out_train) if args.out_train else out_dir / "train_news_v1.json"
    out_val = Path(args.out_val) if args.out_val else out_dir / "val_news_v1.json"

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    items = load_news_items(inp_path, max_items=int(args.max) if int(args.max) > 0 else 0)
    if not items:
        raise SystemExit(f"No items loaded from: {inp_path}")

    existing_ids = iter_existing_ids(out_jsonl) if bool(args.resume) else set()
    if existing_ids:
        logger.info(f"Resume enabled: found {len(existing_ids)} existing ids in {out_jsonl}")

    rag = None
    if not bool(args.no_rag):
        rag = SimpleNewsRAG(items)

    written_ok = 0
    written_fail = 0

    with open(out_jsonl, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(items, start=1):
            if item.id in existing_ids:
                continue

            ctx = []
            if rag is not None:
                try:
                    ctx = rag.search(item, top_k=int(args.rag_top_k), exclude_id=item.id)
                except Exception:
                    ctx = []

            messages = build_teacher_messages(item=item, context_items=ctx)

            raw = ""
            parsed: Optional[Dict[str, Any]] = None
            err = ""
            missing: List[str] = []
            extra: List[str] = []

            try:
                raw = call_openai_compatible_chat(
                    base_url=str(args.teacher_base_url),
                    api_key=api_key,
                    model=str(args.teacher_model),
                    messages=messages,
                    timeout=int(args.timeout),
                    max_tokens=int(args.max_output_tokens),
                    temperature=float(args.temperature),
                )

                json_text = extract_first_json(raw.strip())
                try:
                    parsed = json.loads(json_text)
                except Exception:
                    parsed = json.loads(repair_json_text(json_text))

                if not isinstance(parsed, dict):
                    raise ValueError("Teacher output JSON is not an object")

                if "market" in parsed:
                    parsed["market"] = "US"
                else:
                    parsed["market"] = "US"

                sa = parsed.get("subject_assets")
                if isinstance(sa, list):
                    parsed["subject_assets"] = [str(x).strip().upper() for x in sa if str(x).strip()]

                missing, extra = validate_schema_v1(parsed)
                if missing or extra:
                    raise ValueError(f"Schema mismatch missing={missing} extra={extra}")
            except Exception as e:
                err = str(e)
                parsed = None

            out_obj: Dict[str, Any] = {
                "id": item.id,
                "meta": {
                    "market": "US",
                    "published_at": item.published_at,
                    "source": item.source,
                    "url": item.url,
                    "teacher_base_url": str(args.teacher_base_url),
                    "teacher_model": str(args.teacher_model),
                    "teacher_error": err,
                    "teacher_missing": missing,
                    "teacher_extra": extra,
                },
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are a US financial market intelligence analyst. Output STRICT JSON only.",
                    },
                    {
                        "role": "user",
                        "content": messages[1]["content"],
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "{}",
                    },
                ],
                "raw": raw,
            }

            status = "OK" if parsed is not None else "FAIL"
            if status == "OK":
                written_ok += 1
                existing_ids.add(item.id)
            else:
                written_fail += 1

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()

            logger.info(
                f"[{idx}/{len(items)}] {item.id} {status} ok={written_ok} fail={written_fail} missing={missing} extra={extra} err={err[:120]}"
            )

            if args.sleep and float(args.sleep) > 0:
                time.sleep(float(args.sleep))

    ok_items: List[Dict[str, Any]] = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            conv = obj.get("conversations")
            if not isinstance(conv, list) or len(conv) < 3:
                continue
            assistant = conv[-1]
            if not isinstance(assistant, dict):
                continue
            if str(assistant.get("content") or "").strip() == "{}":
                continue
            ok_items.append({"conversations": conv, "meta": obj.get("meta")})

    import random

    rng = random.Random(int(args.seed))
    rng.shuffle(ok_items)

    val_ratio = float(args.val_ratio)
    val_n = int(max(1, round(len(ok_items) * val_ratio))) if ok_items else 0
    val = ok_items[:val_n]
    train = ok_items[val_n:]

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(out_val, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Saved teacher_jsonl={out_jsonl} ok={len(ok_items)} train={len(train)} val={len(val)} fail={written_fail}"
    )


if __name__ == "__main__":
    main()
