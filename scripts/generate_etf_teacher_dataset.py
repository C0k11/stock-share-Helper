#!/usr/bin/env python

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


_DATE_RE = re.compile(r"^etf_features_(\d{4}-\d{2}-\d{2})\.json$", re.IGNORECASE)


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


def stable_sample_id(date_str: str, symbol: str, payload: Dict[str, Any]) -> str:
    basis = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    raw = f"{date_str}|{symbol}|{basis}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def iter_etf_feature_files(daily_dir: Path, start_date: Optional[str], end_date: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for p in sorted(daily_dir.glob("etf_features_*.json")):
        m = _DATE_RE.match(p.name)
        if not m:
            continue
        d = m.group(1)
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        out.append(p)
    return out


def load_existing_ids_jsonl(out_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not out_path.exists():
        return ids
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = str(obj.get("id") or "")
                    if sid:
                        ids.add(sid)
                except Exception:
                    continue
    except Exception:
        return ids
    return ids


def load_json_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    return data


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    return data


def build_risk_watch_summary(daily_dir: Path, date_str: str) -> Dict[str, Any]:
    sig_path = daily_dir / f"signals_{date_str}.json"
    if not sig_path.exists():
        return {"signals_path": str(sig_path), "available": False}

    try:
        items = load_json_list(sig_path)
    except Exception:
        return {"signals_path": str(sig_path), "available": False}

    cn_ok = [it for it in items if str(it.get("market") or "").upper() == "CN" and it.get("parse_ok")]

    rc: List[Dict[str, Any]] = []
    for it in cn_ok:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        if str(sig.get("event_type") or "").strip() == "regulation_crackdown":
            rc.append(it)

    def compact(it: Dict[str, Any]) -> Dict[str, Any]:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        return {
            "title": str(it.get("title") or "")[:120],
            "source": str(it.get("source") or ""),
            "impact_equity": sig.get("impact_equity"),
            "summary": str(sig.get("summary") or "")[:200],
        }

    top = [compact(x) for x in rc[:5]]
    total_ok = len(cn_ok)
    n = len(rc)
    return {
        "signals_path": str(sig_path),
        "available": True,
        "cn_parse_ok": total_ok,
        "regulation_crackdown": {
            "count": n,
            "share": (n / total_ok) if total_ok > 0 else 0.0,
            "top": top,
        },
    }


def build_teacher_messages(
    *,
    symbol: str,
    date_str: str,
    etf_item: Dict[str, Any],
    risk_watch: Dict[str, Any],
    include_cot: bool,
) -> List[Dict[str, str]]:
    system = (
        "You are a senior portfolio manager and risk officer. "
        "You will be given ETF/index market features (technical, volatility, drawdown, regime) and optional risk-watch news summary. "
        "Your goal is to recommend an action and target position under risk constraints. "
        "Output MUST be a single valid JSON object and nothing else."
    )

    role_spec = (
        "Use 3 roles and then a synthesis:\n"
        "- role_aggressive: macro hedge fund manager (opportunistic)\n"
        "- role_risk: bank risk officer (capital preservation)\n"
        "- role_quant: systematic quant (rules + data)\n"
        "Finally write synthesis and output label."
    )

    cot = (
        "Provide detailed reasoning in each role (300-600 Chinese characters) before concluding."
        if include_cot
        else "Provide concise reasoning in each role (<=120 Chinese characters)."
    )

    label_schema = (
        "The JSON object MUST have keys: role_aggressive, role_risk, role_quant, synthesis, label. "
        "label MUST be an object with keys: action, target_position, risk_notes, rationale. "
        "action MUST be one of: buy, add, hold, reduce, clear. "
        "target_position MUST be a number between 0 and 1. "
        "risk_notes MUST be an array of short strings. "
        "Do not include any other keys."
    )

    user_payload = {
        "date": date_str,
        "symbol": symbol,
        "etf_features": etf_item,
        "risk_watch": risk_watch,
    }

    user = (
        f"{role_spec}\n{cot}\n{label_schema}\n\n"
        f"INPUT_JSON={json.dumps(user_payload, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_label(obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    required = {"role_aggressive", "role_risk", "role_quant", "synthesis", "label"}
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj.keys() if k not in required]

    label = obj.get("label") if isinstance(obj.get("label"), dict) else None
    if label is None:
        missing.append("label(action/target_position/risk_notes/rationale)")
        return missing, extra

    req2 = {"action", "target_position", "risk_notes", "rationale"}
    missing += [f"label.{k}" for k in req2 if k not in label]

    action = str(label.get("action") or "").strip().lower()
    if action and action not in {"buy", "add", "hold", "reduce", "clear"}:
        missing.append("label.action(in buy/add/hold/reduce/clear)")

    try:
        tp = float(label.get("target_position"))
        if not (0.0 <= tp <= 1.0):
            missing.append("label.target_position(range 0..1)")
    except Exception:
        missing.append("label.target_position(number)")

    if "risk_notes" in label and not isinstance(label.get("risk_notes"), list):
        missing.append("label.risk_notes(array)")

    return missing, extra


def main():
    parser = argparse.ArgumentParser(description="Generate ETF teacher dataset via DeepSeek (OpenAI-compatible) with multi-role reasoning")
    parser.add_argument("--daily-dir", default="data/daily")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)

    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL", "deepseek-chat"))
    parser.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")

    parser.add_argument("--out", default="data/finetune/teacher_etf/teacher_etf.jsonl")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max", type=int, default=0, help="Limit number of symbols across all days (0=all)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--include-cot", action="store_true")
    args = parser.parse_args()

    try_load_dotenv()

    api_key = os.getenv(args.teacher_api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.teacher_api_key_env}")

    daily_dir = Path(args.daily_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: Set[str] = set()
    if args.resume:
        existing_ids = load_existing_ids_jsonl(out_path)
        logger.info(f"Resume enabled: existing_ids={len(existing_ids)}")

    files = iter_etf_feature_files(daily_dir, args.start_date, args.end_date)
    if not files:
        logger.warning(f"No etf_features files found in {daily_dir} for range {args.start_date}..{args.end_date}")
        return

    written = 0
    skipped = 0
    failed = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for fp in files:
            date_str = _DATE_RE.match(fp.name).group(1)  # type: ignore
            payload = load_json_dict(fp)
            items = payload.get("items")
            if not isinstance(items, list):
                continue

            risk_watch = build_risk_watch_summary(daily_dir, date_str)

            for it in items:
                if not isinstance(it, dict):
                    continue
                symbol = str(it.get("symbol") or "").strip()
                if not symbol:
                    continue

                sid = stable_sample_id(date_str, symbol, {"etf": it, "risk": risk_watch})
                if sid in existing_ids:
                    skipped += 1
                    continue

                messages = build_teacher_messages(
                    symbol=symbol,
                    date_str=date_str,
                    etf_item=it,
                    risk_watch=risk_watch,
                    include_cot=bool(args.include_cot),
                )

                raw = ""
                parsed: Optional[Dict[str, Any]] = None
                err = ""
                missing: List[str] = []
                extra: List[str] = []

                for attempt in range(1, int(args.max_retries) + 1):
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

                        missing, extra = validate_label(parsed)
                        if missing or extra:
                            raise ValueError(f"Schema mismatch missing={missing} extra={extra}")

                        break
                    except Exception as e:
                        err = str(e)
                        parsed = None
                        if attempt < int(args.max_retries):
                            time.sleep(0.8 * attempt)
                            continue

                out_obj = {
                    "id": sid,
                    "date": date_str,
                    "symbol": symbol,
                    "teacher": {
                        "base_url": str(args.teacher_base_url),
                        "model": str(args.teacher_model),
                        "include_cot": bool(args.include_cot),
                        "max_output_tokens": int(args.max_output_tokens),
                        "temperature": float(args.temperature),
                        "error": err,
                        "missing": missing,
                        "extra": extra,
                    },
                    "input": {
                        "etf_features": it,
                        "risk_watch": risk_watch,
                    },
                    "output": parsed if parsed is not None else {},
                    "raw": raw,
                }

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                fout.flush()

                if parsed is None:
                    failed += 1
                    status = "FAIL"
                else:
                    written += 1
                    status = "OK"
                    existing_ids.add(sid)

                logger.info(
                    f"[{date_str}] {symbol} {status} written={written} failed={failed} skipped={skipped} err={err[:120]}"
                )

                if args.sleep and float(args.sleep) > 0:
                    time.sleep(float(args.sleep))

                if args.max and int(args.max) > 0 and written >= int(args.max):
                    logger.info(f"Reached --max {args.max}, stopping")
                    break

            if args.max and int(args.max) > 0 and written >= int(args.max):
                break

    logger.info(f"Saved teacher dataset: {out_path} ok={written} failed={failed} skipped={skipped}")


if __name__ == "__main__":
    main()
