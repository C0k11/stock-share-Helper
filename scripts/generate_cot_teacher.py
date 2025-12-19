#!/usr/bin/env python
"""
Phase 10.2: Generate Chain-of-Thought (CoT) corrections using DeepSeek-R1 / V3 as teacher.

Usage:
    python scripts/generate_cot_teacher.py \
        --in data/finetune/mistakes_100.jsonl \
        --out data/finetune/cot_mistakes_100.jsonl \
        --model deepseek-reasoner

Environment variables:
    DEEPSEEK_API_KEY   - Required
    DEEPSEEK_BASE_URL  - Optional (default: https://api.deepseek.com)
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


_ENV_LOCAL_KEYS: List[str] = []

SYSTEM_PROMPT = """\
You are a senior quantitative trader reviewing a past trading mistake.
Given the original decision, the actual market outcome, the news context, and technical features,
analyze why the decision was wrong and provide the CORRECT decision with structured reasoning.

Output STRICT JSON only (no markdown, no extra text):
{
  "decision": "BUY" | "SELL" | "HOLD",
  "analysis": "<corrected summary, max 30 words>",
  "reasoning_trace": [
    "1. <first key insight>",
    "2. <second key insight>",
    "3. <third key insight>"
  ]
}

Rules:
- reasoning_trace must have exactly 3 bullet points
- Each bullet point max 25 words
- decision must be BUY, SELL, or HOLD
- Do NOT output anything except the JSON object
"""


def _load_env_local(repo_root: Path) -> None:
    _ENV_LOCAL_KEYS.clear()
    fp = repo_root / ".env.local"
    if not fp.exists():
        return

    try:
        raw = fp.read_bytes()
    except Exception:
        return

    text: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue

        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        if line.lower().startswith("export "):
            line = line[len("export ") :].strip()
        if line.lower().startswith("set "):
            line = line[len("set ") :].strip()

        if line.startswith("$env:"):
            line = line[len("$env:") :].strip()

        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue

        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        v = v.rstrip(";").strip()

        key_alias = {
            "DEEPSEEK_APIKEY": "DEEPSEEK_API_KEY",
            "DEEPSEEK_KEY": "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "OPENAI_BASE_URL": "OPENAI_BASE_URL",
        }
        k_norm = key_alias.get(k.strip(), k.strip())
        _ENV_LOCAL_KEYS.append(k_norm)
        if k_norm not in os.environ:
            os.environ[k_norm] = v


def _build_user_prompt(rec: Dict[str, Any]) -> str:
    date = rec.get("date", "?")
    ticker = rec.get("ticker", "?")
    orig_decision = rec.get("decision", "?")
    orig_analysis = rec.get("analysis", "N/A")
    fwd_ret = rec.get("forward_return", 0.0)
    news_top = rec.get("news_top", [])
    features = rec.get("features", {})

    news_lines = []
    for i, n in enumerate(news_top[:3], 1):
        title = n.get("title", "")
        event_type = n.get("event_type", "")
        impact = n.get("impact_equity", 0)
        summary = n.get("summary", "")
        news_lines.append(f"{i}. [{event_type}] {title} (impact={impact})\n   {summary}")
    news_block = "\n".join(news_lines) if news_lines else "No significant news."

    tech = features.get("technical", {})
    tech_lines = []
    for k in ["close", "price_vs_ma20", "price_vs_ma200", "return_5d", "return_21d", "volatility_20d"]:
        if k in tech:
            tech_lines.append(f"{k}={tech[k]}")
    tech_block = ", ".join(tech_lines) if tech_lines else "N/A"

    regime = features.get("market_regime", {})
    regime_str = f"{regime.get('regime', '?')} (score={regime.get('score', '?')})" if regime else "N/A"

    return f"""\
Date: {date}
Ticker: {ticker}
Original Decision: {orig_decision}
Original Analysis: {orig_analysis}
Actual Forward Return: {fwd_ret:.4f} ({'+' if fwd_ret >= 0 else ''}{fwd_ret*100:.2f}%)

News Context (top 3):
{news_block}

Technical Snapshot:
{tech_block}

Market Regime: {regime_str}

Based on the above, the original {orig_decision} was WRONG.
Provide the CORRECT decision and reasoning in STRICT JSON format.
"""


def _parse_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    if (not text.startswith("{")) or (not text.endswith("}")):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None
    if "decision" not in obj:
        return None

    decision = str(obj.get("decision", "")).strip().upper()
    if decision not in {"BUY", "SELL", "HOLD"}:
        return None

    analysis = str(obj.get("analysis", ""))[:200]
    reasoning = obj.get("reasoning_trace", [])
    if not isinstance(reasoning, list):
        reasoning = []
    reasoning = [str(r)[:150] for r in reasoning[:5]]

    return {
        "decision": decision,
        "analysis": analysis,
        "reasoning_trace": reasoning,
    }


def _post_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    response_format: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    base = str(base_url).rstrip("/")
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if isinstance(response_format, dict) and response_format:
        payload["response_format"] = response_format
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
    except Exception as e:
        raise RuntimeError(str(e))

    try:
        data = json.loads(raw)
    except Exception:
        raise RuntimeError(f"Non-JSON response: {raw[:500]}")

    try:
        choice0 = data["choices"][0]
        msg = choice0.get("message") if isinstance(choice0, dict) else None
        if not isinstance(msg, dict):
            raise RuntimeError("missing_message")

        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content

        for k in ("reasoning_content", "reasoning", "thought", "output_text"):
            v = msg.get(k)
            if isinstance(v, str) and v.strip():
                return v

        v = choice0.get("text") if isinstance(choice0, dict) else None
        if isinstance(v, str) and v.strip():
            return v

        raise RuntimeError(f"empty_content keys={sorted(list(msg.keys()))}")
    except Exception:
        raise RuntimeError(f"Unexpected response schema: {raw[:500]}")


def _call_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    json_mode: bool,
    max_retries: int = 3,
    timeout: float = 60.0,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            response_format = {"type": "json_object"} if bool(json_mode) else None
            content = _post_chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0 if "reasoner" in str(model).lower() else 0.3,
                max_tokens=512,
                timeout=timeout,
                response_format=response_format,
            ) or ""
            parsed = _parse_response(content)
            if parsed:
                return parsed
            print(f"  [warn] parse failed on attempt {attempt+1}, raw={content[:200]}")
        except Exception as e:
            print(f"  [warn] API error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Generate CoT corrections using DeepSeek teacher")
    p.add_argument("--in", dest="input", required=True, help="Input JSONL (mistakes)")
    p.add_argument("--out", required=True, help="Output JSONL (with CoT)")
    p.add_argument("--model", default=None, help="Model name (overrides TEACHER_MODEL)")
    p.add_argument("--fallback-model", default="deepseek-chat")
    p.add_argument("--json-mode", action="store_true")
    p.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    p.set_defaults(json_mode=True)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--delay", type=float, default=1.0, help="Delay between requests (rate limit)")
    p.add_argument("--debug-env", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not os.environ.get("DEEPSEEK_API_KEY"):
        _load_env_local(repo_root)

    if not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("TEACHER_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["TEACHER_API_KEY"]
    if not os.environ.get("DEEPSEEK_BASE_URL") and os.environ.get("TEACHER_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = os.environ["TEACHER_BASE_URL"]

    if not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if not os.environ.get("DEEPSEEK_BASE_URL") and os.environ.get("OPENAI_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = os.environ["OPENAI_BASE_URL"]

    if not args.model:
        if os.environ.get("TEACHER_MODEL"):
            args.model = str(os.environ.get("TEACHER_MODEL"))
        else:
            args.model = "deepseek-reasoner"

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        env_fp = repo_root / ".env.local"
        if args.debug_env:
            diag = {
                "env_file": str(env_fp),
                "env_file_exists": env_fp.exists(),
                "env_local_keys_found": sorted(set(_ENV_LOCAL_KEYS)),
                "has_DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
                "has_OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
                "has_DEEPSEEK_BASE_URL": bool(os.environ.get("DEEPSEEK_BASE_URL")),
                "has_OPENAI_BASE_URL": bool(os.environ.get("OPENAI_BASE_URL")),
            }
            print(json.dumps(diag, ensure_ascii=False))
        raise SystemExit(
            "DEEPSEEK_API_KEY environment variable not set. "
            "Set DEEPSEEK_API_KEY in your environment or in .env.local (supported forms: KEY=VALUE, export KEY=VALUE, set KEY=VALUE, $env:KEY=VALUE). "
            f"Checked: {env_fp}"
        )

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    records: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    done_keys: set = set()
    existing: List[Dict[str, Any]] = []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    existing.append(obj)
                    key = (obj.get("date"), obj.get("ticker"))
                    done_keys.add(key)
        print(f"Resuming: {len(done_keys)} already processed")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    fail = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            key = (rec.get("date"), rec.get("ticker"))
            if key in done_keys:
                continue

            print(f"[{i+1}/{len(records)}] {rec.get('date')} {rec.get('ticker')} ...")

            user_prompt = _build_user_prompt(rec)
            result = _call_api(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                user_prompt=user_prompt,
                json_mode=bool(args.json_mode),
                max_retries=args.max_retries,
                timeout=args.timeout,
            )

            if (not result) and str(args.fallback_model or "").strip() and str(args.fallback_model).strip() != str(args.model):
                result = _call_api(
                    base_url=base_url,
                    api_key=api_key,
                    model=str(args.fallback_model).strip(),
                    user_prompt=user_prompt,
                    json_mode=bool(args.json_mode),
                    max_retries=max(1, int(args.max_retries)),
                    timeout=args.timeout,
                )

            if result:
                out_rec = {
                    "date": rec.get("date"),
                    "ticker": rec.get("ticker"),
                    "original_decision": rec.get("decision"),
                    "original_analysis": rec.get("analysis"),
                    "forward_return": rec.get("forward_return"),
                    "corrected": result,
                    "news_top": rec.get("news_top", []),
                    "features": rec.get("features", {}),
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
                print(f"  -> {result['decision']}: {result['analysis'][:60]}...")
            else:
                fail += 1
                print(f"  -> FAILED after retries")

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone: success={success}, fail={fail}, total_output={len(done_keys)+success}")


if __name__ == "__main__":
    main()
