#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_prompt_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Prompt yaml not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read prompt yaml: {path}") from e

    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid prompt yaml format: {path}")

    sp = payload.get("system_prompt")
    up = payload.get("user_prompt")
    if not isinstance(sp, str) or not sp.strip():
        raise SystemExit(f"Missing system_prompt in: {path}")
    if not isinstance(up, str) or not up.strip():
        raise SystemExit(f"Missing user_prompt in: {path}")

    return {"system_prompt": sp, "user_prompt": up}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input jsonl not found: {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _extract_image_data_url(rec: Dict[str, Any]) -> str:
    b64 = rec.get("image_base64")
    if not isinstance(b64, str) or not b64.strip():
        return ""
    return f"data:image/png;base64,{b64.strip()}"


def _format_user_prompt(template: str, *, ticker: str, asof: str) -> str:
    try:
        return template.format(ticker=str(ticker), asof=str(asof))
    except Exception:
        return template


def _call_vlm(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    max_tokens: int,
    temperature: float,
) -> str:
    # Uses OpenAI-compatible Chat Completions with image_url content.
    resp = client.chat.completions.create(
        model=str(model),
        messages=[
            {"role": "system", "content": str(system_prompt)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(user_prompt)},
                    {"type": "image_url", "image_url": {"url": str(image_data_url)}},
                ],
            },
        ],
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )

    try:
        return str(resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _dry_run_response(*, ticker: str, asof: str) -> Dict[str, Any]:
    return {
        "signal": "NEUTRAL",
        "confidence": 0.0,
        "reasoning": f"dry_run: no VLM call executed for {ticker} {asof}",
    }


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()

    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to extract first JSON object substring
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        sub = s[i : j + 1]
        try:
            obj = json.loads(sub)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-jsonl",
        default="",
        help="charts_base64.jsonl path (default: data/charts/<asof>/charts_base64.jsonl if --asof provided)",
    )
    ap.add_argument("--asof", default="", help="Optional as-of date YYYY-MM-DD used to build default input path")

    ap.add_argument("--out-jsonl", default="results/phase21_chartist/chart_signals.jsonl")

    ap.add_argument("--prompt-yaml", default="configs/prompts/chartist_prompt.yaml")

    ap.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen2-VL-7B-Instruct"))

    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--dry-run", action="store_true", default=False)

    ap.add_argument("--limit", type=int, default=0, help="Optional limit N records")
    ap.add_argument("--verbose", action="store_true", default=False)

    args = ap.parse_args()

    input_path = str(args.input_jsonl or "").strip()
    if not input_path and str(args.asof or "").strip():
        input_path = str(Path("data") / "charts" / str(args.asof).strip() / "charts_base64.jsonl")
    if not input_path:
        raise SystemExit("Missing --input-jsonl (or provide --asof to use default path)")

    prompt_cfg = _load_prompt_yaml(Path(str(args.prompt_yaml)))
    system_prompt = prompt_cfg["system_prompt"]
    user_template = prompt_cfg["user_prompt"]

    items = _read_jsonl(Path(input_path))
    if int(args.limit) > 0:
        items = items[: int(args.limit)]

    client: Any = None
    if not bool(args.dry_run):
        try:
            from openai import OpenAI
        except Exception as e:
            raise SystemExit("openai python client is required. Install it via requirements.txt") from e

        client = OpenAI(base_url=str(args.api_base), api_key=str(args.api_key))

    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0

    with out_path.open("w", encoding="utf-8") as f:
        for rec in items:
            ticker = str(rec.get("ticker") or "").upper().strip()
            asof = str(rec.get("asof") or str(args.asof or "")).strip()
            img_url = _extract_image_data_url(rec)

            if not ticker or not img_url:
                failed += 1
                continue

            user_prompt = _format_user_prompt(user_template, ticker=ticker, asof=asof)

            if bool(args.dry_run):
                parsed = _dry_run_response(ticker=ticker, asof=asof)
                text = json.dumps(parsed, ensure_ascii=False)
            else:
                try:
                    text = _call_vlm(
                        client=client,
                        model=str(args.model),
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_data_url=img_url,
                        max_tokens=int(args.max_tokens),
                        temperature=float(args.temperature),
                    )
                except Exception as e:
                    failed += 1
                    if bool(args.verbose):
                        f.write(
                            json.dumps(
                                {
                                    "ticker": ticker,
                                    "asof": asof,
                                    "ok": False,
                                    "error": repr(e),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                parsed = _try_parse_json(text)
                if parsed is None:
                    failed += 1
                    if bool(args.verbose):
                        f.write(
                            json.dumps(
                                {
                                    "ticker": ticker,
                                    "asof": asof,
                                    "ok": False,
                                    "raw": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

            out_rec: Dict[str, Any] = {
                "ticker": ticker,
                "asof": asof,
                "ok": True,
                "model": str(args.model),
                "dry_run": bool(args.dry_run),
                "signal": parsed.get("signal"),
                "confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning"),
                "raw": text,
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            ok += 1

    print(json.dumps({"input": input_path, "out": str(out_path.as_posix()), "ok": ok, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
