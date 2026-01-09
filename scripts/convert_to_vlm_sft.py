#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"jsonl not found: {path}")
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


def _load_prompt_yaml(path: Path) -> Tuple[str, str]:
    if not path.exists():
        raise SystemExit(f"prompt yaml not found: {path}")
    try:
        import yaml

        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"failed to read yaml: {path}: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit(f"prompt yaml must be a dict: {path}")
    sp = str(obj.get("system_prompt") or "").strip()
    up = str(obj.get("user_prompt") or "").strip()
    if (not sp) or (not up):
        raise SystemExit(f"prompt yaml missing system_prompt/user_prompt: {path}")
    return sp, up


def _format_user_prompt(template: str, *, ticker: str, asof: str) -> str:
    try:
        return template.format(ticker=str(ticker), asof=str(asof))
    except Exception:
        return template


def _key(ticker: str, asof: str) -> str:
    return f"{str(ticker).upper().strip()}::{str(asof).strip()}"


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--signals-jsonl",
        default="results/phase21_chartist/chart_signals.jsonl",
        help="B-step output jsonl from run_chart_expert.py",
    )
    ap.add_argument(
        "--charts-jsonl",
        default="",
        help="charts_base64.jsonl path (default: inferred from --asof if provided)",
    )
    ap.add_argument(
        "--asof",
        default="",
        help="Optional as-of date YYYY-MM-DD; used to infer charts jsonl as data/charts/<asof>/charts_base64.jsonl",
    )
    ap.add_argument("--prompt-yaml", default="configs/prompts/chartist_prompt.yaml")

    ap.add_argument(
        "--out-jsonl",
        default="data/finetune/vlm/chartist_sft.jsonl",
        help="Output VLM SFT dataset jsonl",
    )

    ap.add_argument("--min-confidence", type=float, default=-1.0)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    signals_path = Path(str(args.signals_jsonl))
    if not signals_path.exists():
        candidates: List[Path] = []
        try:
            root = Path("results")
            if root.exists():
                for p in root.rglob("chart_signals*.jsonl"):
                    if p.is_file():
                        candidates.append(p)
        except Exception:
            candidates = []

        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            picked = candidates[0]
            print(f"[warn] signals jsonl not found: {signals_path}")
            print(f"[warn] auto-pick signals: {picked}")
            for i, p in enumerate(candidates[:10]):
                try:
                    sz = int(p.stat().st_size)
                except Exception:
                    sz = -1
                print(f"[cand {i}] {p} size={sz}")
            signals_path = picked
        else:
            raise SystemExit(f"signals jsonl not found: {signals_path}")

    charts_path_s = str(args.charts_jsonl or "").strip()
    if (not charts_path_s) and str(args.asof or "").strip():
        charts_path_s = str(Path("data") / "charts" / str(args.asof).strip() / "charts_base64.jsonl")
    if not charts_path_s:
        raise SystemExit("Missing --charts-jsonl (or provide --asof to infer default path)")

    charts_path = Path(charts_path_s)
    if not charts_path.exists():
        candidates2: List[Path] = []
        try:
            root2 = Path("data") / "charts"
            if root2.exists():
                for p in root2.rglob("*charts_base64*.jsonl"):
                    if p.is_file():
                        candidates2.append(p)
        except Exception:
            candidates2 = []

        if candidates2:
            candidates2 = sorted(candidates2, key=lambda p: p.stat().st_mtime, reverse=True)
            picked2 = candidates2[0]
            print(f"[warn] charts jsonl not found: {charts_path}")
            print(f"[warn] auto-pick charts: {picked2}")
            for i, p in enumerate(candidates2[:10]):
                try:
                    sz = int(p.stat().st_size)
                except Exception:
                    sz = -1
                print(f"[cand2 {i}] {p} size={sz}")
            charts_path = picked2
        else:
            raise SystemExit(f"jsonl not found: {charts_path}")
    prompt_path = Path(str(args.prompt_yaml))

    system_prompt, user_template = _load_prompt_yaml(prompt_path)

    charts = _read_jsonl(charts_path)
    chart_map: Dict[str, Dict[str, Any]] = {}
    for rec in charts:
        ticker = str(rec.get("ticker") or "").upper().strip()
        asof = str(rec.get("asof") or "").strip()
        b64 = rec.get("image_base64")
        if not ticker or not asof or (not isinstance(b64, str)) or (not b64.strip()):
            continue
        chart_map[_key(ticker, asof)] = rec

    signals = _read_jsonl(signals_path)
    if int(args.limit) > 0:
        signals = signals[: int(args.limit)]

    out_rows: List[Dict[str, Any]] = []

    min_conf = float(args.min_confidence)

    for rec in signals:
        ticker = str(rec.get("ticker") or "").upper().strip()
        asof = str(rec.get("asof") or "").strip()
        if not ticker or not asof:
            continue

        cm = chart_map.get(_key(ticker, asof))
        if not cm:
            continue
        image_b64 = str(cm.get("image_base64") or "").strip()
        if not image_b64:
            continue

        signal = str(rec.get("signal") or "").strip().upper()
        try:
            conf = float(rec.get("confidence"))
        except Exception:
            conf = 0.0
        reasoning = rec.get("reasoning")
        if reasoning is None:
            reasoning = ""
        reasoning_s = str(reasoning).strip()

        if not signal:
            continue
        if min_conf >= 0.0 and conf < min_conf:
            continue

        assistant_obj = {"signal": signal, "confidence": float(conf), "reasoning": reasoning_s}
        assistant_text = json.dumps(assistant_obj, ensure_ascii=False)

        user_prompt = _format_user_prompt(user_template, ticker=ticker, asof=asof)

        conversations = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_base64", "image_base64": image_b64},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": assistant_text},
        ]

        out_rows.append(
            {
                "id": f"{ticker}_{asof}",
                "ticker": ticker,
                "asof": asof,
                "image_base64": image_b64,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant": assistant_text,
                "conversations": conversations,
            }
        )

    out_path = Path(str(args.out_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "signals": str(signals_path.as_posix()),
                "charts": str(charts_path.as_posix()),
                "prompt_yaml": str(prompt_path.as_posix()),
                "out": str(out_path.as_posix()),
                "charts_indexed": int(len(chart_map)),
                "out_rows": int(len(out_rows)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
