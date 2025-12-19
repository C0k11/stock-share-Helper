#!/usr/bin/env python

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    out: List[Dict[str, Any]] = []
    for it in data:
        if isinstance(it, dict):
            out.append(it)
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


@dataclass
class DayScore:
    day: str
    file: str
    strong_count: int
    parse_ok_count: int
    sum_abs_impact: float
    max_abs_impact: float
    snippets: List[str]


def iter_signals_files(signals_dir: Path) -> List[Path]:
    if not signals_dir.exists() or not signals_dir.is_dir():
        return []
    return sorted(signals_dir.glob("signals_????-??-??.json"))


def day_from_filename(path: Path) -> Optional[str]:
    name = path.name
    if not (name.startswith("signals_") and name.endswith(".json")):
        return None
    day = name[len("signals_") : len("signals_") + 10]
    if len(day) == 10 and day[4] == "-" and day[7] == "-":
        return day
    return None


def score_day(
    *,
    path: Path,
    min_abs_impact: float,
    max_snippets: int,
) -> Optional[DayScore]:
    day = day_from_filename(path)
    if not day:
        return None

    try:
        items = _load_json_list(path)
    except Exception:
        return None

    parse_ok = 0
    strong = 0
    sum_abs = 0.0
    max_abs = 0.0

    snippets_scored: List[Tuple[float, str]] = []

    for it in items:
        if not it.get("parse_ok"):
            continue
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
        if sig is None:
            continue

        parse_ok += 1
        impact = _safe_float(sig.get("impact_equity"), 0.0)
        a = abs(impact)
        sum_abs += a
        if a > max_abs:
            max_abs = a

        if a >= float(min_abs_impact):
            strong += 1

            evt = _safe_str(sig.get("event_type"))
            summary = _safe_str(sig.get("summary"))
            if summary:
                if len(summary) > 120:
                    summary = summary[:120] + "..."
            snippet = f"{evt} | impact_equity={impact:g} | {summary}".strip()
            snippets_scored.append((a, snippet))

    snippets_scored.sort(key=lambda x: x[0], reverse=True)
    snippets = [s for _a, s in snippets_scored[: max(0, int(max_snippets))]]

    return DayScore(
        day=day,
        file=str(path),
        strong_count=int(strong),
        parse_ok_count=int(parse_ok),
        sum_abs_impact=float(sum_abs),
        max_abs_impact=float(max_abs),
        snippets=snippets,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Find strongest-news trading day from signals_YYYY-MM-DD.json")
    parser.add_argument("--signals-dir", default="data/daily")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-abs-impact", type=float, default=0.5)
    parser.add_argument("--max-snippets", type=int, default=3)
    args = parser.parse_args()

    signals_dir = Path(args.signals_dir)
    files = iter_signals_files(signals_dir)

    scores: List[DayScore] = []
    for fp in files:
        s = score_day(path=fp, min_abs_impact=float(args.min_abs_impact), max_snippets=int(args.max_snippets))
        if s is None:
            continue
        scores.append(s)

    scores.sort(key=lambda x: (x.sum_abs_impact, x.strong_count, x.max_abs_impact), reverse=True)
    topk = scores[: max(1, int(args.top))]

    out = {
        "signals_dir": str(signals_dir),
        "min_abs_impact": float(args.min_abs_impact),
        "ranked_days": [
            {
                "day": s.day,
                "strong_count": s.strong_count,
                "parse_ok_count": s.parse_ok_count,
                "sum_abs_impact": round(s.sum_abs_impact, 6),
                "max_abs_impact": round(s.max_abs_impact, 6),
                "file": s.file,
                "snippets": s.snippets,
            }
            for s in topk
        ],
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
