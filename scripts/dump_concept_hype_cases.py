#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _date_from_filename(p: Path) -> str:
    name = p.stem
    for prefix in ("signals_full_", "signals_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_signal_files(data_dir: Path) -> List[Path]:
    files_full = sorted(list(data_dir.glob("signals_full_*.json")))
    files_simple_all = sorted(list(data_dir.glob("signals_*.json")))
    files_simple = [p for p in files_simple_all if not p.name.startswith("signals_full_")]

    chosen_by_date: Dict[str, Path] = {}
    for fp in files_simple:
        chosen_by_date[_date_from_filename(fp)] = fp
    for fp in files_full:
        chosen_by_date[_date_from_filename(fp)] = fp

    return [chosen_by_date[k] for k in sorted(chosen_by_date.keys())]


def _extract_case(fp: Path, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sig = record.get("signal") if isinstance(record.get("signal"), dict) else record
    if not isinstance(sig, dict):
        return None

    if str(sig.get("event_type") or "").strip() != "concept_hype":
        return None

    title = (
        (record.get("title") if isinstance(record.get("title"), str) else None)
        or (sig.get("title") if isinstance(sig.get("title"), str) else None)
        or ""
    ).strip()

    summary = (
        (record.get("summary") if isinstance(record.get("summary"), str) else None)
        or (sig.get("summary") if isinstance(sig.get("summary"), str) else None)
        or ""
    ).strip()

    source = (
        (record.get("source") if isinstance(record.get("source"), str) else None)
        or (sig.get("source") if isinstance(sig.get("source"), str) else None)
        or "Unknown"
    ).strip()

    assets = sig.get("related_assets")
    if assets is None:
        assets = sig.get("assets")

    impact = sig.get("impact_equity", record.get("impact_equity", 0))

    return {
        "file": fp.name,
        "date": _date_from_filename(fp),
        "source": source,
        "assets": assets if isinstance(assets, list) else [],
        "title": title,
        "summary": summary,
        "impact_equity": impact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample concept_hype cases from signals*.json")
    parser.add_argument("--n", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0=none)")
    parser.add_argument("--data-dir", default="data/daily", help="Daily data dir")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = _pick_signal_files(data_dir)

    print(f"Scanning {len(files)} files for 'concept_hype'...")

    cases: List[Dict[str, Any]] = []
    for fp in files:
        obj = _load_json(fp)
        if not isinstance(obj, list):
            continue

        for it in obj:
            if not isinstance(it, dict):
                continue
            c = _extract_case(fp, it)
            if c is not None:
                cases.append(c)

    print(f"Found {len(cases)} concept_hype cases.")
    if not cases:
        return

    if args.seed and args.seed != 0:
        random.seed(int(args.seed))

    sample_size = min(max(1, int(args.n)), len(cases))
    samples = random.sample(cases, sample_size)

    print(f"\n=== Case Study: {sample_size} Random Samples ===")
    print("-" * 80)

    for i, c in enumerate(samples, 1):
        print(f"[{i}] {c['date']} | Src: {c['source']} | Impact: {c['impact_equity']} | File: {c['file']}")
        if c["assets"]:
            print(f"    Assets: {c['assets']}")
        if c["title"]:
            print(f"    Title: {c['title']}")
        if c["summary"]:
            s = c["summary"].replace("\n", " ").strip()
            if len(s) > 220:
                s = s[:220] + "..."
            print(f"    Summary: {s}")
        print("-" * 80)


if __name__ == "__main__":
    main()
