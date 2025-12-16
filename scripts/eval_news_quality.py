#!/usr/bin/env python

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ScanStats:
    total_files: int = 0
    unreadable_files: int = 0
    empty_files: int = 0
    total_records: int = 0
    parse_ok_true: int = 0
    parse_ok_false: int = 0


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_signal_like_records(obj: Any) -> Iterable[Tuple[Optional[bool], Optional[Dict[str, Any]]]]:
    if not isinstance(obj, list):
        return

    for it in obj:
        if not isinstance(it, dict):
            continue

        if "signal" in it:
            sig = it.get("signal")
            if isinstance(sig, dict):
                yield (it.get("parse_ok") if isinstance(it.get("parse_ok"), bool) else None), sig
            else:
                yield (it.get("parse_ok") if isinstance(it.get("parse_ok"), bool) else None), None
        else:
            yield None, it


def _date_from_filename(p: Path) -> str:
    name = p.stem
    for prefix in ("signals_full_", "signals_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def scan_signals(data_dir: Path) -> None:
    files_full = sorted(list(data_dir.glob("signals_full_*.json")))
    files_simple_all = sorted(list(data_dir.glob("signals_*.json")))
    files_simple = [p for p in files_simple_all if not p.name.startswith("signals_full_")]

    chosen_by_date: Dict[str, Path] = {}
    dup_dates: List[str] = []
    for fp in files_simple:
        chosen_by_date[_date_from_filename(fp)] = fp
    for fp in files_full:
        d = _date_from_filename(fp)
        if d in chosen_by_date:
            dup_dates.append(d)
        chosen_by_date[d] = fp

    files = [chosen_by_date[k] for k in sorted(chosen_by_date.keys())]

    if not files:
        print("No signal files found under data/daily (signals_full_*.json or signals_*.json)")
        return

    stats = ScanStats()
    event_types: Counter[str] = Counter()
    impact_equity: Counter[str] = Counter()
    impact_bond: Counter[str] = Counter()
    impact_gold: Counter[str] = Counter()
    per_day: List[Dict[str, Any]] = []

    print(f"Scanning {len(files)} signal files under {data_dir} ...")
    print(f"signals_full_*.json: {len(files_full)}")
    print(f"signals_*.json:      {len(files_simple)}")
    if dup_dates:
        print(f"Dedup (prefer signals_full): {len(dup_dates)} dates -> {sorted(set(dup_dates))}")
    print("Files:")
    for fp in files:
        print(f"- {fp.name}")

    for fp in files:
        stats.total_files += 1
        obj = _load_json(fp)
        if obj is None:
            stats.unreadable_files += 1
            per_day.append({"date": _date_from_filename(fp), "count": 0, "parse_ok_rate": None})
            continue

        if not obj:
            stats.empty_files += 1
            per_day.append({"date": _date_from_filename(fp), "count": 0, "parse_ok_rate": None})
            continue

        day_count = 0
        ok_cnt = 0
        ok_den = 0

        for ok, sig in _iter_signal_like_records(obj):
            day_count += 1
            stats.total_records += 1

            if ok is True:
                stats.parse_ok_true += 1
                ok_cnt += 1
                ok_den += 1
            elif ok is False:
                stats.parse_ok_false += 1
                ok_den += 1

            if not isinstance(sig, dict):
                continue

            et = str(sig.get("event_type") or "unknown").strip() or "unknown"
            event_types[et] += 1

            impact_equity[str(sig.get("impact_equity", "N/A"))] += 1
            impact_bond[str(sig.get("impact_bond", "N/A"))] += 1
            impact_gold[str(sig.get("impact_gold", "N/A"))] += 1

        parse_ok_rate = (ok_cnt / ok_den) if ok_den > 0 else None
        per_day.append({"date": _date_from_filename(fp), "count": day_count, "parse_ok_rate": parse_ok_rate})

    print("\n=== News Signals Health Report ===")
    print(f"Files scanned:     {stats.total_files}")
    print(f"Unreadable files:  {stats.unreadable_files}")
    print(f"Empty files:       {stats.empty_files}")
    print(f"Total records:     {stats.total_records}")

    ok_total = stats.parse_ok_true + stats.parse_ok_false
    if ok_total > 0:
        print(f"Parse OK rate:     {stats.parse_ok_true}/{ok_total} = {stats.parse_ok_true / ok_total:.2%}")
    else:
        print("Parse OK rate:     N/A (no parse_ok field found)")

    if stats.total_files > 0:
        print(f"Avg records/day:   {stats.total_records / stats.total_files:.1f}")

    print("\n--- Event Type Distribution (Top 15) ---")
    for k, v in event_types.most_common(15):
        print(f"{k:<28} {v}")

    print("\n--- Impact Distribution (equity) ---")
    for k, v in impact_equity.items():
        print(f"impact_equity={k:<6} {v}")

    print("\n--- Impact Distribution (bond) ---")
    for k, v in impact_bond.items():
        print(f"impact_bond={k:<6} {v}")

    print("\n--- Impact Distribution (gold) ---")
    for k, v in impact_gold.items():
        print(f"impact_gold={k:<6} {v}")

    counts = [x["count"] for x in per_day if isinstance(x.get("count"), int)]
    if counts:
        print("\n--- Stability Check ---")
        print(f"Min records/day:   {min(counts)}")
        print(f"Max records/day:   {max(counts)}")

    out = data_dir / "news_quality_report.json"
    out.write_text(json.dumps({"summary": stats.__dict__, "per_day": per_day, "event_types": dict(event_types)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


def main() -> None:
    data_dir = Path("data/daily")
    scan_signals(data_dir)


if __name__ == "__main__":
    main()
