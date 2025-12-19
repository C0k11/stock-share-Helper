#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_news_context_from_signal_item(it: Dict[str, Any]) -> Optional[str]:
    sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
    if sig is None:
        return None

    et = str(sig.get("event_type") or "").strip()
    if not et:
        return None

    sent = str(sig.get("sentiment") or "").strip()
    impact_eq = sig.get("impact_equity")
    impact_bond = sig.get("impact_bond")
    impact_gold = sig.get("impact_gold")
    summary = str(sig.get("summary") or "").strip()

    # If the signal has no impact and no summary, treat as no context.
    if (summary == "") and (_safe_float(impact_eq, 0.0) == 0.0) and (_safe_float(impact_bond, 0.0) == 0.0) and (_safe_float(impact_gold, 0.0) == 0.0):
        return None

    lines = []
    lines.append(f"Market News Context:")
    lines.append(f"EventType: {et}")
    if sent:
        lines.append(f"Sentiment: {sent}")
    lines.append(f"ImpactEquity: {impact_eq}")
    lines.append(f"ImpactBond: {impact_bond}")
    lines.append(f"ImpactGold: {impact_gold}")
    if summary:
        lines.append(f"Summary: {summary}")

    return "\n".join(lines)


def index_signals_by_date(signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for it in signals:
        if not isinstance(it, dict):
            continue
        # run_daily_inference writes published_at for news item; file date is in filename.
        # We allow users to provide a single signals file per day.
        dt = str(it.get("published_at") or "")
        # If published_at missing, skip; in that case, use file-level date (handled by caller).
        if not dt:
            continue
        day = dt[:10]
        if len(day) != 10:
            continue
        out.setdefault(day, []).append(it)
    return out


def iter_signals_files(signals_dir: Path) -> List[Path]:
    if not signals_dir.exists() or not signals_dir.is_dir():
        return []
    return sorted(signals_dir.glob("signals_????-??-??.json"))


def infer_day_from_signals_filename(path: Path) -> Optional[str]:
    name = path.name
    if not (name.startswith("signals_") and name.endswith(".json")):
        return None
    day = name[len("signals_") : len("signals_") + 10]
    if len(day) == 10 and day[4] == "-" and day[7] == "-":
        return day
    return None


def build_pool_by_day_from_signals_dir(
    *,
    signals_dir: Path,
    min_abs_impact: float,
    max_signals_per_day: int,
) -> Dict[str, List[str]]:
    pool_by_day: Dict[str, List[str]] = {}

    files = iter_signals_files(signals_dir)
    for fp in files:
        day = infer_day_from_signals_filename(fp)
        if not day:
            continue

        try:
            items = _load_json_list(fp)
        except Exception:
            continue

        candidates: List[Tuple[float, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if not it.get("parse_ok"):
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
            if sig is None:
                continue
            impact = _safe_float(sig.get("impact_equity"), 0.0)
            if abs(impact) < float(min_abs_impact):
                continue
            ctx = build_news_context_from_signal_item(it)
            if not ctx:
                continue
            candidates.append((abs(impact), ctx))

        if not candidates:
            continue

        candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
        k = max(1, int(max_signals_per_day))
        pool_by_day[day] = [c for _score, c in candidates_sorted[:k]]

    return pool_by_day


def inject_context_into_prompt(user_text: str, contexts: List[str]) -> str:
    if not contexts:
        return user_text
    addon = "\n\n" + "\n\n".join(contexts) + "\n\n"
    # Prefer to prefix so the model sees news context before technical prompt.
    return addon + user_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject daily news signals into Trader SFT dataset prompts")
    parser.add_argument("--in", dest="in_path", default="data/finetune/trader_stock_sft_v1.json")
    parser.add_argument("--signals", dest="signals_path", default="", help="Single signals_YYYY-MM-DD.json to inject")
    parser.add_argument("--signals-dir", default="data/daily", help="Directory containing signals_YYYY-MM-DD.json")
    parser.add_argument("--out", dest="out_path", default="data/finetune/trader_stock_sft_v1_plus_news.json")

    parser.add_argument("--date", default="", help="If set, inject only this day (YYYY-MM-DD) and read signals_YYYY-MM-DD.json from --signals-dir")
    parser.add_argument("--min-abs-impact", type=float, default=0.5, help="Only inject signals with |impact_equity| >= threshold")
    parser.add_argument("--max-signals-per-day", type=int, default=3)

    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    dataset = _load_json_list(in_path)

    # Load signals
    signals: List[Dict[str, Any]] = []
    signals_day: Optional[str] = None

    if args.signals_path:
        signals_path = Path(args.signals_path)
        signals = _load_json_list(signals_path)
        # attempt to infer date from filename
        name = signals_path.name
        if name.startswith("signals_") and name.endswith(".json") and len(name) >= len("signals_YYYY-MM-DD.json"):
            signals_day = name[len("signals_") : len("signals_") + 10]
    elif args.date:
        signals_day = str(args.date)
        signals_path = Path(args.signals_dir) / f"signals_{signals_day}.json"
        signals = _load_json_list(signals_path)
    else:
        signals_dir = Path(args.signals_dir)
        pool_by_day = build_pool_by_day_from_signals_dir(
            signals_dir=signals_dir,
            min_abs_impact=float(args.min_abs_impact),
            max_signals_per_day=int(args.max_signals_per_day),
        )

        injected = 0
        for item in dataset:
            convs = item.get("conversations")
            if not isinstance(convs, list) or len(convs) < 3:
                continue

            user = convs[1]
            if not isinstance(user, dict):
                continue

            txt = str(user.get("value") or "")

            day = None
            marker = " on "
            pos = txt.find(marker)
            if pos >= 0:
                maybe = txt[pos + len(marker) : pos + len(marker) + 10]
                if len(maybe) == 10 and maybe[4] == "-" and maybe[7] == "-":
                    day = maybe

            if day is None:
                continue

            contexts = pool_by_day.get(day) or []
            if not contexts:
                continue

            user["value"] = inject_context_into_prompt(txt, contexts)
            injected += 1

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(
            json.dumps(
                {
                    "in": str(in_path),
                    "out": str(out_path),
                    "signals_mode": "signals_dir",
                    "signals_dir": str(Path(args.signals_dir)),
                    "min_abs_impact": float(args.min_abs_impact),
                    "max_signals_per_day": int(args.max_signals_per_day),
                    "injected_samples": int(injected),
                    "total_samples": int(len(dataset)),
                    "covered_days": int(len(pool_by_day)),
                },
                ensure_ascii=False,
            )
        )
        return

    # Build per-day injection pool (single day mode)
    pool_by_day: Dict[str, List[str]] = {}
    if signals_day:
        candidates: List[Tuple[float, str]] = []
        for it in signals:
            if not isinstance(it, dict):
                continue
            if not it.get("parse_ok"):
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
            if sig is None:
                continue
            impact = _safe_float(sig.get("impact_equity"), 0.0)
            if abs(impact) < float(args.min_abs_impact):
                continue
            ctx = build_news_context_from_signal_item(it)
            if ctx:
                candidates.append((abs(impact), ctx))
        candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
        k = max(1, int(args.max_signals_per_day))
        pool_by_day[signals_day] = [c for _score, c in candidates_sorted[:k]]
    else:
        by_day = index_signals_by_date(signals)
        if not by_day:
            raise SystemExit("No usable published_at in signals")
        for day, items in by_day.items():
            candidates = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                if not it.get("parse_ok"):
                    continue
                sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
                if sig is None:
                    continue
                impact = _safe_float(sig.get("impact_equity"), 0.0)
                if abs(impact) < float(args.min_abs_impact):
                    continue
                ctx = build_news_context_from_signal_item(it)
                if ctx:
                    candidates.append(ctx)
            if candidates:
                pool_by_day[day] = candidates

    injected = 0

    for item in dataset:
        convs = item.get("conversations")
        if not isinstance(convs, list) or len(convs) < 3:
            continue

        user = convs[1]
        if not isinstance(user, dict):
            continue
        txt = str(user.get("value") or "")

        # Extract date from the known prompt template: "on YYYY-MM-DD:"
        day = None
        marker = " on "
        pos = txt.find(marker)
        if pos >= 0:
            maybe = txt[pos + len(marker) : pos + len(marker) + 10]
            if len(maybe) == 10 and maybe[4] == "-" and maybe[7] == "-":
                day = maybe

        if day is None:
            continue

        candidates = pool_by_day.get(day) or []
        if not candidates:
            continue

        # Inject top-K (stable order)
        k = max(1, int(args.max_signals_per_day))
        user["value"] = inject_context_into_prompt(txt, candidates[:k])
        injected += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "in": str(in_path),
                "out": str(out_path),
                "signals_day": signals_day,
                "min_abs_impact": float(args.min_abs_impact),
                "max_signals_per_day": int(args.max_signals_per_day),
                "injected_samples": int(injected),
                "total_samples": int(len(dataset)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
