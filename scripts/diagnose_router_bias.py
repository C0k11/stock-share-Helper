import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _pick_first_str(d: Dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _get_rec(days: Dict[str, Any], date: str, ticker: str) -> Dict[str, Any]:
    day = days.get(str(date))
    if not isinstance(day, dict):
        return {}
    items = day.get("items")
    if not isinstance(items, dict):
        return {}
    rec = items.get(str(ticker).upper())
    return rec if isinstance(rec, dict) else {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", required=True)
    p.add_argument("--decisions", required=True)
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    alpha_path = Path(str(args.alpha_csv))
    dec_path = Path(str(args.decisions))

    df = pd.read_csv(alpha_path)
    df["classification"] = df.get("classification", "").astype(str)

    if "planner_allow" not in df.columns:
        raise SystemExit("alpha CSV missing planner_allow")

    missed_allowed = df[(df["classification"].str.contains("Missed", na=False)) & (df["planner_allow"] == True)].copy()
    if "fr_h5" in missed_allowed.columns:
        missed_allowed["fr_h5"] = pd.to_numeric(missed_allowed["fr_h5"], errors="coerce").fillna(0.0)
        missed_allowed = missed_allowed.sort_values("fr_h5", ascending=False)

    missed_allowed = missed_allowed.head(int(args.top))

    payload = json.loads(dec_path.read_text(encoding="utf-8"))
    days = payload.get("days") if isinstance(payload, dict) else None
    if not isinstance(days, dict):
        raise SystemExit("decisions JSON missing days dict")

    rows = []
    for _, r in missed_allowed.iterrows():
        date = str(r.get("date", ""))
        ticker = str(r.get("ticker", "")).upper()
        rec = _get_rec(days, date, ticker)
        router = rec.get("router") if isinstance(rec.get("router"), dict) else {}
        final = rec.get("final") if isinstance(rec.get("final"), dict) else {}

        router_chosen = _pick_first_str(
            router,
            ["chosen_expert", "selected_expert", "routed_expert", "expert", "route", "chosen", "winner"],
        )
        rec_expert = str(rec.get("expert") or "")
        final_action = str(final.get("action") or "")
        final_tp = final.get("target_position")
        final_tp_f = _as_float(final_tp)

        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "fr_h5": _as_float(r.get("fr_h5")),
                "alpha_expert": str(r.get("expert") or ""),
                "alpha_planner_strategy": str(r.get("planner_strategy") or ""),
                "alpha_target_position": _as_float(r.get("target_position")),
                "alpha_turnover": _as_float(r.get("turnover")),
                "router_keys": ",".join(sorted(router.keys())) if isinstance(router, dict) else "",
                "router_chosen": router_chosen,
                "rec_expert": rec_expert,
                "final_action": final_action,
                "final_target_position": final_tp_f,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("(no rows)")
        return

    print(out.to_string(index=False))

    print("---")
    print("router_chosen counts")
    if "router_chosen" in out.columns:
        print(out["router_chosen"].astype(str).value_counts(dropna=False))

    print("---")
    print("rec_expert counts")
    if "rec_expert" in out.columns:
        print(out["rec_expert"].astype(str).value_counts(dropna=False))

    print("---")
    print("final_target_position nonzero count")
    nz = int((pd.to_numeric(out["final_target_position"], errors="coerce").fillna(0.0).abs() > 1e-12).sum())
    print(nz)


if __name__ == "__main__":
    main()
