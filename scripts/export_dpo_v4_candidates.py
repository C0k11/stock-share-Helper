import argparse
import json
from pathlib import Path
from typing import Any, Dict

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
    p.add_argument("--out", required=True)
    p.add_argument("--min-fr", type=float, default=0.0)
    args = p.parse_args()

    alpha_path = Path(str(args.alpha_csv))
    dec_path = Path(str(args.decisions))
    out_path = Path(str(args.out))

    df = pd.read_csv(alpha_path)

    df["classification"] = df.get("classification", "").astype(str)
    if "planner_allow" not in df.columns:
        raise SystemExit("alpha CSV missing planner_allow")

    cand = df[(df["classification"].str.contains("Missed", na=False)) & (df["planner_allow"] == True)].copy()

    if "fr_h5" in cand.columns:
        cand["fr_h5"] = pd.to_numeric(cand["fr_h5"], errors="coerce").fillna(0.0)
        if float(args.min_fr) > 0:
            cand = cand[cand["fr_h5"] >= float(args.min_fr)]

    payload = json.loads(dec_path.read_text(encoding="utf-8"))
    days = payload.get("days") if isinstance(payload, dict) else None
    if not isinstance(days, dict):
        raise SystemExit("decisions JSON missing days dict")

    rows = []
    for _, r in cand.iterrows():
        date = str(r.get("date", ""))
        ticker = str(r.get("ticker", "")).upper()
        rec = _get_rec(days, date, ticker)

        router = rec.get("router") if isinstance(rec.get("router"), dict) else {}
        final = rec.get("final") if isinstance(rec.get("final"), dict) else {}

        router_chosen = ""
        if isinstance(router, dict):
            router_chosen = _pick_first_str(
                router,
                ["chosen_expert", "selected_expert", "routed_expert", "expert", "route", "chosen", "winner"],
            )

        final_action = str(final.get("action") or "")

        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "fr_h5": _as_float(r.get("fr_h5")),
                "news_score": _as_float(r.get("news_score")),
                "volatility_ann_pct": _as_float(r.get("volatility_ann_pct")),
                "planner_allow": bool(r.get("planner_allow") is True),
                "planner_strategy": str(r.get("planner_strategy") or ""),
                "alpha_expert": str(r.get("expert") or ""),
                "alpha_target_position": _as_float(r.get("target_position")),
                "alpha_turnover": _as_float(r.get("turnover")),
                "router_chosen": router_chosen,
                "final_action": final_action,
                "current_rec_expert": str(rec.get("expert") or ""),
                "target_expert": "analyst",
                "target_action": "BUY",
                "reward_weight": _as_float(r.get("fr_h5")),
            }
        )

    out_df = pd.DataFrame(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values(["fr_h5", "date", "ticker"], ascending=[False, True, True]).to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out_df)}")
    if len(out_df):
        print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
