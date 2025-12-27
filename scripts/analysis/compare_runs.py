import argparse
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _mode_str(values: List[str]) -> str:
    vals = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return ""
    s = pd.Series(vals)
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return ""
    return str(vc.index[0])


def _load_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit(f"daily.csv missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for c in ["pnl_h1_net", "pnl_h1", "turnover", "fee", "target_position", "news_count", "news_score", "volatility_ann_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "planner_strategy" in df.columns:
        df["planner_strategy"] = df["planner_strategy"].astype(str)
    else:
        df["planner_strategy"] = ""

    return df


def _daily_to_per_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "planner_strategy", "pnl_h1_net_sum"])

    rows: List[Dict[str, Any]] = []
    for d, g in df.groupby("date"):
        strat = _mode_str([_safe_str(x) for x in g.get("planner_strategy", "").astype(str).tolist()])
        pnl = float(g.get("pnl_h1_net", 0.0).sum()) if "pnl_h1_net" in g.columns else 0.0
        rows.append({"date": str(d), "planner_strategy": strat, "pnl_h1_net_sum": pnl})

    out = pd.DataFrame(rows)
    out = out.sort_values("date")
    return out


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare teacher vs student runs using daily.csv")
    p.add_argument("--teacher", required=True, help="Path to teacher daily.csv")
    p.add_argument("--student", required=True, help="Path to student daily.csv")
    p.add_argument("--checkpoints", nargs="*", default=["2022-06-06", "2022-06-10"]) 
    args = p.parse_args()

    t_raw = _load_daily(str(args.teacher))
    s_raw = _load_daily(str(args.student))

    t = _daily_to_per_day(t_raw)
    s = _daily_to_per_day(s_raw)

    merged = t.merge(s, on="date", how="inner", suffixes=("_teacher", "_student"))

    total_teacher = float(t.get("pnl_h1_net_sum", 0.0).sum()) if len(t) else 0.0
    total_student = float(s.get("pnl_h1_net_sum", 0.0).sum()) if len(s) else 0.0

    match = 0
    denom = 0
    for _i, r in merged.iterrows():
        a = _safe_str(r.get("planner_strategy_teacher"))
        b = _safe_str(r.get("planner_strategy_student"))
        if a and b:
            denom += 1
            if a == b:
                match += 1

    match_rate = float(match) / float(denom) if denom else 0.0

    print("# Showdown Summary")
    print("")
    print("## Overall Metrics")
    print("")
    print("| metric | teacher | student |")
    print("| --- | ---: | ---: |")
    print(f"| total_pnl_h1_net_sum | {total_teacher:+.6f} | {total_student:+.6f} |")
    print(f"| strategy_match_rate (non-empty) | {_fmt_pct(match_rate)} | {_fmt_pct(match_rate)} |")
    print(f"| overlap_days | {len(merged)} | {len(merged)} |")
    print("")

    print("## Checkpoints")
    print("")
    print("| date | teacher_strategy | student_strategy | teacher_pnl_h1_net_sum | student_pnl_h1_net_sum | match |")
    print("| --- | --- | --- | ---: | ---: | --- |")
    for d in [str(x) for x in (args.checkpoints or [])]:
        tr = t[t["date"] == d]
        sr = s[s["date"] == d]
        t_strat = _safe_str(tr["planner_strategy"].iloc[0]) if len(tr) else ""
        s_strat = _safe_str(sr["planner_strategy"].iloc[0]) if len(sr) else ""
        t_pnl = float(tr["pnl_h1_net_sum"].iloc[0]) if len(tr) else 0.0
        s_pnl = float(sr["pnl_h1_net_sum"].iloc[0]) if len(sr) else 0.0
        m = "YES" if (t_strat and s_strat and t_strat == s_strat) else "NO"
        print(f"| {d} | {t_strat} | {s_strat} | {t_pnl:+.6f} | {s_pnl:+.6f} | {m} |")


if __name__ == "__main__":
    main()
