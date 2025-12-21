#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if isinstance(cur, dict):
            cur = cur.get(k)
            continue
        if isinstance(cur, list) and isinstance(k, int) and (0 <= k < len(cur)):
            cur = cur[k]
            continue
        return default
    return cur if cur is not None else default


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return int(_to_float(x))


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in [
        "target_position",
        "turnover",
        "fee",
        "news_score",
        "volatility_ann_pct",
        "pnl_h1",
        "pnl_h5",
        "pnl_h10",
        "pnl_h1_net",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["news_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["planner_allow"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    for c in ["expert", "ticker", "date"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _bucket_stats(df: pd.DataFrame, *, pred: Callable[[pd.DataFrame], pd.Series], horizon: int, high_news_score_threshold: float, high_news_count_threshold: int) -> Dict[str, Any]:
    if df.empty:
        return {
            "items": 0,
            "date_count": 0,
            "trade_count": 0,
            "fees_total": 0.0,
            "pnl_sum": 0.0,
            "pnl_sum_net": 0.0,
            "analyst_coverage": 0.0,
        }

    mask = pred(df)
    sub = df.loc[mask].copy()
    if sub.empty:
        return {
            "items": 0,
            "date_count": 0,
            "trade_count": 0,
            "fees_total": 0.0,
            "pnl_sum": 0.0,
            "pnl_sum_net": 0.0,
            "analyst_coverage": 0.0,
        }

    pnl_col = f"pnl_h{int(horizon)}"
    pnl = float(sub[pnl_col].fillna(0.0).sum()) if pnl_col in sub.columns else 0.0
    fees = float(sub["fee"].fillna(0.0).sum()) if "fee" in sub.columns else 0.0

    pnl_net = pnl
    if int(horizon) == 1:
        pnl_net = pnl - fees

    # trade_count counts rebalance events
    trade_count = int((sub["turnover"].fillna(0.0) > 1e-12).sum()) if "turnover" in sub.columns else 0

    analyst_cov = 0.0
    if "expert" in sub.columns:
        analyst_cov = float((sub["expert"].astype(str) == "analyst").sum()) / float(len(sub))

    return {
        "items": int(len(sub)),
        "date_count": int(sub["date"].astype(str).nunique()) if "date" in sub.columns else 0,
        "trade_count": int(trade_count),
        "fees_total": float(fees),
        "pnl_sum": float(pnl),
        "pnl_sum_net": float(pnl_net),
        "analyst_coverage": float(analyst_cov),
        "high_news_score_threshold": float(high_news_score_threshold),
        "high_news_count_threshold": int(high_news_count_threshold),
    }


def _global_from_daily(df: pd.DataFrame, horizons: List[int]) -> Dict[str, Any]:
    if df.empty or ("date" not in df.columns):
        return {
            "pnl_sum": {str(h): 0.0 for h in horizons},
            "pnl_sum_net": {str(h): 0.0 for h in horizons},
            "trade_count": 0,
            "fees_total": 0.0,
            "nav_end": 0.0,
            "max_drawdown": 0.0,
            "analyst_coverage": 0.0,
            "planner_allow_rate": 0.0,
        }

    df2 = df.copy()
    for c in ["turnover", "fee", "pnl_h1", "pnl_h5", "pnl_h10"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)
    if "expert" in df2.columns:
        df2["expert"] = df2["expert"].astype(str)
    if "planner_allow" in df2.columns:
        df2["planner_allow"] = df2["planner_allow"].astype(str).str.lower().isin(["true", "1", "yes"])

    pnl_sum = {str(h): float(df2.get(f"pnl_h{int(h)}", 0.0).sum()) for h in horizons}
    fees_total = float(df2.get("fee", 0.0).sum())
    pnl_sum_net = dict(pnl_sum)
    if "1" in pnl_sum_net:
        pnl_sum_net["1"] = float(pnl_sum_net["1"]) - float(fees_total)

    trade_count = int((df2.get("turnover", 0.0) > 1e-12).sum())
    analyst_cov = float((df2.get("expert", "").astype(str) == "analyst").sum()) / float(len(df2)) if len(df2) else 0.0

    # planner_allow_rate: fraction of dates where planner_allow is True (max over tickers that day)
    planner_allow_rate = 0.0
    if "planner_allow" in df2.columns:
        by_day = df2.groupby("date")["planner_allow"].max()
        if len(by_day):
            planner_allow_rate = float(by_day.mean())

    # NAV/max_drawdown computed from h=1 net daily pnl
    day_pnl1 = df2.groupby("date")["pnl_h1"].sum() if "pnl_h1" in df2.columns else pd.Series(dtype=float)
    day_fee = df2.groupby("date")["fee"].sum() if "fee" in df2.columns else pd.Series(dtype=float)
    day_ret = (day_pnl1.reindex(day_pnl1.index).fillna(0.0) - day_fee.reindex(day_pnl1.index).fillna(0.0)).sort_index()

    nav = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in day_ret.tolist():
        nav = nav * (1.0 + float(r))
        peak = max(peak, nav)
        dd = (nav / peak) - 1.0
        max_dd = min(max_dd, float(dd))

    return {
        "pnl_sum": pnl_sum,
        "pnl_sum_net": pnl_sum_net,
        "trade_count": int(trade_count),
        "fees_total": float(fees_total),
        "nav_end": float(nav),
        "max_drawdown": float(max_dd),
        "analyst_coverage": float(analyst_cov),
        "planner_allow_rate": float(planner_allow_rate),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="Path to results/{run_id}/metrics.json")
    p.add_argument("--out", default="", help="Optional path to write report.md")
    args = p.parse_args()

    metrics_path = Path(str(args.metrics))
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir = metrics_path.parent
    horizons = _get(payload, ["protocol", "horizons"], default=[1, 5, 10])
    horizons = [int(x) for x in horizons]
    high_vol_thr = float(_get(payload, ["protocol", "high_vol_ann_pct_threshold"], default=25.0))
    high_news_score_thr = float(_get(payload, ["protocol", "high_news_score_threshold"], default=0.8))
    high_news_count_thr = int(_get(payload, ["protocol", "high_news_count_threshold"], default=1))

    base_daily_path = run_dir / "baseline_fast" / "daily.csv"
    gold_daily_path = run_dir / "golden_strict" / "daily.csv"
    df_base = _load_csv(base_daily_path) if base_daily_path.exists() else pd.DataFrame()
    df_gold = _load_csv(gold_daily_path) if gold_daily_path.exists() else pd.DataFrame()

    lines: List[str] = []
    lines.append(f"# Report: {run_dir.name}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    proto = _get(payload, ["protocol"], default={})
    if isinstance(proto, dict):
        for k in [
            "horizons",
            "trade_cost_bps",
            "min_news_abs_impact",
            "high_news_score_threshold",
            "high_news_count_threshold",
            "high_vol_ann_pct_threshold",
        ]:
            if k in proto:
                lines.append(f"- **{k}**: `{proto[k]}`")
    lines.append("")

    lines.append("## Global Scorecard")
    lines.append("")
    lines.append("| System | pnl_sum(h=1) | pnl_sum(h=5) | pnl_sum(h=10) | pnl_sum_net(h=1) | trade_count | fees_total | nav_end | max_drawdown | analyst_coverage | planner_allow_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    bg_raw = _global_from_daily(df_base, horizons)
    gg_raw = _global_from_daily(df_gold, horizons)
    bg = {
        "p1": _to_float(bg_raw["pnl_sum"].get("1")),
        "p5": _to_float(bg_raw["pnl_sum"].get("5")),
        "p10": _to_float(bg_raw["pnl_sum"].get("10")),
        "p1n": _to_float(bg_raw["pnl_sum_net"].get("1")),
        "tc": int(bg_raw["trade_count"]),
        "fees": float(bg_raw["fees_total"]),
        "nav": float(bg_raw["nav_end"]),
        "mdd": float(bg_raw["max_drawdown"]),
        "ac": float(bg_raw["analyst_coverage"]),
        "ar": float(bg_raw["planner_allow_rate"]),
    }
    gg = {
        "p1": _to_float(gg_raw["pnl_sum"].get("1")),
        "p5": _to_float(gg_raw["pnl_sum"].get("5")),
        "p10": _to_float(gg_raw["pnl_sum"].get("10")),
        "p1n": _to_float(gg_raw["pnl_sum_net"].get("1")),
        "tc": int(gg_raw["trade_count"]),
        "fees": float(gg_raw["fees_total"]),
        "nav": float(gg_raw["nav_end"]),
        "mdd": float(gg_raw["max_drawdown"]),
        "ac": float(gg_raw["analyst_coverage"]),
        "ar": float(gg_raw["planner_allow_rate"]),
    }
    lines.append(
        "| Baseline Fast | {p1:.6f} | {p5:.6f} | {p10:.6f} | {p1n:.6f} | {tc} | {fees:.6f} | {nav:.6f} | {mdd:.6f} | {ac:.4f} | {ar:.4f} |".format(
            **bg
        )
    )
    lines.append(
        "| Golden Strict | {p1:.6f} | {p5:.6f} | {p10:.6f} | {p1n:.6f} | {tc} | {fees:.6f} | {nav:.6f} | {mdd:.6f} | {ac:.4f} | {ar:.4f} |".format(
            **gg
        )
    )
    lines.append("")

    # Build DATE-LEVEL buckets (shared date sets across systems for fair comparison)
    all_dates = sorted(set(df_base.get("date", pd.Series(dtype=str)).astype(str).tolist()) | set(df_gold.get("date", pd.Series(dtype=str)).astype(str).tolist()))
    all_date_set = set([d for d in all_dates if str(d).strip()])

    high_news_dates: set = set()
    for _df in [df_base, df_gold]:
        if "date" not in _df.columns:
            continue
        # 1) signal-based strong-news days
        if "has_strong_news_day" in _df.columns:
            dn = set(_df.loc[_df["has_strong_news_day"].astype(bool) == True, "date"].astype(str).tolist())
            high_news_dates |= dn
        # 2) router meta thresholds (align with MoE routing intent)
        if ("news_score" in _df.columns) or ("news_count" in _df.columns):
            g = _df.groupby("date").agg(
                {
                    "news_score": "max" if "news_score" in _df.columns else "size",
                    "news_count": "max" if "news_count" in _df.columns else "size",
                }
            )
            if "news_score" in g.columns:
                g["news_score"] = pd.to_numeric(g["news_score"], errors="coerce").fillna(0.0)
            if "news_count" in g.columns:
                g["news_count"] = pd.to_numeric(g["news_count"], errors="coerce").fillna(0.0)
            dn2 = set(
                g.index[
                    (g.get("news_score", 0.0) >= float(high_news_score_thr))
                    | (g.get("news_count", 0.0) >= float(high_news_count_thr))
                ]
                .astype(str)
                .tolist()
            )
            high_news_dates |= dn2

    high_vol_dates: set = set()
    for _df in [df_base, df_gold]:
        if "volatility_ann_pct" in _df.columns and "date" in _df.columns:
            dv = _df.groupby("date")["volatility_ann_pct"].quantile(0.75)
            dv = pd.to_numeric(dv, errors="coerce").fillna(0.0)
            high_vol_dates |= set(dv.index[dv >= float(high_vol_thr)].astype(str).tolist())

    planner_allow_dates: set = set()
    if "planner_allow" in df_gold.columns and "date" in df_gold.columns:
        planner_allow_dates = set(df_gold.loc[df_gold["planner_allow"].astype(bool) == True, "date"].astype(str).tolist())
    planner_disallow_dates: set = set([d for d in all_date_set if d not in planner_allow_dates])

    def _date_bucket(date_set: set) -> Callable[[pd.DataFrame], pd.Series]:
        def _pred(df: pd.DataFrame) -> pd.Series:
            if df.empty or ("date" not in df.columns):
                return pd.Series([], dtype=bool)
            return df["date"].astype(str).isin(date_set)

        return _pred

    buckets: List[Tuple[str, str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("all", "All days", _date_bucket(all_date_set)),
        ("high_news", "High-news days", _date_bucket(high_news_dates)),
        ("high_vol", "High-vol days", _date_bucket(high_vol_dates)),
        ("planner_allow", "Planner-allow days", _date_bucket(planner_allow_dates)),
        ("planner_disallow", "Planner-disallow days", _date_bucket(planner_disallow_dates)),
        (
            "analyst_only",
            "Analyst-only trades",
            lambda df: (df.get("expert", "").astype(str) == "analyst")
            & (df.get("target_position", 0.0).fillna(0.0).abs() > 1e-9),
        ),
    ]

    def emit_bucket(h: int, key: str, title: str, pred: Callable[[pd.DataFrame], pd.Series]) -> None:
        b = _bucket_stats(df_base, pred=pred, horizon=int(h), high_news_score_threshold=high_news_score_thr, high_news_count_threshold=high_news_count_thr)
        g = _bucket_stats(df_gold, pred=pred, horizon=int(h), high_news_score_threshold=high_news_score_thr, high_news_count_threshold=high_news_count_thr)

        delta = {
            "pnl": float(g["pnl_sum"]) - float(b["pnl_sum"]),
            "pnl_net": float(g["pnl_sum_net"]) - float(b["pnl_sum_net"]),
            "trades": int(g["trade_count"]) - int(b["trade_count"]),
            "fees": float(g["fees_total"]) - float(b["fees_total"]),
        }

        lines.append(f"### Bucket: {title}")
        lines.append("")
        lines.append("| System | date_count | pnl_sum | pnl_sum_net(h=1) | trade_count | fees_total | analyst_coverage |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        lines.append(
            "| Baseline Fast | {dc} | {pnl:.6f} | {pnl_net:.6f} | {trades} | {fees:.6f} | {ac:.4f} |".format(
                dc=int(b.get("date_count") or 0),
                pnl=float(b["pnl_sum"]),
                pnl_net=float(b["pnl_sum_net"]),
                trades=int(b["trade_count"]),
                fees=float(b["fees_total"]),
                ac=float(b["analyst_coverage"]),
            )
        )
        lines.append(
            "| Golden Strict | {dc} | {pnl:.6f} | {pnl_net:.6f} | {trades} | {fees:.6f} | {ac:.4f} |".format(
                dc=int(g.get("date_count") or 0),
                pnl=float(g["pnl_sum"]),
                pnl_net=float(g["pnl_sum_net"]),
                trades=int(g["trade_count"]),
                fees=float(g["fees_total"]),
                ac=float(g["analyst_coverage"]),
            )
        )
        lines.append(
            "| Delta (G-B) |  | {pnl:.6f} | {pnl_net:.6f} | {trades} | {fees:.6f} |  |".format(
                pnl=float(delta["pnl"]),
                pnl_net=float(delta["pnl_net"]),
                trades=int(delta["trades"]),
                fees=float(delta["fees"]),
            )
        )
        lines.append("")

    for h in horizons:
        lines.append(f"## Horizon h={h}")
        lines.append("")
        for key, title, pred in buckets:
            emit_bucket(int(h), key, title, pred)

    out_text = "\n".join(lines) + "\n"
    if str(args.out).strip():
        out_path = Path(str(args.out))
    else:
        out_path = metrics_path.parent / "report.md"

    out_path.write_text(out_text, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
