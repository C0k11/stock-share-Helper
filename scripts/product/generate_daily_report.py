import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _find_single(path: Path, pattern: str) -> Optional[Path]:
    cands = sorted(path.glob(pattern))
    if not cands:
        return None
    return cands[-1]


def _load_decisions(decisions_path: Path) -> Dict[str, Any]:
    return json.loads(decisions_path.read_text(encoding="utf-8"))


def _get_day_decisions(decisions_obj: Dict[str, Any], date: str) -> Dict[str, Any]:
    days = decisions_obj.get("days") if isinstance(decisions_obj, dict) else None
    if not isinstance(days, dict):
        return {}
    day = days.get(date)
    return day if isinstance(day, dict) else {}


def _extract_risk_alerts(day_decisions: Dict[str, Any]) -> List[str]:
    items = day_decisions.get("items") if isinstance(day_decisions.get("items"), dict) else {}
    alerts: List[str] = []
    for ticker, it in items.items():
        if not isinstance(it, dict):
            continue
        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        trace = final.get("trace") if isinstance(final.get("trace"), list) else []
        for t in trace:
            s = _safe_str(t)
            if not s:
                continue
            if "FORCE CLEAR" in s or "[RISK]" in s or "Drawdown" in s:
                alerts.append(f"{ticker}: {s}")
    return alerts


def _load_alpha_days(alpha_days_path: Path) -> pd.DataFrame:
    df = pd.read_csv(alpha_days_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _load_daily(daily_path: Path) -> pd.DataFrame:
    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["pnl_h1_net", "turnover", "fee", "target_position", "news_count", "news_score", "volatility_ann_pct", "fr_h1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    for c in ["expert", "planner_strategy", "ticker"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
        else:
            df[c] = ""
    return df


def _load_alpha_pairs(alpha_pairs_path: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(alpha_pairs_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return obj if isinstance(obj, list) else []


def _format_md_table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return ""
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _planner_ai_insight_lines(
    *,
    day_decisions: Dict[str, Any],
    daily: pd.DataFrame,
    date: str,
    model_path: Path,
    planner_policy: str = "",
) -> List[str]:
    planner = day_decisions.get("planner") if isinstance(day_decisions.get("planner"), dict) else {}
    recorded_strategy = _safe_str(planner.get("strategy")) if planner else ""

    inputs = planner.get("inputs") if isinstance(planner.get("inputs"), dict) else {}
    probs = inputs.get("probs") if isinstance(inputs.get("probs"), dict) else {}
    feats = inputs.get("features") if isinstance(inputs.get("features"), dict) else {}

    policy = str(planner_policy or "").strip().lower()
    force_mri = policy == "sft"
    disable_mri = policy == "rule"
    if force_mri:
        probs = {}
        feats = {}

    mri: Optional[Dict[str, Any]] = None
    if (not disable_mri) and (not probs):
        mri = _try_planner_sft_mri(daily=daily, date=str(date), model_path=model_path)
        if isinstance(mri, dict):
            probs = mri.get("probs") if isinstance(mri.get("probs"), dict) else probs
            feats = mri.get("features") if isinstance(mri.get("features"), dict) else feats

    out: List[str] = []
    strategy = ""
    if isinstance(mri, dict):
        strategy = _safe_str(mri.get("strategy"))
    if not strategy:
        strategy = recorded_strategy

    if strategy:
        out.append(f"- Current Strategy: `{strategy.upper()}`")
    else:
        out.append("- Current Strategy: `(unknown)`")

    if recorded_strategy and (recorded_strategy != strategy):
        out.append(f"- Recorded Strategy (run): `{recorded_strategy.upper()}`")

    if probs:
        out.append("- Probabilities Source: `recorded`" if (inputs.get("probs") and not force_mri) else "- Probabilities Source: `mri`")
    else:
        if disable_mri:
            out.append("- MRI Status: `disabled (--planner-policy rule)`")
        elif not model_path.exists():
            out.append(f"- MRI Status: `skipped (missing model: {model_path.as_posix()})`")
        else:
            out.append("- MRI Status: `failed (model present but could not run)`")

    if probs:
        ps: List[Tuple[str, float]] = []
        for k, v in probs.items():
            ps.append((_safe_str(k), _safe_float(v, 0.0)))
        ps = sorted(ps, key=lambda x: x[1], reverse=True)

        conf = 0.0
        if strategy and (strategy in probs):
            conf = _safe_float(probs.get(strategy), 0.0)
        elif ps:
            conf = float(ps[0][1])
        out.append(f"- Confidence: `{conf * 100.0:.1f}%`")

        top2 = ps[:2]
        if top2:
            out.append("- Top Probabilities:")
            for k, v in top2:
                out.append(f"  - `{k}`: `{v * 100.0:.1f}%`")
    else:
        out.append("- Confidence: `(n/a)`")

    if feats:
        keys = [
            "n_tickers",
            "vol_mean",
            "vol_max",
            "news_count_sum",
            "news_count_mean",
            "news_score_mean",
            "news_score_max",
            "has_strong_news_day",
            "prev_gross_exposure",
            "prev_net_exposure",
            "prev_abs_exposure_mean",
            "prev_long_count",
            "prev_short_count",
        ]
        out.append("- Input Context:")
        for k in keys:
            if k in feats:
                out.append(f"  - `{k}`: `{_safe_float(feats.get(k), 0.0):.6f}`")

    return out


def _compute_prev_exposure_from_daily(daily: pd.DataFrame, date: str) -> Dict[str, float]:
    if daily.empty:
        return {}
    try:
        dates = sorted([d for d in daily["date"].astype(str).unique().tolist() if str(d).strip()])
    except Exception:
        return {}
    prev_date = ""
    for d in dates:
        if str(d) < str(date):
            prev_date = str(d)
    if not prev_date:
        return {}

    prev_rows = daily[daily["date"].astype(str) == str(prev_date)].copy()
    if prev_rows.empty:
        return {}

    if "target_position" not in prev_rows.columns:
        prev_rows["target_position"] = 0.0
    tp = pd.to_numeric(prev_rows["target_position"], errors="coerce").fillna(0.0)

    gross = float(tp.abs().sum())
    net = float(tp.sum())
    abs_mean = float(tp.abs().mean()) if len(tp) else 0.0
    long_count = float((tp > 0).sum())
    short_count = float((tp < 0).sum())

    return {
        "prev_gross_exposure": gross,
        "prev_net_exposure": net,
        "prev_abs_exposure_mean": abs_mean,
        "prev_long_count": long_count,
        "prev_short_count": short_count,
    }


def _try_planner_sft_mri(
    *,
    daily: pd.DataFrame,
    date: str,
    model_path: Path,
) -> Optional[Dict[str, Any]]:
    if not model_path.exists():
        return None
    if daily.empty:
        return None

    day_rows = daily[daily["date"].astype(str) == str(date)].copy()
    if day_rows.empty:
        return None

    try:
        import numpy as np
        import torch
    except Exception:
        return None

    try:
        payload = torch.load(str(model_path), map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    feature_names = payload.get("feature_names")
    mean = payload.get("scaler_mean")
    std = payload.get("scaler_std")
    idx_to_label = payload.get("idx_to_label")
    state = payload.get("model_state")
    if not (isinstance(feature_names, list) and isinstance(mean, list) and isinstance(std, list) and isinstance(idx_to_label, dict) and isinstance(state, dict)):
        return None

    class _MLP(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(int(input_dim), 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, int(output_dim)),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    try:
        idx_map = {int(k): _safe_str(v) for k, v in idx_to_label.items()}
        model = _MLP(len(feature_names), len(idx_map), float(payload.get("dropout", 0.2)))
        model.load_state_dict(state)
        model.eval()
    except Exception:
        return None

    if "volatility_ann_pct" not in day_rows.columns:
        day_rows["volatility_ann_pct"] = 0.0
    if "news_count" not in day_rows.columns:
        day_rows["news_count"] = 0.0
    if "news_score" not in day_rows.columns:
        day_rows["news_score"] = 0.0

    has_strong = 0.0
    if "has_strong_news_day" in day_rows.columns:
        try:
            has_strong = 1.0 if bool(pd.to_numeric(day_rows["has_strong_news_day"], errors="coerce").fillna(0.0).astype(bool).any()) else 0.0
        except Exception:
            has_strong = 0.0

    vol = pd.to_numeric(day_rows["volatility_ann_pct"], errors="coerce").fillna(0.0)
    nc = pd.to_numeric(day_rows["news_count"], errors="coerce").fillna(0.0)
    ns = pd.to_numeric(day_rows["news_score"], errors="coerce").fillna(0.0)

    if has_strong <= 0.0:
        try:
            has_strong = 1.0 if (float(ns.max()) > 0.0 if len(ns) else False) else 0.0
        except Exception:
            has_strong = 0.0

    feats: Dict[str, float] = {
        "n_tickers": float(len(day_rows)),
        "vol_mean": float(vol.mean()) if len(vol) else 0.0,
        "vol_max": float(vol.max()) if len(vol) else 0.0,
        "news_count_sum": float(nc.sum()) if len(nc) else 0.0,
        "news_count_mean": float(nc.mean()) if len(nc) else 0.0,
        "news_score_mean": float(ns.mean()) if len(ns) else 0.0,
        "news_score_max": float(ns.max()) if len(ns) else 0.0,
        "has_strong_news_day": float(has_strong),
    }
    feats.update(_compute_prev_exposure_from_daily(daily, date))

    x_list: List[float] = []
    for i, name in enumerate([_safe_str(x) for x in feature_names]):
        v = float(feats.get(name, 0.0) or 0.0)
        mu = float(mean[i]) if i < len(mean) else 0.0
        sd = float(std[i]) if i < len(std) else 1.0
        if sd <= 1e-12:
            sd = 1.0
        x_list.append((v - mu) / sd)

    x_np = np.asarray([x_list], dtype=np.float32)
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs_t).item())
        probs = {idx_map.get(i, str(i)): float(probs_t[i].item()) for i in range(int(probs_t.shape[0]))}

    return {
        "strategy": idx_map.get(idx, "cash_preservation"),
        "probs": probs,
        "features": feats,
    }


def _append_paper_log(log_path: Path, row: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    write_header = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--system", default="golden_strict", choices=["golden_strict", "baseline_fast"])
    parser.add_argument("--out", default=None, help="Output markdown path (default: reports/daily/YYYY-MM-DD.md)")
    parser.add_argument("--paper-log", default="paper_trading/backtest_history.csv")
    parser.add_argument("--write-paper-log", action="store_true", default=False)
    parser.add_argument("--planner-sft-model", default="models/planner_sft_v1.pt")
    parser.add_argument("--planner-policy", default="", choices=["", "rule", "sft"])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    date = str(args.date).strip()

    sys_dir = run_dir / str(args.system)
    daily_path = sys_dir / "daily.csv"
    decisions_path = _find_single(sys_dir, "decisions_*.json")
    alpha_days_path = sys_dir / "alpha_days.csv"
    alpha_pairs_path = run_dir / "alpha_pairs.json"

    if not daily_path.exists():
        raise SystemExit(f"Missing daily.csv: {daily_path}")
    if not decisions_path or (not decisions_path.exists()):
        raise SystemExit(f"Missing decisions_*.json under: {sys_dir}")

    daily = _load_daily(daily_path)
    day_rows = daily[daily["date"] == date].copy()

    decisions_obj = _load_decisions(decisions_path)
    day_decisions = _get_day_decisions(decisions_obj, date)

    risk_alerts = _extract_risk_alerts(day_decisions)

    alpha_day_row: Optional[pd.Series] = None
    if alpha_days_path.exists():
        alpha_days = _load_alpha_days(alpha_days_path)
        hit = alpha_days[alpha_days["date"] == date]
        if len(hit):
            alpha_day_row = hit.iloc[0]

    alpha_pairs = _load_alpha_pairs(alpha_pairs_path) if alpha_pairs_path.exists() else []
    alpha_watch = [
        r
        for r in alpha_pairs
        if _safe_str(r.get("date")) == date
        and _safe_str(r.get("type")) in ["POSITIVE_SAMPLE", "NEGATIVE_SAMPLE"]
    ]

    alpha_watch_rows: List[List[str]] = []
    for r in sorted(alpha_watch, key=lambda x: -abs(_safe_float(x.get("diff_pnl"), 0.0))):
        pair = r.get("pair") if isinstance(r.get("pair"), dict) else {}
        w = pair.get("winner") if isinstance(pair.get("winner"), dict) else {}
        l = pair.get("loser") if isinstance(pair.get("loser"), dict) else {}
        alpha_watch_rows.append(
            [
                _safe_str(r.get("ticker")),
                f"{_safe_float(r.get('diff_pnl'), 0.0):+.6f}",
                f"{_safe_str(w.get('system'))}:{_safe_str(w.get('action'))}({w.get('target_position', 0.0)})",
                f"{_safe_str(l.get('system'))}:{_safe_str(l.get('action'))}({l.get('target_position', 0.0)})",
            ]
        )

    action_plan_rows: List[List[str]] = []
    items = day_decisions.get("items") if isinstance(day_decisions.get("items"), dict) else {}
    for ticker, it in items.items():
        if not isinstance(it, dict):
            continue
        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        router = it.get("router") if isinstance(it.get("router"), dict) else {}
        action_plan_rows.append(
            [
                str(ticker),
                _safe_str(it.get("expert") or router.get("expert")),
                _safe_str(final.get("action")),
                f"{_safe_float(final.get('target_position'), 0.0):+.3f}",
                _safe_str(router.get("planner_strategy")),
            ]
        )

    action_plan_rows = sorted(action_plan_rows, key=lambda r: abs(_safe_float(r[3], 0.0)), reverse=True)

    pnl_sum = float(day_rows["pnl_h1_net"].sum()) if len(day_rows) else 0.0
    turnover_sum = float(day_rows["turnover"].sum()) if len(day_rows) else 0.0
    trade_count = int((day_rows["turnover"] > 1e-12).sum()) if len(day_rows) else 0
    fee_sum = float(day_rows["fee"].sum()) if len(day_rows) else 0.0

    fr_h1 = float(day_rows["fr_h1"].median()) if len(day_rows) else 0.0
    avg_vol = float(day_rows["volatility_ann_pct"].mean()) if len(day_rows) else 0.0

    day_type = _safe_str(alpha_day_row.get("day_type")) if alpha_day_row is not None else ""
    total_news = _safe_float(alpha_day_row.get("total_news_vol"), 0.0) if alpha_day_row is not None else 0.0
    max_news_impact = _safe_float(alpha_day_row.get("max_news_impact"), 0.0) if alpha_day_row is not None else 0.0

    out_path = Path(args.out) if args.out else (Path("reports") / "daily" / f"{date}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Daily Report ({date})")
    lines.append("")
    lines.append(f"**Source run**: `{run_dir.as_posix()}`")
    lines.append(f"**System**: `{args.system}`")
    lines.append("")

    lines.append("## Risk Alert")
    if risk_alerts:
        for a in risk_alerts[:20]:
            lines.append(f"- {a}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Market Pulse")
    lines.append(f"- day_type: `{day_type}`")
    lines.append(f"- market_proxy_fr_h1(median): `{fr_h1:+.4f}`")
    lines.append(f"- avg_volatility_ann_pct(mean): `{avg_vol:.2f}`")
    lines.append(f"- total_news_vol(day): `{total_news:.0f}`")
    lines.append(f"- max_news_impact(day): `{max_news_impact:.2f}`")
    lines.append("")

    lines.append("## Planner AI Insight")
    lines.extend(
        _planner_ai_insight_lines(
            day_decisions=day_decisions,
            daily=daily,
            date=str(date),
            model_path=Path(str(args.planner_sft_model)),
            planner_policy=str(args.planner_policy),
        )
    )
    lines.append("")

    lines.append("## Alpha Watch")
    if alpha_watch_rows:
        lines.append(_format_md_table(alpha_watch_rows, ["ticker", "diff_pnl", "winner", "loser"]))
    else:
        lines.append("(no significant action-pair diffs for this date)")
    lines.append("")

    lines.append("## Action Plan")
    if action_plan_rows:
        lines.append(_format_md_table(action_plan_rows[:40], ["ticker", "expert", "action", "target_position", "planner_strategy"]))
    else:
        lines.append("(no decisions found)")
    lines.append("")

    lines.append("## PnL Summary")
    lines.append(f"- pnl_h1_net_sum: `{pnl_sum:+.6f}`")
    lines.append(f"- trade_count: `{trade_count}`")
    lines.append(f"- turnover_sum: `{turnover_sum:.4f}`")
    lines.append(f"- fee_sum: `{fee_sum:.6f}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {out_path}")

    if bool(args.write_paper_log):
        paper_row = {
            "date": date,
            "run_dir": str(run_dir).replace("\\", "/"),
            "system": str(args.system),
            "day_type": day_type,
            "pnl_h1_net_sum": pnl_sum,
            "trade_count": trade_count,
            "turnover_sum": turnover_sum,
            "fee_sum": fee_sum,
        }
        _append_paper_log(Path(args.paper_log), paper_row)
        print(f"Appended paper log: {args.paper_log}")


if __name__ == "__main__":
    main()
