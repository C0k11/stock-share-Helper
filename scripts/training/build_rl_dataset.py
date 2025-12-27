import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_latest_daily_csv(system_dir: Path) -> Optional[Path]:
    p = system_dir / "daily.csv"
    if p.exists():
        return p
    cands = sorted(system_dir.glob("daily_*.csv"))
    if not cands:
        return None
    return cands[-1]


def _load_latest_decisions_json(system_dir: Path) -> Optional[Path]:
    cands = sorted(system_dir.glob("decisions_*.json"))
    if not cands:
        return None
    return cands[-1]


def _load_decisions_by_date(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid decisions json: {path}")

    if isinstance(obj.get("days"), dict):
        out: Dict[str, Dict[str, Any]] = {}
        for d, day in obj["days"].items():
            if isinstance(day, dict):
                out[str(d)] = day
        return out

    if isinstance(obj.get("items"), dict):
        d = str(obj.get("date") or "")
        if not d:
            raise ValueError(f"Single-day decisions missing date: {path}")
        return {d: obj}

    raise ValueError(f"Unrecognized decisions json format: {path}")


def _extract_gate_context(day_obj: Dict[str, Any]) -> Dict[str, float]:
    planner_obj = day_obj.get("planner") if isinstance(day_obj.get("planner"), dict) else {}
    inputs = planner_obj.get("inputs") if isinstance(planner_obj.get("inputs"), dict) else {}

    mr = inputs.get("market_regime") if isinstance(inputs.get("market_regime"), dict) else {}
    mr_regime = str(mr.get("regime") or "").strip().lower()
    mr_score = _safe_float(mr.get("score"), default=0.0)

    probs = inputs.get("probs") if isinstance(inputs.get("probs"), dict) else {}
    p_aggr = _safe_float(probs.get("aggressive_long"), default=0.0)
    p_def = _safe_float(probs.get("defensive"), default=0.0)
    p_cash = _safe_float(probs.get("cash_preservation"), default=0.0)
    conf = float(max(p_aggr, p_def, p_cash))

    strat = str(planner_obj.get("strategy") or "").strip().lower()

    return {
        "market_regime_score": float(mr_score),
        "market_regime_is_risk_off": 1.0 if mr_regime == "risk_off" else 0.0,
        "market_regime_is_risk_on": 1.0 if mr_regime == "risk_on" else 0.0,
        "sft_is_aggressive_long": 1.0 if strat == "aggressive_long" else 0.0,
        "sft_is_defensive": 1.0 if strat == "defensive" else 0.0,
        "sft_is_cash_preservation": 1.0 if strat == "cash_preservation" else 0.0,
        "sft_confidence": float(conf),
    }


def _extract_planner_inputs(day_obj: Dict[str, Any]) -> Dict[str, Any]:
    planner_obj = day_obj.get("planner") if isinstance(day_obj.get("planner"), dict) else {}
    inputs = planner_obj.get("inputs") if isinstance(planner_obj.get("inputs"), dict) else {}
    return {"planner": planner_obj, "inputs": inputs}


def _ensure_sft_context(
    *,
    day_obj: Dict[str, Any],
    sft_planner: Any,
) -> Dict[str, float]:
    base = _extract_gate_context(day_obj)
    if float(base.get("sft_confidence", 0.0)) > 0:
        return base

    if sft_planner is None:
        return base

    pack = _extract_planner_inputs(day_obj)
    inputs = pack.get("inputs") if isinstance(pack.get("inputs"), dict) else {}
    mr = inputs.get("market_regime") if isinstance(inputs.get("market_regime"), dict) else {}
    feats = inputs.get("features") if isinstance(inputs.get("features"), dict) else {}
    if not (isinstance(mr, dict) and isinstance(feats, dict) and feats):
        return base

    try:
        dec = sft_planner.decide(market_regime=mr, features={str(k): _safe_float(v) for k, v in feats.items()}).to_dict()
    except Exception:
        return base

    tmp_day = dict(day_obj)
    tmp_day["planner"] = dec
    return _extract_gate_context(tmp_day)


def build_dataset(*, run_dir: Path, system: str, reward_h: int, risk_penalty_coef: float, sft_planner: Any) -> pd.DataFrame:
    system_dir = run_dir / system
    daily_path = _load_latest_daily_csv(system_dir)
    if daily_path is None:
        raise FileNotFoundError(f"No daily csv found under: {system_dir}")

    df = pd.read_csv(daily_path)
    if "date" not in df.columns:
        raise ValueError(f"daily.csv missing date column: {daily_path}")

    decisions_path = _load_latest_decisions_json(system_dir)
    decisions_by_date: Dict[str, Dict[str, Any]] = {}
    if decisions_path is not None:
        decisions_by_date = _load_decisions_by_date(decisions_path)

    pnl_col = f"pnl_h{int(reward_h)}_net"

    rows: List[Dict[str, Any]] = []
    for date_str, g in df.groupby("date"):
        g2 = g.copy()

        news_count = pd.to_numeric(g2.get("news_count", 0.0), errors="coerce").fillna(0.0)
        news_score = pd.to_numeric(g2.get("news_score", 0.0), errors="coerce").fillna(0.0)
        vol = pd.to_numeric(g2.get("volatility_ann_pct", 0.0), errors="coerce").fillna(0.0)

        target_pos = pd.to_numeric(g2.get("target_position", 0.0), errors="coerce").fillna(0.0)

        pnl_series = pd.to_numeric(g2.get(pnl_col, 0.0), errors="coerce").fillna(0.0)
        pnl_sum = float(pnl_series.sum())

        vol_pen = float(risk_penalty_coef) * float(max(0.0, float(vol.max()))) / 10000.0
        reward_allow = float(pnl_sum - vol_pen)

        has_strong = 0.0
        if "has_strong_news_day" in g2.columns:
            try:
                has_strong = 1.0 if bool(pd.to_numeric(g2["has_strong_news_day"], errors="coerce").fillna(0.0).astype(bool).any()) else 0.0
            except Exception:
                has_strong = 0.0

        feats = {
            "date": str(date_str),
            "system": str(system),
            "n_tickers": float(len(g2)),
            "vol_mean": float(vol.mean()),
            "vol_max": float(vol.max()),
            "news_count_sum": float(news_count.sum()),
            "news_count_mean": float(news_count.mean()),
            "news_score_mean": float(news_score.mean()),
            "news_score_max": float(news_score.max()),
            "has_strong_news_day": float(has_strong),
            "gross_exposure": float(target_pos.abs().sum()),
            "net_exposure": float(target_pos.sum()),
            "abs_exposure_mean": float(target_pos.abs().mean()),
            "long_count": float((target_pos > 0).sum()),
            "short_count": float((target_pos < 0).sum()),
            "y_reward_allow": float(reward_allow),
        }

        dec_obj = decisions_by_date.get(str(date_str))
        if isinstance(dec_obj, dict):
            feats.update(_ensure_sft_context(day_obj=dec_obj, sft_planner=sft_planner))
        else:
            feats.update(
                {
                    "market_regime_score": 0.0,
                    "market_regime_is_risk_off": 0.0,
                    "market_regime_is_risk_on": 0.0,
                    "sft_is_aggressive_long": 0.0,
                    "sft_is_defensive": 0.0,
                    "sft_is_cash_preservation": 0.0,
                    "sft_confidence": 0.0,
                }
            )

        rows.append(feats)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = out["date"].astype(str)
    out = out.sort_values(["date", "system"])

    out["prev_gross_exposure"] = out.groupby("system")["gross_exposure"].shift(1).fillna(0.0)
    out["prev_net_exposure"] = out.groupby("system")["net_exposure"].shift(1).fillna(0.0)
    out["prev_abs_exposure_mean"] = out.groupby("system")["abs_exposure_mean"].shift(1).fillna(0.0)
    out["prev_long_count"] = out.groupby("system")["long_count"].shift(1).fillna(0.0)
    out["prev_short_count"] = out.groupby("system")["short_count"].shift(1).fillna(0.0)

    out = out.drop(
        columns=["gross_exposure", "net_exposure", "abs_exposure_mean", "long_count", "short_count"],
        errors="ignore",
    )

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 19.1: build offline dataset for RL gatekeeper (contextual bandit)")
    p.add_argument("--run-dir", dest="results_run_dir", action="append", required=True)
    p.add_argument("--system", default="golden_strict")
    p.add_argument("--out", default="data/training/rl_gatekeeper_dataset_v1.csv")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--reward-h", type=int, default=1)
    p.add_argument("--risk-penalty-coef", type=float, default=0.0)
    p.add_argument("--planner-sft-model", default="", help="Optional: if decisions lack planner probs, compute SFT strategy/probs using this model")
    args = p.parse_args()

    sft_planner = None
    if str(args.planner_sft_model).strip():
        try:
            from src.agent.planner import Planner

            sft_planner = Planner(policy="sft", sft_model_path=str(args.planner_sft_model).strip())
        except Exception:
            sft_planner = None

    dfs: List[pd.DataFrame] = []
    for rd in args.results_run_dir:
        dfs.append(
            build_dataset(
                run_dir=Path(str(rd)),
                system=str(args.system),
                reward_h=int(args.reward_h),
                risk_penalty_coef=float(args.risk_penalty_coef),
                sft_planner=sft_planner,
            )
        )

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "system"], keep="last").sort_values(["date", "system"])

    if str(args.start).strip():
        df = df[df["date"] >= str(args.start).strip()]
    if str(args.end).strip():
        df = df[df["date"] <= str(args.end).strip()]

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
