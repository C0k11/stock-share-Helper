import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


_DATE_RE = re.compile(r"stock_features_(\d{4}-\d{2}-\d{2})\.json$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _to_threshold(x: float) -> float:
    x = float(x)
    if abs(x) > 1.0:
        return abs(x) / 100.0
    return abs(x)


@dataclass
class DecisionItem:
    date: str
    ticker: str
    item: Dict[str, Any]


def _iter_decisions(payload: Dict[str, Any]) -> Iterable[DecisionItem]:
    if not isinstance(payload, dict):
        return

    if isinstance(payload.get("days"), dict):
        for date_str, day in payload.get("days").items():
            if not isinstance(day, dict):
                continue
            items = day.get("items")
            if not isinstance(items, dict):
                continue
            for ticker, it in items.items():
                if isinstance(it, dict):
                    yield DecisionItem(date=str(date_str), ticker=str(ticker).upper(), item=it)
        return

    date_str = str(payload.get("date") or "").strip()
    items = payload.get("items")
    if date_str and isinstance(items, dict):
        for ticker, it in items.items():
            if isinstance(it, dict):
                yield DecisionItem(date=str(date_str), ticker=str(ticker).upper(), item=it)


def _load_decisions_by_date(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = _read_json(path)
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


class FeatureCache:
    def __init__(self, daily_dir: Path) -> None:
        self.daily_dir = daily_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._close_cache: Dict[str, Dict[str, float]] = {}

    def get_stock_item(self, date_str: str, ticker: str) -> Optional[Dict[str, Any]]:
        date_str = str(date_str)
        ticker = str(ticker).upper()
        if date_str not in self._cache:
            fp = self.daily_dir / f"stock_features_{date_str}.json"
            if not fp.exists():
                self._cache[date_str] = {}
            else:
                payload = _read_json(fp)
                items = payload.get("items") if isinstance(payload, dict) else None
                m: Dict[str, Any] = {}
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        sym = str(it.get("symbol") or "").upper().strip()
                        if sym:
                            m[sym] = it
                self._cache[date_str] = m
        return self._cache[date_str].get(ticker)

    def get_close(self, date_str: str, ticker: str) -> Optional[float]:
        date_str = str(date_str)
        ticker = str(ticker).upper().strip()
        if date_str not in self._close_cache:
            fp = self.daily_dir / f"stock_features_{date_str}.json"
            if not fp.exists():
                self._close_cache[date_str] = {}
            else:
                payload = _read_json(fp)
                items = payload.get("items") if isinstance(payload, dict) else None
                m: Dict[str, float] = {}
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        sym = str(it.get("symbol") or "").upper().strip()
                        if not sym:
                            continue
                        tech = it.get("technical") if isinstance(it.get("technical"), dict) else {}
                        close = tech.get("close")
                        if close is None:
                            continue
                        v = _to_float(close, default=0.0)
                        if abs(v) > 1e-12:
                            m[sym] = float(v)
                self._close_cache[date_str] = m
        return self._close_cache[date_str].get(ticker)


def _list_available_stock_feature_dates(daily_dir: Path) -> List[str]:
    out: List[str] = []
    for fp in daily_dir.glob("stock_features_*.json"):
        m = _DATE_RE.search(fp.name)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def _forward_return_from_closes(
    *,
    date_str: str,
    sym: str,
    trading_dates: List[str],
    fc: FeatureCache,
    horizon: int,
) -> Optional[float]:
    if horizon <= 0:
        return 0.0
    try:
        idx = trading_dates.index(str(date_str))
    except ValueError:
        return None
    j = idx + int(horizon)
    if j >= len(trading_dates):
        return None
    d0 = str(date_str)
    d1 = str(trading_dates[j])
    sym_u = str(sym).upper().strip()
    c0 = fc.get_close(d0, sym_u)
    c1 = fc.get_close(d1, sym_u)
    if c0 is None or c1 is None:
        return None
    if abs(float(c0)) < 1e-12:
        return None
    return float(c1) / float(c0) - 1.0


def _extract_news_title_phrase(news_contexts: List[str]) -> str:
    for ctx in news_contexts:
        s = str(ctx)
        m = None
        for line in s.splitlines():
            if line.strip().lower().startswith("title:"):
                m = line.split(":", 1)[1].strip()
                break
        if m:
            return m[:60]
    return ""


def _synthetic_clear_json(*, ticker: str, news_contexts: List[str], variant: str) -> str:
    quote = _extract_news_title_phrase(news_contexts)
    if variant == "punish_wrong_buy":
        analysis = "Negative forward return; safer to stay in cash."
        b3 = "3. Remain in cash to avoid drawdown risk."
    else:
        analysis = "Avoiding exposure would miss an uptrend."
        b3 = "3. Staying out risks missing a meaningful upside move."

    b1 = "1. Risk control: capital preservation is prioritized."
    if quote:
        b2 = f"2. News is mixed; do not overreact to \"{quote}\"."
    else:
        b2 = "2. Technical support is insufficient to justify exposure."

    obj = {
        "decision": "CLEAR",
        "ticker": str(ticker).upper(),
        "analysis": analysis,
        "reasoning_trace": [b1, b2, b3],
    }
    return json.dumps(obj, ensure_ascii=False)


def _stringify_original_output(it: Dict[str, Any]) -> str:
    raw = it.get("raw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    parsed = it.get("parsed")
    if isinstance(parsed, dict) and parsed:
        return json.dumps(parsed, ensure_ascii=False)
    return json.dumps({"decision": "HOLD", "ticker": it.get("ticker") or "", "analysis": ""}, ensure_ascii=False)


def _get_model_decision(it: Dict[str, Any]) -> str:
    parsed = it.get("parsed")
    if isinstance(parsed, dict):
        d = str(parsed.get("decision") or "").strip().upper()
        if d:
            return d
    raw = it.get("raw")
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                d = str(obj.get("decision") or "").strip().upper()
                if d:
                    return d
        except Exception:
            pass
    return ""


def _infer_expert(it: Dict[str, Any]) -> str:
    expert = str(it.get("expert") or "").strip().lower()
    if expert:
        return expert
    router = it.get("router") if isinstance(it.get("router"), dict) else {}
    return str(router.get("expert") or "").strip().lower()


def _infer_expert_before_planner_gate(it: Dict[str, Any]) -> str:
    router = it.get("router") if isinstance(it.get("router"), dict) else {}
    return str(router.get("expert_before_planner_gate") or "").strip().lower()


def _patch_system_prompt_allow_clear(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        return messages
    m0 = messages[0] if isinstance(messages[0], dict) else None
    if not isinstance(m0, dict):
        return messages
    if str(m0.get("role") or "") != "system":
        return messages

    content = str(m0.get("content") or "")
    content2 = content.replace('"decision": "BUY" | "SELL" | "HOLD"', '"decision": "BUY" | "SELL" | "HOLD" | "CLEAR"')
    if content2 == content:
        return messages

    out = list(messages)
    out[0] = dict(m0)
    out[0]["content"] = content2
    return out


def _extract_gate_context(day_obj: Dict[str, Any]) -> Dict[str, float]:
    planner_obj = day_obj.get("planner") if isinstance(day_obj.get("planner"), dict) else {}
    inputs = planner_obj.get("inputs") if isinstance(planner_obj.get("inputs"), dict) else {}

    mr = inputs.get("market_regime") if isinstance(inputs.get("market_regime"), dict) else {}
    mr_regime = str(mr.get("regime") or "").strip().lower()
    mr_score = _to_float(mr.get("score"), default=0.0)

    probs = inputs.get("probs") if isinstance(inputs.get("probs"), dict) else {}
    p_aggr = _to_float(probs.get("aggressive_long"), default=0.0)
    p_def = _to_float(probs.get("defensive"), default=0.0)
    p_cash = _to_float(probs.get("cash_preservation"), default=0.0)
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


def _ensure_sft_context(*, day_obj: Dict[str, Any], sft_planner: Any) -> Dict[str, float]:
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
        dec = sft_planner.decide(market_regime=mr, features={str(k): _to_float(v) for k, v in feats.items()}).to_dict()
    except Exception:
        return base

    tmp_day = dict(day_obj)
    tmp_day["planner"] = dec
    return _extract_gate_context(tmp_day)


def _extract_gatekeeper_fields(day_obj: Dict[str, Any]) -> Dict[str, Any]:
    gk = day_obj.get("gatekeeper") if isinstance(day_obj.get("gatekeeper"), dict) else {}
    present = bool(gk)
    allow = gk.get("allow")
    q_allow = gk.get("q_allow")
    thr = gk.get("threshold")
    return {
        "gate_present": 1.0 if present else 0.0,
        "gate_allow": bool(allow) if allow is not None else (not present),
        "gate_q_allow": _to_float(q_allow, default=0.0),
        "gate_threshold": _to_float(thr, default=0.0),
    }


def _resolve_decision_files(inputs: List[str], system_filter: str) -> List[Path]:
    out: List[Path] = []
    for x in inputs:
        p = Path(str(x))
        if p.is_file() and p.name.startswith("decisions_") and p.suffix.lower() == ".json":
            if system_filter:
                if str(p.parent.name).strip() != system_filter:
                    continue
            out.append(p)
            continue
        if p.is_dir():
            for fp in p.rglob("decisions_*.json"):
                if system_filter and str(fp.parent.name).strip() != system_filter:
                    continue
                out.append(fp)
            continue
        for fp in sorted(Path().glob(str(x))):
            if fp.is_file() and fp.name.startswith("decisions_") and fp.suffix.lower() == ".json":
                if system_filter and str(fp.parent.name).strip() != system_filter:
                    continue
                out.append(fp)
    out = sorted(set(out))
    return out


def _load_latest_daily_csv(system_dir: Path) -> Optional[Path]:
    p = system_dir / "daily.csv"
    if p.exists():
        return p
    cands = sorted(system_dir.glob("daily_*.csv"))
    if not cands:
        return None
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 20.1: Unified Data Harvester (Analyst DPO pairs + Gatekeeper RL rewards)")
    ap.add_argument("--inputs", nargs="+", required=True, help="Decision JSON files, run dirs, system dirs, or glob patterns")
    ap.add_argument("--daily-dir", default="data/daily", help="Directory containing stock_features_YYYY-MM-DD.json")
    ap.add_argument("--system", default="golden_strict", help="Only process decisions under this system dir (e.g. golden_strict). Use empty string to disable.")

    ap.add_argument("--out-dpo", default="data/flywheel/analyst_dpo_pairs.jsonl")
    ap.add_argument("--out-rl", default="data/flywheel/gatekeeper_rl_rewards.csv")

    ap.add_argument("--dpo-horizon", type=int, default=5)
    ap.add_argument("--dpo-x", type=float, default=0.02)
    ap.add_argument("--dpo-target-expert", default="analyst", choices=["analyst", "scalper", "any"])
    ap.add_argument("--dpo-only-buy", action="store_true", default=True)
    ap.add_argument("--dpo-include-nonbuy", dest="dpo_only_buy", action="store_false")
    ap.add_argument("--min-abs-impact", type=float, default=0.5)
    ap.add_argument("--max-news-signals", type=int, default=3)
    ap.add_argument("--max-dpo-pairs", type=int, default=0)

    ap.add_argument("--reward-h", type=int, default=1)
    ap.add_argument("--risk-penalty-coef", type=float, default=0.0)
    ap.add_argument("--planner-sft-model", default="")

    args = ap.parse_args()

    daily_dir = Path(str(args.daily_dir))
    daily_dir.mkdir(parents=True, exist_ok=True)

    sys_filter = str(args.system or "").strip()
    decision_files = _resolve_decision_files([str(x) for x in args.inputs], sys_filter)
    if not decision_files:
        raise SystemExit("No decisions_*.json found for given inputs")

    dpo_h = int(args.dpo_horizon)
    if dpo_h <= 0:
        raise SystemExit("dpo_horizon must be > 0")

    thr = _to_threshold(float(args.dpo_x))

    try:
        from scripts.run_trading_inference import build_stock_messages, load_daily_news_contexts
    except Exception as e:
        raise SystemExit(f"Failed to import prompt builders from scripts/run_trading_inference.py: {e}")

    sft_planner = None
    if str(args.planner_sft_model).strip():
        try:
            from src.agent.planner import Planner

            sft_planner = Planner(policy="sft", sft_model_path=str(args.planner_sft_model).strip())
        except Exception:
            sft_planner = None

    fc = FeatureCache(daily_dir)
    trading_dates = _list_available_stock_feature_dates(daily_dir)
    if not trading_dates:
        raise SystemExit(f"No stock_features_*.json found under {daily_dir}")

    out_dpo = Path(str(args.out_dpo))
    out_dpo.parent.mkdir(parents=True, exist_ok=True)

    out_rl = Path(str(args.out_rl))
    out_rl.parent.mkdir(parents=True, exist_ok=True)

    rl_rows: List[Dict[str, Any]] = []

    written = 0
    n_total = 0
    n_skip_expert = 0
    n_skip_not_buy = 0
    n_skip_no_feat = 0
    n_skip_mid = 0

    def should_keep_expert(it: Dict[str, Any]) -> bool:
        te = str(args.dpo_target_expert)
        if te == "any":
            return True
        if _infer_expert(it) == te:
            return True
        if te == "analyst" and _infer_expert_before_planner_gate(it) == "analyst":
            return True
        return False

    with out_dpo.open("w", encoding="utf-8") as f_out:
        for decisions_path in decision_files:
            system_dir = decisions_path.parent
            run_dir = system_dir.parent
            system_name = str(system_dir.name)
            source_run = str(run_dir.name)

            payload = _read_json(decisions_path)
            for d in _iter_decisions(payload):
                n_total += 1
                it = d.item
                if not should_keep_expert(it):
                    n_skip_expert += 1
                    continue

                model_decision = _get_model_decision(it)
                if bool(args.dpo_only_buy) and model_decision != "BUY":
                    n_skip_not_buy += 1
                    continue

                fr = _forward_return_from_closes(date_str=str(d.date), sym=str(d.ticker), trading_dates=trading_dates, fc=fc, horizon=dpo_h)
                if fr is None:
                    n_skip_no_feat += 1
                    continue

                if (-thr < float(fr) < thr):
                    n_skip_mid += 1
                    continue

                stock_item = fc.get_stock_item(str(d.date), str(d.ticker))
                if stock_item is None:
                    n_skip_no_feat += 1
                    continue

                news_contexts = load_daily_news_contexts(
                    daily_dir=daily_dir,
                    date_str=str(d.date),
                    signals_path="",
                    min_abs_impact=float(args.min_abs_impact),
                    max_signals=int(args.max_news_signals),
                    ticker=str(d.ticker),
                )

                messages = build_stock_messages(str(d.ticker), str(d.date), stock_item, news_contexts)
                messages = _patch_system_prompt_allow_clear(messages)

                original = _stringify_original_output(it)

                if float(fr) <= -thr:
                    chosen = _synthetic_clear_json(ticker=d.ticker, news_contexts=news_contexts, variant="punish_wrong_buy")
                    rejected = original
                else:
                    chosen = original
                    rejected = _synthetic_clear_json(ticker=d.ticker, news_contexts=news_contexts, variant="miss_uptrend")

                row = {
                    "prompt": messages,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "date": str(d.date),
                        "ticker": str(d.ticker),
                        "horizon": int(dpo_h),
                        "x": float(thr),
                        "forward_return": float(fr),
                        "model_decision": str(model_decision),
                        "expert": _infer_expert(it),
                        "source_run": str(source_run),
                        "system": str(system_name),
                        "source_file": str(decisions_path).replace("\\", "/"),
                    },
                }

                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                if int(args.max_dpo_pairs) > 0 and written >= int(args.max_dpo_pairs):
                    break
            if int(args.max_dpo_pairs) > 0 and written >= int(args.max_dpo_pairs):
                break

            daily_path = _load_latest_daily_csv(system_dir)
            if daily_path is None:
                continue

            df = pd.read_csv(daily_path)
            if "date" not in df.columns:
                continue

            decisions_by_date: Dict[str, Dict[str, Any]] = {}
            try:
                decisions_by_date = _load_decisions_by_date(decisions_path)
            except Exception:
                decisions_by_date = {}

            pnl_col = f"pnl_h{int(args.reward_h)}_net"

            for date_str, g in df.groupby("date"):
                g2 = g.copy()

                news_count = pd.to_numeric(g2.get("news_count", 0.0), errors="coerce").fillna(0.0)
                news_score = pd.to_numeric(g2.get("news_score", 0.0), errors="coerce").fillna(0.0)
                vol = pd.to_numeric(g2.get("volatility_ann_pct", 0.0), errors="coerce").fillna(0.0)

                target_pos = pd.to_numeric(g2.get("target_position", 0.0), errors="coerce").fillna(0.0)
                pnl_series = pd.to_numeric(g2.get(pnl_col, 0.0), errors="coerce").fillna(0.0)
                pnl_sum = float(pnl_series.sum())

                vol_pen = float(args.risk_penalty_coef) * float(max(0.0, float(vol.max()))) / 10000.0
                reward_allow = float(pnl_sum - vol_pen)

                has_strong = 0.0
                if "has_strong_news_day" in g2.columns:
                    try:
                        has_strong = 1.0 if bool(pd.to_numeric(g2["has_strong_news_day"], errors="coerce").fillna(0.0).astype(bool).any()) else 0.0
                    except Exception:
                        has_strong = 0.0

                feats: Dict[str, Any] = {
                    "source_run": str(source_run),
                    "date": str(date_str),
                    "system": str(system_name),
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

                day_obj = decisions_by_date.get(str(date_str))
                if isinstance(day_obj, dict):
                    feats.update(_ensure_sft_context(day_obj=day_obj, sft_planner=sft_planner))
                    feats.update(_extract_gatekeeper_fields(day_obj))
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
                            "gate_present": 0.0,
                            "gate_allow": False,
                            "gate_q_allow": 0.0,
                            "gate_threshold": 0.0,
                        }
                    )

                action_allow = 1.0 if bool(feats.get("gate_allow")) else 0.0
                feats["action_allow"] = float(action_allow)
                feats["y_reward_realized"] = float(reward_allow) if action_allow > 0.0 else 0.0
                feats["y_reward_deny"] = 0.0

                rl_rows.append(feats)

    df_rl = pd.DataFrame(rl_rows)
    if not df_rl.empty:
        df_rl["date"] = df_rl["date"].astype(str)
        df_rl = df_rl.sort_values(["source_run", "system", "date"]).reset_index(drop=True)

        df_rl["prev_gross_exposure"] = df_rl.groupby(["source_run", "system"])["gross_exposure"].shift(1).fillna(0.0)
        df_rl["prev_net_exposure"] = df_rl.groupby(["source_run", "system"])["net_exposure"].shift(1).fillna(0.0)
        df_rl["prev_abs_exposure_mean"] = df_rl.groupby(["source_run", "system"])["abs_exposure_mean"].shift(1).fillna(0.0)
        df_rl["prev_long_count"] = df_rl.groupby(["source_run", "system"])["long_count"].shift(1).fillna(0.0)
        df_rl["prev_short_count"] = df_rl.groupby(["source_run", "system"])["short_count"].shift(1).fillna(0.0)

        df_rl = df_rl.drop(columns=["gross_exposure", "net_exposure", "abs_exposure_mean", "long_count", "short_count"], errors="ignore")

        df_rl = df_rl.drop_duplicates(subset=["source_run", "system", "date"], keep="last")
        df_rl.to_csv(out_rl, index=False)

    print(
        json.dumps(
            {
                "inputs": [str(p).replace("\\", "/") for p in decision_files],
                "out_dpo": str(out_dpo).replace("\\", "/"),
                "out_rl": str(out_rl).replace("\\", "/"),
                "dpo": {
                    "pairs_written": int(written),
                    "n_total_items": int(n_total),
                    "skips": {
                        "expert_filter": int(n_skip_expert),
                        "not_buy": int(n_skip_not_buy),
                        "no_feature": int(n_skip_no_feat),
                        "mid_band": int(n_skip_mid),
                    },
                },
                "rl": {"rows_written": int(len(df_rl))},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
