#!/usr/bin/env python

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_ymd(s: str) -> Tuple[int, int, int]:
    a = str(s).strip().split("-")
    if len(a) != 3:
        raise ValueError(f"invalid date: {s}")
    return int(a[0]), int(a[1]), int(a[2])


def _iter_dates_inclusive(start_ymd: str, end_ymd: str) -> List[str]:
    import datetime as _dt

    y, m, d = _parse_ymd(start_ymd)
    a = _dt.date(y, m, d)
    y, m, d = _parse_ymd(end_ymd)
    b = _dt.date(y, m, d)
    if b < a:
        a, b = b, a
    out: List[str] = []
    cur = a
    while cur <= b:
        out.append(cur.isoformat())
        cur = cur + _dt.timedelta(days=1)
    return out


def _iter_decision_records(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return

    if isinstance(payload.get("days"), dict):
        days = payload.get("days")
        if isinstance(days, dict):
            for date_str, day in days.items():
                if not isinstance(day, dict):
                    continue
                items = day.get("items") if isinstance(day.get("items"), dict) else {}
                for sym, rec in items.items():
                    if not isinstance(rec, dict):
                        continue
                    yield str(date_str), str(sym).upper().strip(), rec
        return

    date_str = str(payload.get("date") or "")
    items = payload.get("items") if isinstance(payload.get("items"), dict) else {}
    for sym, rec in items.items():
        if not isinstance(rec, dict):
            continue
        yield date_str, str(sym).upper().strip(), rec


def _get_final(rec: Dict[str, Any]) -> Dict[str, Any]:
    final = rec.get("final") if isinstance(rec.get("final"), dict) else {}
    return final


def _get_position(rec: Dict[str, Any]) -> float:
    return _to_float(_get_final(rec).get("target_position"))


def _get_action(rec: Dict[str, Any]) -> str:
    a = _get_final(rec).get("action")
    return str(a).upper().strip() if isinstance(a, str) else ""


def _get_expert(rec: Dict[str, Any], default_expert: str) -> str:
    e = rec.get("expert")
    if isinstance(e, str) and e.strip():
        return e.strip()
    return str(default_expert).strip() or "unknown"


def _get_router(rec: Dict[str, Any]) -> Dict[str, Any]:
    r = rec.get("router")
    return r if isinstance(r, dict) else {}


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _load_stock_maps(daily_dir: Path, dates: List[str]) -> Tuple[List[str], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    close_map: Dict[Tuple[str, str], float] = {}
    vol_map: Dict[Tuple[str, str], float] = {}
    available_dates: List[str] = []

    for d in sorted(set([str(x) for x in dates if str(x).strip()])):
        fp = daily_dir / f"stock_features_{d}.json"
        if not fp.exists():
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        any_row = False
        for it in items:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").upper().strip()
            if not sym:
                continue
            tech = it.get("technical") if isinstance(it.get("technical"), dict) else {}
            close = tech.get("close")
            if close is not None:
                close_map[(str(d), sym)] = _to_float(close)
                any_row = True

            vol = it.get("volatility_ann_pct")
            if vol is None:
                vol = tech.get("volatility_20d")
            if vol is not None:
                vol_map[(str(d), sym)] = _to_float(vol)

        if any_row:
            available_dates.append(str(d))

    available_dates = sorted(set(available_dates))
    return available_dates, close_map, vol_map


def _forward_return_from_closes(
    date_str: str,
    sym: str,
    trading_dates: List[str],
    close_map: Dict[Tuple[str, str], float],
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
    c0 = close_map.get((d0, sym_u))
    c1 = close_map.get((d1, sym_u))
    if c0 is None or c1 is None:
        return None
    if abs(float(c0)) < 1e-12:
        return None
    return float(c1) / float(c0) - 1.0


def _detect_strong_news_days(signals_dir: Path, dates: List[str], min_abs_impact: float) -> set:
    strong: set = set()
    for d in sorted(set([str(x) for x in dates if str(x).strip()])):
        fp = signals_dir / f"signals_{d}.json"
        if not fp.exists():
            continue
        try:
            arr = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(arr, list):
            continue
        ok = False
        for it in arr:
            if not isinstance(it, dict):
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            ie = sig.get("impact_equity")
            if ie is None:
                continue
            if abs(_to_float(ie)) >= float(min_abs_impact):
                ok = True
                break
        if ok:
            strong.add(str(d))
    return strong


@dataclass
class SliceMetric:
    items_scored: int = 0
    trades_scored: int = 0
    wins: int = 0
    pnl_sum: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        winrate = (float(self.wins) / float(self.trades_scored)) if self.trades_scored else 0.0
        avg_pnl = (float(self.pnl_sum) / float(self.trades_scored)) if self.trades_scored else 0.0
        return {
            "items_scored": int(self.items_scored),
            "trades_scored": int(self.trades_scored),
            "wins": int(self.wins),
            "winrate": float(winrate),
            "pnl_sum": float(self.pnl_sum),
            "avg_pnl_per_trade": float(avg_pnl),
        }


def _score_system(
    decision_path: Path,
    default_expert: str,
    daily_dir: Path,
    date_list: List[str],
    horizons: List[int],
    min_news_abs_impact: float,
    high_vol_ann_pct_threshold: float,
    planner_non_aggressive_dates: set,
    trade_cost_bps: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    payload = json.loads(decision_path.read_text(encoding="utf-8"))

    all_dates: List[str] = []
    for d, _, _ in _iter_decision_records(payload):
        if d:
            all_dates.append(str(d))
    if date_list:
        all_dates.extend(date_list)

    trading_dates, close_map, vol_map = _load_stock_maps(daily_dir, all_dates)

    strong_news_days = _detect_strong_news_days(daily_dir, all_dates, float(min_news_abs_impact))

    rows: List[Dict[str, Any]] = []
    for d, sym, rec in _iter_decision_records(payload):
        if not d:
            continue
        final = _get_final(rec)
        pos = _get_position(rec)
        expert = _get_expert(rec, default_expert)
        router = _get_router(rec)

        vol_ann = router.get("volatility_ann_pct")
        if vol_ann is None:
            vol_ann = vol_map.get((str(d), str(sym).upper().strip()))
        vol_ann_f = _to_float(vol_ann)

        news_count = router.get("news_count")
        news_count_i = int(news_count) if isinstance(news_count, int) else int(_to_float(news_count))
        has_news = bool(news_count_i > 0) if router else (str(d) in strong_news_days)

        planner_strategy = router.get("planner_strategy") if router else ""

        rows.append(
            {
                "date": str(d),
                "ticker": str(sym).upper().strip(),
                "expert": str(expert),
                "action": _get_action(rec),
                "target_position": float(pos),
                "volatility_ann_pct": float(vol_ann_f),
                "has_news": bool(has_news),
                "planner_strategy": str(planner_strategy or ""),
                "risk_trace": _safe_text(final.get("trace")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        metrics = {str(h): {} for h in horizons}
        return metrics, df

    # Turnover-based transaction costs (charged on each date per ticker)
    # turnover := abs(pos_t - pos_{t-1})
    # cost_return := turnover * (trade_cost_bps / 10000)
    cost_rate = float(trade_cost_bps) / 10000.0 if float(trade_cost_bps) > 0 else 0.0
    turnover_map: Dict[Tuple[str, str], float] = {}
    if cost_rate > 0 and trading_dates:
        last_pos: Dict[str, float] = {}
        # Only compute turnover on trading dates we have features for, so ordering is deterministic.
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_date.setdefault(str(r["date"]), []).append(r)
        for d in trading_dates:
            day_rows = by_date.get(str(d), [])
            for r in day_rows:
                sym = str(r["ticker"]).upper().strip()
                pos = float(r["target_position"])
                prev = float(last_pos.get(sym, 0.0))
                turnover = abs(pos - prev)
                turnover_map[(str(d), sym)] = float(turnover)
                last_pos[sym] = float(pos)

    metrics: Dict[str, Any] = {}

    for h in horizons:
        slice_defs = {
            "all": lambda r: True,
            "high_news": lambda r: bool(r["has_news"]),
            "high_vol": lambda r: float(r["volatility_ann_pct"]) >= float(high_vol_ann_pct_threshold),
            "planner_non_aggressive": lambda r: str(r["date"]) in planner_non_aggressive_dates,
        }

        h_metrics: Dict[str, Any] = {}

        for slice_name, pred in slice_defs.items():
            m_all = SliceMetric()
            m_by_expert: Dict[str, SliceMetric] = {}

            for r in rows:
                if not pred(r):
                    continue
                fr = _forward_return_from_closes(
                    date_str=str(r["date"]),
                    sym=str(r["ticker"]),
                    trading_dates=trading_dates,
                    close_map=close_map,
                    horizon=int(h),
                )
                if fr is None:
                    continue

                m_all.items_scored += 1

                pos = float(r["target_position"])
                sym_u = str(r["ticker"]).upper().strip()
                turnover = float(turnover_map.get((str(r["date"]), sym_u), 0.0))
                cost_penalty = float(turnover) * float(cost_rate)
                if abs(pos) > 1e-9:
                    m_all.trades_scored += 1
                    pnl = float(pos) * float(fr) - float(cost_penalty)
                    m_all.pnl_sum += pnl
                    if pnl > 0:
                        m_all.wins += 1

                exp = str(r.get("expert") or "unknown")
                if exp not in m_by_expert:
                    m_by_expert[exp] = SliceMetric()
                m_by_expert[exp].items_scored += 1
                if abs(pos) > 1e-9:
                    m_by_expert[exp].trades_scored += 1
                    pnl = float(pos) * float(fr) - float(cost_penalty)
                    m_by_expert[exp].pnl_sum += pnl
                    if pnl > 0:
                        m_by_expert[exp].wins += 1

            h_metrics[slice_name] = {
                "all": m_all.to_dict(),
                "by_expert": {k: v.to_dict() for k, v in sorted(m_by_expert.items(), key=lambda x: x[0])},
            }

        metrics[str(h)] = h_metrics

    return metrics, df


def _planner_non_aggressive_dates_from_moe(moe_path: Path) -> set:
    try:
        payload = json.loads(moe_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    days = payload.get("days") if isinstance(payload, dict) else None
    if not isinstance(days, dict):
        return set()

    out: set = set()
    for d, day in days.items():
        if not isinstance(day, dict):
            continue
        strat = ""
        planner = day.get("planner") if isinstance(day.get("planner"), dict) else {}
        s = planner.get("strategy")
        if isinstance(s, str) and s.strip():
            strat = s.strip()
        if not strat:
            items = day.get("items") if isinstance(day.get("items"), dict) else {}
            for _sym, rec in items.items():
                if isinstance(rec, dict):
                    router = rec.get("router") if isinstance(rec.get("router"), dict) else {}
                    rs = router.get("planner_strategy")
                    if isinstance(rs, str) and rs.strip():
                        strat = rs.strip()
                        break
        if strat and strat != "aggressive_long":
            out.add(str(d))
    return out


def _run_inference_if_missing(
    *,
    cfg: Dict[str, Any],
    out_path: Path,
    start: str,
    end: str,
    universe: str,
    daily_dir: Path,
    base_model: str,
    load_4bit: bool,
) -> None:
    if out_path.exists():
        return

    args: List[str] = [
        sys.executable,
        "scripts/run_trading_inference.py",
        "--date-range",
        str(start),
        str(end),
        "--daily-dir",
        str(daily_dir),
        "--universe",
        str(universe),
        "--model",
        str(base_model),
        "--out",
        str(out_path),
    ]

    if load_4bit:
        args.append("--load-in-4bit")
    else:
        args.append("--no-load-4bit")

    if bool(cfg.get("moe_mode")):
        args.append("--moe-mode")
        if str(cfg.get("moe_scalper") or "").strip():
            args.extend(["--moe-scalper", str(cfg.get("moe_scalper"))])
        if str(cfg.get("moe_analyst") or "").strip():
            args.extend(["--moe-analyst", str(cfg.get("moe_analyst"))])
        if str(cfg.get("planner_mode") or "").strip():
            args.extend(["--planner-mode", str(cfg.get("planner_mode"))])
        if cfg.get("moe_any_news") is False:
            args.append("--no-moe-any-news")
        if cfg.get("moe_any_news") is True:
            args.append("--moe-any-news")
        if cfg.get("moe_news_threshold") is not None:
            args.extend(["--moe-news-threshold", str(cfg.get("moe_news_threshold"))])
        if cfg.get("moe_vol_threshold") is not None:
            args.extend(["--moe-vol-threshold", str(cfg.get("moe_vol_threshold"))])
    else:
        if bool(cfg.get("use_lora")):
            args.append("--use-lora")
        if str(cfg.get("adapter") or "").strip():
            args.extend(["--adapter", str(cfg.get("adapter"))])
        if str(cfg.get("planner_mode") or "").strip():
            args.extend(["--planner-mode", str(cfg.get("planner_mode"))])

    if bool(cfg.get("allow_clear")):
        args.append("--allow-clear")

    if cfg.get("risk_max_drawdown") is not None:
        args.extend(["--risk-max-drawdown", str(cfg.get("risk_max_drawdown"))])
    if cfg.get("risk_vol_limit") is not None:
        args.extend(["--risk-vol-limit", str(cfg.get("risk_vol_limit"))])
    if str(cfg.get("risk_watch_market") or "").strip():
        args.extend(["--risk-watch-market", str(cfg.get("risk_watch_market"))])

    if cfg.get("max_new_tokens") is not None:
        args.extend(["--max-new-tokens", str(cfg.get("max_new_tokens"))])
    if cfg.get("temperature") is not None:
        args.extend(["--temperature", str(cfg.get("temperature"))])

    subprocess.run(args, check=True)


def _materialize_existing_output(*, src: str, dst: Path, force: bool) -> bool:
    src_path = Path(str(src))
    if not str(src).strip() or not src_path.exists() or not src_path.is_file():
        return False
    if dst.exists() and not force:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="golden_strict_config.yaml")
    p.add_argument("--run-id", default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--force-rerun", action="store_true", default=False)
    args = p.parse_args()

    cfg_path = Path(str(args.config))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    start = str(args.start or (cfg.get("date_range") or {}).get("start") or "").strip()
    end = str(args.end or (cfg.get("date_range") or {}).get("end") or "").strip()
    if not start or not end:
        raise SystemExit("missing start/end date")

    universe = str(cfg.get("universe") or "stock")
    daily_dir = Path(str(cfg.get("daily_dir") or "data/daily"))
    base_model = str(cfg.get("base_model") or "Qwen/Qwen2.5-7B-Instruct")
    load_4bit = bool(cfg.get("load_4bit") is True)

    horizons = list((cfg.get("eval") or {}).get("horizons") or [1, 5, 10])
    horizons = [int(x) for x in horizons]

    min_news_abs_impact = float((cfg.get("eval") or {}).get("min_news_abs_impact") or 0.5)
    high_vol_thr = float((cfg.get("eval") or {}).get("high_vol_ann_pct_threshold") or 25.0)
    trade_cost_bps = float((cfg.get("eval") or {}).get("trade_cost_bps") or 0.0)

    run_id = str(args.run_id).strip()
    if not run_id:
        run_id = f"{start}_to_{end}".replace("-", "")

    results_dir = Path(str((cfg.get("outputs") or {}).get("results_dir") or "results")) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Decide output paths (copy default out paths into results dir)
    systems = cfg.get("systems") if isinstance(cfg.get("systems"), dict) else {}
    baseline_cfg = (systems.get("baseline_fast") or {}).get("inference") or {}
    golden_cfg = (systems.get("golden_strict") or {}).get("inference") or {}

    baseline_src = str((systems.get("baseline_fast") or {}).get("out") or "").strip()
    golden_src = str((systems.get("golden_strict") or {}).get("out") or "").strip()

    baseline_out = results_dir / "baseline_fast.json"
    golden_out = results_dir / "golden_strict.json"

    if bool(args.force_rerun):
        if baseline_out.exists():
            baseline_out.unlink()
        if golden_out.exists():
            golden_out.unlink()

    force = bool(args.force_rerun)
    baseline_ready = _materialize_existing_output(src=baseline_src, dst=baseline_out, force=force)
    golden_ready = _materialize_existing_output(src=golden_src, dst=golden_out, force=force)

    if not baseline_ready:
        _run_inference_if_missing(
            cfg=baseline_cfg,
            out_path=baseline_out,
            start=start,
            end=end,
            universe=universe,
            daily_dir=daily_dir,
            base_model=base_model,
            load_4bit=load_4bit,
        )

    if not golden_ready:
        _run_inference_if_missing(
            cfg=golden_cfg,
            out_path=golden_out,
            start=start,
            end=end,
            universe=universe,
            daily_dir=daily_dir,
            base_model=base_model,
            load_4bit=load_4bit,
        )

    if not baseline_out.exists() or not golden_out.exists():
        raise SystemExit("missing decision outputs")

    date_list = _iter_dates_inclusive(start, end)

    planner_non_aggressive_dates = _planner_non_aggressive_dates_from_moe(golden_out)

    baseline_metrics, baseline_df = _score_system(
        baseline_out,
        default_expert="scalper",
        daily_dir=daily_dir,
        date_list=date_list,
        horizons=horizons,
        min_news_abs_impact=min_news_abs_impact,
        high_vol_ann_pct_threshold=high_vol_thr,
        planner_non_aggressive_dates=planner_non_aggressive_dates,
        trade_cost_bps=trade_cost_bps,
    )

    golden_metrics, golden_df = _score_system(
        golden_out,
        default_expert="unknown",
        daily_dir=daily_dir,
        date_list=date_list,
        horizons=horizons,
        min_news_abs_impact=min_news_abs_impact,
        high_vol_ann_pct_threshold=high_vol_thr,
        planner_non_aggressive_dates=planner_non_aggressive_dates,
        trade_cost_bps=trade_cost_bps,
    )

    metrics_out = {
        "protocol": {
            "start": start,
            "end": end,
            "universe": universe,
            "base_model": base_model,
            "horizons": horizons,
            "min_news_abs_impact": min_news_abs_impact,
            "high_vol_ann_pct_threshold": high_vol_thr,
            "trade_cost_bps": trade_cost_bps,
        },
        "inputs": {
            "baseline_decisions": str(baseline_out),
            "golden_decisions": str(golden_out),
        },
        "slices": {
            "planner_non_aggressive_dates": sorted(list(planner_non_aggressive_dates)),
        },
        "results": {
            "baseline_fast": baseline_metrics,
            "golden_strict": golden_metrics,
        },
    }

    (results_dir / str((cfg.get("outputs") or {}).get("metrics_filename") or "metrics.json")).write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if bool((cfg.get("outputs") or {}).get("write_parquet") is True):
        out_name = str((cfg.get("outputs") or {}).get("parquet_filename") or "daily_decisions.parquet")
        df_all = pd.concat(
            [
                baseline_df.assign(system="baseline_fast"),
                golden_df.assign(system="golden_strict"),
            ],
            ignore_index=True,
        )
        df_all.to_parquet(results_dir / out_name, index=False)

    print(f"Saved: {results_dir}")


if __name__ == "__main__":
    main()
