#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_single_decisions_file(system_dir: Path) -> Optional[Path]:
    cands = sorted(system_dir.glob("decisions_*.json"))
    if not cands:
        return None
    return cands[-1]


def _iter_decisions_items(payload: Any) -> List[Tuple[str, str, Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return []

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    if isinstance(payload.get("days"), dict):
        for date_str, day in payload.get("days").items():
            if not isinstance(day, dict):
                continue
            items = day.get("items")
            if not isinstance(items, dict):
                continue
            for ticker, it in items.items():
                if isinstance(it, dict):
                    out.append((str(date_str), str(ticker).upper(), it))
        return out

    date_str = str(payload.get("date") or "").strip()
    items = payload.get("items")
    if date_str and isinstance(items, dict):
        for ticker, it in items.items():
            if isinstance(it, dict):
                out.append((str(date_str), str(ticker).upper(), it))
    return out


def _get_decision_from_item(it: Dict[str, Any]) -> str:
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
    final = it.get("final") if isinstance(it.get("final"), dict) else {}
    d = str(final.get("action") or "").strip().upper()
    return d


def _stringify_model_output(it: Dict[str, Any]) -> str:
    raw = it.get("raw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    parsed = it.get("parsed")
    if isinstance(parsed, dict) and parsed:
        return json.dumps(parsed, ensure_ascii=False)
    return json.dumps({"decision": "HOLD", "ticker": "", "analysis": ""}, ensure_ascii=False)


def _load_decision_map(decisions_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    try:
        payload = _read_json(decisions_path)
    except Exception:
        return {}

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d, t, it in _iter_decisions_items(payload):
        out[(str(d), str(t).upper())] = it
    return out


def _load_stock_item(*, daily_dir: Path, date_str: str, ticker: str) -> Optional[Dict[str, Any]]:
    fp = daily_dir / f"stock_features_{date_str}.json"
    if not fp.exists():
        return None
    try:
        payload = _read_json(fp)
    except Exception:
        return None

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None

    t = str(ticker).upper().strip()
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").upper().strip()
        if sym == t:
            return it
    return None


def _extract_news_title_phrase(news_contexts: List[str]) -> str:
    for ctx in news_contexts:
        s = str(ctx)
        for line in s.splitlines():
            if line.strip().lower().startswith("title:"):
                return line.split(":", 1)[1].strip()[:80]
    return ""


def _synthetic_decision_json(*, ticker: str, decision: str, news_contexts: List[str], role_hint: str) -> str:
    d = str(decision).strip().upper()
    if not d:
        d = "HOLD"

    quote = _extract_news_title_phrase(news_contexts)

    b1 = f"1. Decision: {d}."
    if quote:
        b2 = f"2. Evidence: \"{quote}\"."
    else:
        b2 = "2. Evidence: news/technical context considered."

    if d == "BUY":
        b3 = "3. Plan: build exposure with risk control over the next 5 days."
    elif d == "SELL":
        b3 = "3. Plan: reduce exposure to avoid drawdown risk."
    elif d == "CLEAR":
        b3 = "3. Plan: stay in cash until signal improves."
    else:
        b3 = "3. Plan: hold and wait for clearer signal."

    obj = {
        "decision": d,
        "ticker": str(ticker).upper().strip(),
        "analysis": str(role_hint),
        "reasoning_trace": [b1, b2, b3],
    }
    return json.dumps(obj, ensure_ascii=False)


def _pair_to_chosen_rejected(pair: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    winner = pair.get("winner") if isinstance(pair.get("winner"), dict) else {}
    loser = pair.get("loser") if isinstance(pair.get("loser"), dict) else {}
    return winner, loser


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 15.3: Build Analyst DPO dataset from alpha_pairs.json")
    p.add_argument(
        "--alpha-pairs",
        required=True,
        help="Path to alpha_pairs.json (from scripts/mining/mine_alpha_pairs.py)",
    )
    p.add_argument(
        "--run-dir",
        default="",
        help="Optional run dir containing baseline_fast/ and golden_strict/ (to pull real model outputs from decisions_*.json)",
    )
    p.add_argument("--daily-dir", default="data/daily", help="Directory containing stock_features_YYYY-MM-DD.json and signals_YYYY-MM-DD.json")
    p.add_argument("--signals-path", default="", help="Optional explicit signals.json path; if empty uses <daily-dir>/signals_DATE.json")
    p.add_argument("--out", default="data/dpo/alpha_pairs_dpo.jsonl")
    p.add_argument("--min-abs-impact", type=float, default=0.5)
    p.add_argument("--max-news-signals", type=int, default=3)
    p.add_argument("--only-dpo-candidate", action="store_true", default=True)
    p.add_argument("--include-non-dpo", dest="only_dpo_candidate", action="store_false")
    p.add_argument("--types", default="POSITIVE_SAMPLE,NEGATIVE_SAMPLE")
    p.add_argument("--target-expert", default="analyst", choices=["analyst", "any"])
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()

    alpha_pairs_path = Path(args.alpha_pairs)
    if not alpha_pairs_path.exists():
        raise SystemExit(f"alpha_pairs not found: {alpha_pairs_path}")

    daily_dir = Path(args.daily_dir)

    try:
        from scripts.run_trading_inference import build_stock_messages, load_daily_news_contexts
    except Exception as e:
        raise SystemExit(f"Failed to import prompt builders from scripts/run_trading_inference.py: {e}")

    allowed_types = {s.strip() for s in str(args.types).split(",") if s.strip()}

    decision_maps: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    if str(args.run_dir or "").strip():
        run_dir = Path(str(args.run_dir))
        base_dir = run_dir / "baseline_fast"
        gold_dir = run_dir / "golden_strict"
        base_decisions = _find_single_decisions_file(base_dir) if base_dir.exists() else None
        gold_decisions = _find_single_decisions_file(gold_dir) if gold_dir.exists() else None
        if base_decisions is not None:
            decision_maps["baseline_fast"] = _load_decision_map(base_decisions)
        if gold_decisions is not None:
            decision_maps["golden_strict"] = _load_decision_map(gold_decisions)

    obj = _read_json(alpha_pairs_path)
    rows = obj if isinstance(obj, list) else []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = {
        "type": 0,
        "not_dpo_candidate": 0,
        "no_pair": 0,
        "no_feature": 0,
        "no_actions": 0,
        "expert_filter": 0,
        "same_action": 0,
    }

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            if not isinstance(r, dict):
                continue

            tp = _safe_str(r.get("type"))
            if allowed_types and (tp not in allowed_types):
                skipped["type"] += 1
                continue

            if bool(args.only_dpo_candidate) and (not bool(r.get("dpo_candidate"))):
                skipped["not_dpo_candidate"] += 1
                continue

            date = _safe_str(r.get("date"))
            ticker = _safe_str(r.get("ticker")).upper()
            if not date or not ticker:
                continue

            pair = r.get("pair") if isinstance(r.get("pair"), dict) else None
            if not isinstance(pair, dict):
                skipped["no_pair"] += 1
                continue

            winner, loser = _pair_to_chosen_rejected(pair)

            w_action = _safe_str(winner.get("action")).upper()
            l_action = _safe_str(loser.get("action")).upper()
            if not w_action or not l_action:
                skipped["no_actions"] += 1
                continue

            if w_action == l_action:
                skipped["same_action"] += 1
                continue

            if str(args.target_expert) == "analyst":
                w_exp = _safe_str(winner.get("expert")).lower()
                l_exp = _safe_str(loser.get("expert")).lower()
                if (w_exp != "analyst") and (l_exp != "analyst"):
                    skipped["expert_filter"] += 1
                    continue

            stock_item = _load_stock_item(daily_dir=daily_dir, date_str=date, ticker=ticker)
            if stock_item is None:
                skipped["no_feature"] += 1
                continue

            news_contexts = load_daily_news_contexts(
                daily_dir=daily_dir,
                date_str=str(date),
                signals_path=str(args.signals_path),
                min_abs_impact=float(args.min_abs_impact),
                max_signals=int(args.max_news_signals),
                ticker=str(ticker),
            )

            # prompt: keep as messages for tokenizer chat_template
            messages = build_stock_messages(str(ticker), str(date), stock_item, news_contexts, allow_clear=True)

            def _maybe_real_output(system: str, want_action: str) -> Optional[str]:
                mp = decision_maps.get(str(system))
                if not isinstance(mp, dict):
                    return None
                it = mp.get((str(date), str(ticker).upper()))
                if not isinstance(it, dict):
                    return None
                got = _get_decision_from_item(it)
                if str(got).upper() != str(want_action).upper():
                    return None
                return _stringify_model_output(it)

            chosen = _maybe_real_output(str(winner.get("system") or ""), w_action)
            rejected = _maybe_real_output(str(loser.get("system") or ""), l_action)

            if not chosen:
                chosen = _synthetic_decision_json(
                    ticker=ticker,
                    decision=w_action,
                    news_contexts=news_contexts,
                    role_hint=f"Preferred action from alpha_pairs ({tp}).",
                )
            if not rejected:
                rejected = _synthetic_decision_json(
                    ticker=ticker,
                    decision=l_action,
                    news_contexts=news_contexts,
                    role_hint=f"Non-preferred action from alpha_pairs ({tp}).",
                )

            row = {
                "prompt": messages,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "date": date,
                    "ticker": ticker,
                    "type": tp,
                    "diff_pnl": _safe_float(r.get("diff_pnl"), 0.0),
                    "winner": winner,
                    "loser": loser,
                    "alpha_pairs": str(alpha_pairs_path).replace("\\", "/"),
                },
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if int(args.max_rows) > 0 and written >= int(args.max_rows):
                break

    print(
        json.dumps(
            {
                "out": str(out_path).replace("\\", "/"),
                "written": written,
                "skipped": skipped,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
