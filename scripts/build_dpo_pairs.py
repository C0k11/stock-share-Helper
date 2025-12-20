#!/usr/bin/env python

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@dataclass
class DecisionItem:
    date: str
    ticker: str
    item: Dict[str, Any]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_threshold(x: float) -> float:
    x = float(x)
    if abs(x) > 1.0:
        return abs(x) / 100.0
    return abs(x)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_decisions(payload: Dict[str, Any]) -> Iterable[DecisionItem]:
    if not isinstance(payload, dict):
        return

    # Multi-day format
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

    # Single-day format
    date_str = str(payload.get("date") or "").strip()
    items = payload.get("items")
    if date_str and isinstance(items, dict):
        for ticker, it in items.items():
            if isinstance(it, dict):
                yield DecisionItem(date=str(date_str), ticker=str(ticker).upper(), item=it)


class FeatureCache:
    def __init__(self, daily_dir: Path) -> None:
        self.daily_dir = daily_dir
        self._cache: Dict[str, Dict[str, Any]] = {}

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
                        sym = str(it.get("symbol") or "").upper()
                        if sym:
                            m[sym] = it
                self._cache[date_str] = m
        return self._cache[date_str].get(ticker)


def _extract_forward_return_5d(stock_item: Dict[str, Any]) -> Optional[float]:
    tech = stock_item.get("technical") if isinstance(stock_item.get("technical"), dict) else {}
    v = tech.get("return_5d")
    if v is None:
        return None
    return float(_to_float(v))


def _extract_news_title_phrase(news_contexts: List[str]) -> str:
    for ctx in news_contexts:
        s = str(ctx)
        m = None
        # prefer explicit Title: lines
        for line in s.splitlines():
            if line.strip().lower().startswith("title:"):
                m = line.split(":", 1)[1].strip()
                break
        if m:
            # keep a short quote fragment
            return m[:60]
    return ""


def _synthetic_clear_json(*, ticker: str, news_contexts: List[str], variant: str) -> str:
    # variant: punish_wrong_buy | miss_uptrend
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
    # fallback: try to parse raw JSON
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
    # Expand allowed decision set for DPO counterfactuals (CLEAR for cash)
    content = content.replace('"decision": "BUY" | "SELL" | "HOLD"', '"decision": "BUY" | "SELL" | "HOLD" | "CLEAR"')
    if content == str(m0.get("content") or ""):
        return messages

    out = list(messages)
    out[0] = dict(m0)
    out[0]["content"] = content
    return out


def _extract_router_meta(it: Dict[str, Any]) -> Dict[str, Any]:
    router = it.get("router") if isinstance(it.get("router"), dict) else {}
    return dict(router)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from trading decisions")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Decision JSON files (single-day or multi-day). e.g. data/daily/moe_race_dec2025_loose.json",
    )
    parser.add_argument("--daily-dir", default="data/daily", help="Directory containing stock_features_YYYY-MM-DD.json and signals_YYYY-MM-DD.json")
    parser.add_argument("--out", default="data/dpo/pairs_h5_x002.jsonl")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--x", type=float, default=0.02, help="Return threshold (e.g. 0.02 for 2%%). If >1, treated as percent points (e.g. 2 for 2%%).")
    parser.add_argument("--target-expert", default="analyst", choices=["analyst", "scalper", "any"])
    parser.add_argument("--min-abs-impact", type=float, default=0.5)
    parser.add_argument("--max-news-signals", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=0, help="If >0, stop after writing N pairs")
    parser.add_argument("--only-buy", action="store_true", default=True)
    parser.add_argument("--include-nonbuy", dest="only_buy", action="store_false")
    args = parser.parse_args()

    horizon = int(args.horizon)
    if horizon != 5:
        raise SystemExit("MVP only supports horizon=5 (uses technical.return_5d)")

    thr = _to_threshold(float(args.x))
    daily_dir = Path(args.daily_dir)

    try:
        from scripts.run_trading_inference import build_stock_messages, load_daily_news_contexts
    except Exception as e:
        raise SystemExit(f"Failed to import prompt builders from scripts/run_trading_inference.py: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fc = FeatureCache(daily_dir)

    written = 0
    n_total = 0
    n_skip_no_feat = 0
    n_skip_expert = 0
    n_skip_not_buy = 0
    n_skip_mid = 0
    n_pairs_pos = 0
    n_pairs_neg = 0

    def should_keep_expert(it: Dict[str, Any]) -> bool:
        te = str(args.target_expert)
        if te == "any":
            return True
        if _infer_expert(it) == te:
            return True
        # Include planner-quarantined cases: router wanted analyst but planner forced scalper
        if te == "analyst" and _infer_expert_before_planner_gate(it) == "analyst":
            return True
        return False

    with out_path.open("w", encoding="utf-8") as f:
        for in_fp in args.inputs:
            p = Path(in_fp)
            if not p.exists():
                raise SystemExit(f"Input not found: {p}")
            payload = _read_json(p)
            for d in _iter_decisions(payload):
                n_total += 1

                it = d.item
                if not should_keep_expert(it):
                    n_skip_expert += 1
                    continue

                model_decision = _get_model_decision(it)
                if args.only_buy and model_decision != "BUY":
                    n_skip_not_buy += 1
                    continue

                stock_item = fc.get_stock_item(d.date, d.ticker)
                if stock_item is None:
                    n_skip_no_feat += 1
                    continue

                fr = _extract_forward_return_5d(stock_item)
                if fr is None:
                    n_skip_no_feat += 1
                    continue

                if (-thr < fr < thr):
                    n_skip_mid += 1
                    continue

                news_contexts = load_daily_news_contexts(
                    daily_dir=daily_dir,
                    date_str=str(d.date),
                    signals_path="",
                    min_abs_impact=float(args.min_abs_impact),
                    max_signals=int(args.max_news_signals),
                    ticker=str(d.ticker),
                )

                # prompt: keep as messages list for downstream DPO / chat-template consumers
                messages = build_stock_messages(str(d.ticker), str(d.date), stock_item, news_contexts)
                messages = _patch_system_prompt_allow_clear(messages)

                original = _stringify_original_output(it)

                if fr <= -thr:
                    chosen = _synthetic_clear_json(ticker=d.ticker, news_contexts=news_contexts, variant="punish_wrong_buy")
                    rejected = original
                    n_pairs_neg += 1
                else:
                    chosen = original
                    rejected = _synthetic_clear_json(ticker=d.ticker, news_contexts=news_contexts, variant="miss_uptrend")
                    n_pairs_pos += 1

                row = {
                    "prompt": messages,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "date": d.date,
                        "ticker": d.ticker,
                        "horizon": horizon,
                        "x": thr,
                        "forward_return": fr,
                        "model_decision": model_decision,
                        "expert": _infer_expert(it),
                        "router": _extract_router_meta(it),
                        "source_file": str(p).replace("\\", "/"),
                    },
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                if int(args.max_pairs) > 0 and written >= int(args.max_pairs):
                    break
            if int(args.max_pairs) > 0 and written >= int(args.max_pairs):
                break

    print(
        json.dumps(
            {
                "out": str(out_path).replace("\\", "/"),
                "pairs_written": written,
                "n_total_items": n_total,
                "n_pairs_pos": n_pairs_pos,
                "n_pairs_neg": n_pairs_neg,
                "skips": {
                    "no_feature_or_return": n_skip_no_feat,
                    "expert_filter": n_skip_expert,
                    "not_buy": n_skip_not_buy,
                    "mid_band": n_skip_mid,
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
