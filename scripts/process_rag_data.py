#!/usr/bin/env python

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.risk.gate import RiskGate
from scripts.run_daily_inference import post_process_cn_signals, prepare_signals_for_save


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_pct(x: Any) -> float:
    v = _to_float(x)
    if abs(v) <= 2.5:
        return v * 100.0
    return v


def extract_features(etf_item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(etf_item, dict):
        return {}

    feats = etf_item.get("features")
    base = feats if isinstance(feats, dict) else etf_item

    tech = base.get("technical") if isinstance(base.get("technical"), dict) else {}

    change_5d = base.get("change_5d_pct")
    if change_5d is None:
        change_5d = base.get("return_5d")
    if change_5d is None:
        change_5d = tech.get("return_5d")

    vol = base.get("volatility_ann_pct")
    if vol is None:
        vol = base.get("volatility_20d")
    if vol is None:
        vol = tech.get("volatility_20d")

    dd = base.get("drawdown_20d_pct")
    if dd is None:
        dd = base.get("max_drawdown_20d")
    if dd is None:
        dd = tech.get("max_drawdown_20d")
    if dd is None:
        dd = base.get("drawdown")
    if dd is None:
        dd = tech.get("drawdown")

    merged = dict(base)
    merged["change_5d_pct"] = _to_pct(change_5d)
    merged["volatility_ann_pct"] = _to_pct(vol)
    merged["drawdown_20d_pct"] = _to_pct(dd)
    return merged


def extract_news_signals(rw: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(rw, dict):
        return out

    us_top = rw.get("us_top_events") if isinstance(rw.get("us_top_events"), list) else []
    for it in us_top:
        if not isinstance(it, dict):
            continue
        et = str(it.get("event_type") or "").strip()
        if not et:
            continue
        out.append({"event_type": et, "impact_equity": it.get("impact_equity")})

    cn_rc = rw.get("cn_regulation_crackdown") if isinstance(rw.get("cn_regulation_crackdown"), dict) else {}
    cn_top = cn_rc.get("top") if isinstance(cn_rc.get("top"), list) else []
    for it in cn_top:
        if not isinstance(it, dict):
            continue
        et = str(it.get("event_type") or "").strip()
        if not et:
            continue
        out.append({"event_type": et, "impact_equity": it.get("impact_equity")})

    return out


def _load_signals(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in data:
        if isinstance(it, dict):
            out.append(it)
    return out


def _build_risk_watch_from_signals(
    *,
    signals: List[Dict[str, Any]],
    signals_path: Path,
    market_mode: str,
    top_k: int,
) -> Dict[str, Any]:
    market_mode = (market_mode or "CN").strip().upper()
    if market_mode not in {"CN", "US", "BOTH", "NONE"}:
        market_mode = "CN"

    top_k = int(top_k)
    if top_k <= 0:
        top_k = 5

    def compact(it: Dict[str, Any]) -> Dict[str, Any]:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        return {
            "title": str(it.get("title") or "")[:120],
            "source": str(it.get("source") or ""),
            "event_type": str(sig.get("event_type") or ""),
            "impact_equity": sig.get("impact_equity"),
            "impact_bond": sig.get("impact_bond"),
            "impact_gold": sig.get("impact_gold"),
            "summary": str(sig.get("summary") or "")[:220],
        }

    def impact_score(it: Dict[str, Any]) -> float:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}

        def _to_num(x: Any) -> float:
            try:
                return abs(float(x))
            except Exception:
                return 0.0

        return _to_num(sig.get("impact_equity")) + _to_num(sig.get("impact_bond")) + _to_num(sig.get("impact_gold"))

    out: Dict[str, Any] = {
        "signals_path": str(signals_path),
        "available": True,
        "mode": market_mode,
    }

    if market_mode == "NONE":
        return out

    if market_mode in {"CN", "BOTH"}:
        cn_ok = [it for it in signals if str(it.get("market") or "").upper() == "CN" and it.get("parse_ok")]

        rc: List[Dict[str, Any]] = []
        for it in cn_ok:
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            if str(sig.get("event_type") or "").strip() == "regulation_crackdown":
                rc.append(it)

        top_cn = [compact(x) for x in rc[:top_k]]
        total_ok = len(cn_ok)
        n = len(rc)
        out["cn_parse_ok"] = total_ok
        out["cn_regulation_crackdown"] = {
            "count": n,
            "share": (n / total_ok) if total_ok > 0 else 0.0,
            "top": top_cn,
        }

    if market_mode in {"US", "BOTH"}:
        us_ok = [it for it in signals if str(it.get("market") or "").upper() == "US" and it.get("parse_ok")]
        us_ok_sorted = sorted(us_ok, key=impact_score, reverse=True)
        out["us_parse_ok"] = len(us_ok)
        out["us_top_events"] = [compact(x) for x in us_ok_sorted[:top_k]]

    return out


def _maybe_rebuild_risk_watch(
    *,
    entry: Dict[str, Any],
    daily_dir: Path,
    market_mode: str,
    top_k: int,
    cn_hype_cap: int,
    debug: bool = False,
    debug_state: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    date_str = str(entry.get("date") or "").strip()
    if not date_str:
        return {}

    sig_path = daily_dir / f"signals_{date_str}.json"
    if not sig_path.exists():
        return {}

    signals = _load_signals(sig_path)
    if not signals:
        return {}

    if debug and isinstance(debug_state, dict):
        try:
            debug_state["rebuild_seen"] = int(debug_state.get("rebuild_seen", 0)) + 1
        except Exception:
            debug_state["rebuild_seen"] = 1

    before_hype = 0
    try:
        for it in signals:
            if not isinstance(it, dict):
                continue
            if str(it.get("market") or "").strip().upper() != "CN":
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            if str(sig.get("event_type") or "").strip() == "concept_hype":
                before_hype += 1
    except Exception:
        before_hype = 0

    cleaned = [post_process_cn_signals(dict(it)) for it in signals]
    after_clean_hype = 0
    try:
        for it in cleaned:
            if not isinstance(it, dict):
                continue
            if str(it.get("market") or "").strip().upper() != "CN":
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            if str(sig.get("event_type") or "").strip() == "concept_hype":
                after_clean_hype += 1
    except Exception:
        after_clean_hype = 0

    cleaned = prepare_signals_for_save(cleaned, int(cn_hype_cap))
    after_cap_hype = 0
    try:
        for it in cleaned:
            if not isinstance(it, dict):
                continue
            if str(it.get("market") or "").strip().upper() != "CN":
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            if str(sig.get("event_type") or "").strip() == "concept_hype":
                after_cap_hype += 1
    except Exception:
        after_cap_hype = 0

    if debug and isinstance(debug_state, dict):
        max_print = int(debug_state.get("debug_print_max", 20) or 20)
        printed = int(debug_state.get("rebuild_printed", 0) or 0)
        if printed < max_print:
            print(
                f"[Rebuild] date={date_str} signals={len(signals)} cn_concept_hype {before_hype}->{after_clean_hype}->{after_cap_hype}"
            )
            debug_state["rebuild_printed"] = printed + 1

    return _build_risk_watch_from_signals(
        signals=cleaned,
        signals_path=sig_path,
        market_mode=market_mode,
        top_k=int(top_k),
    )


def _safe_import_prompt_builder():
    try:
        from scripts.generate_etf_teacher_dataset import build_teacher_messages, validate_label

        return build_teacher_messages, validate_label
    except Exception as e:
        raise SystemExit(f"Failed to import build_teacher_messages/validate_label: {e}")


def build_prompt_for_entry(
    *,
    entry: Dict[str, Any],
    market_rag: Any,
    risk_gate: RiskGate,
    daily_dir: Optional[Path] = None,
    risk_watch_market: str = "BOTH",
    risk_watch_top: int = 3,
    cn_hype_cap: int = 30,
    debug: bool = False,
    debug_state: Optional[Dict[str, int]] = None,
) -> Optional[List[Dict[str, str]]]:
    build_teacher_messages, _ = _safe_import_prompt_builder()

    symbol = str(entry.get("symbol") or "")
    date_str = str(entry.get("date") or "")

    inp = entry.get("input") if isinstance(entry.get("input"), dict) else {}
    etf_item = inp.get("etf_features") if isinstance(inp.get("etf_features"), dict) else {}
    risk_watch_orig = inp.get("risk_watch") if isinstance(inp.get("risk_watch"), dict) else {}

    risk_watch = {}
    if daily_dir is not None:
        rebuilt = _maybe_rebuild_risk_watch(
            entry=entry,
            daily_dir=daily_dir,
            market_mode=str(risk_watch_market),
            top_k=int(risk_watch_top),
            cn_hype_cap=int(cn_hype_cap),
            debug=bool(debug),
            debug_state=debug_state,
        )
        if isinstance(rebuilt, dict) and rebuilt:
            risk_watch = rebuilt

    if not risk_watch:
        if debug and isinstance(debug_state, dict):
            max_print = int(debug_state.get("debug_print_max", 20) or 20)
            printed = int(debug_state.get("fallback_printed", 0) or 0)
            if printed < max_print:
                date_str_dbg = str(entry.get("date") or "").strip()
                sig_path_dbg = (daily_dir / f"signals_{date_str_dbg}.json") if daily_dir is not None else None
                exists_dbg = sig_path_dbg.exists() if isinstance(sig_path_dbg, Path) else False
                print(f"[Fallback] date={date_str_dbg} signals_file_exists={exists_dbg}")
                debug_state["fallback_printed"] = printed + 1

        risk_watch = risk_watch_orig

    feats = extract_features(etf_item)
    news_signals = extract_news_signals(risk_watch)

    similar_days: List[Dict[str, Any]] = []
    if market_rag is not None:
        try:
            similar_days = market_rag.retrieve(feats, k=3, ticker=symbol, exclude_date=date_str)
        except Exception:
            similar_days = []

    try:
        _, _, risk_trace = risk_gate.adjudicate(feats, news_signals, "BUY", 1.0)
    except Exception:
        risk_trace = []

    vi = int(entry.get("variant_index") or 0)
    teacher = entry.get("teacher") if isinstance(entry.get("teacher"), dict) else {}
    include_cot = bool(teacher.get("include_cot"))

    messages = build_teacher_messages(
        symbol=symbol,
        date_str=date_str,
        etf_item=etf_item,
        risk_watch=risk_watch,
        history=similar_days,
        risk_constraints=risk_trace,
        include_cot=include_cot,
        variant_index=vi,
    )

    if not isinstance(messages, list) or len(messages) < 2:
        return None

    return messages


def to_chatml_sample(messages: List[Dict[str, str]], output_obj: Dict[str, Any]) -> Dict[str, Any]:
    assistant_content = json.dumps(output_obj, ensure_ascii=False)
    conv = [
        {"role": "system", "content": str(messages[0].get("content") or "")},
        {"role": "user", "content": str(messages[1].get("content") or "")},
        {"role": "assistant", "content": assistant_content},
    ]
    return {"conversations": conv, "messages": conv}


def is_valid_entry(entry: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, "not_object"

    if not str(entry.get("id") or "").strip():
        return False, "missing_id"
    if not str(entry.get("date") or "").strip():
        return False, "missing_date"
    if not str(entry.get("symbol") or "").strip():
        return False, "missing_symbol"

    inp = entry.get("input")
    if not isinstance(inp, dict):
        return False, "missing_input"
    if not isinstance(inp.get("etf_features"), dict):
        return False, "missing_input_etf_features"
    if not isinstance(inp.get("risk_watch"), dict):
        return False, "missing_input_risk_watch"

    out = entry.get("output")
    if not isinstance(out, dict) or not out:
        return False, "missing_output"

    teacher = entry.get("teacher") if isinstance(entry.get("teacher"), dict) else {}
    if str(teacher.get("error") or "").strip():
        return False, "teacher_error"

    _, validate_label = _safe_import_prompt_builder()
    missing, extra = validate_label(out)
    if missing or extra:
        return False, "schema_mismatch"

    return True, "ok"


def split_train_val(data: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    val_ratio = float(val_ratio)
    if val_ratio <= 0:
        return data, []
    if val_ratio >= 1:
        return [], data

    try:
        from sklearn.model_selection import train_test_split  # type: ignore

        train, val = train_test_split(data, test_size=val_ratio, random_state=int(seed), shuffle=True)
        return list(train), list(val)
    except Exception:
        rng = random.Random(int(seed))
        idx = list(range(len(data)))
        rng.shuffle(idx)
        cut = int(round(len(data) * (1.0 - val_ratio)))
        train_idx = idx[:cut]
        val_idx = idx[cut:]
        train = [data[i] for i in train_idx]
        val = [data[i] for i in val_idx]
        return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Process RAG-enhanced teacher JSONL into Qwen ChatML train/val JSON")
    parser.add_argument(
        "--input",
        type=str,
        default="data/finetune/teacher_etf/teacher_etf_rag_enhanced_25000.jsonl",
    )
    parser.add_argument("--daily-dir", type=str, default="data/daily")
    parser.add_argument("--out-dir", type=str, default="data/finetune/teacher_etf")
    parser.add_argument("--train-name", type=str, default="train_rag_final.json")
    parser.add_argument("--val-name", type=str, default="val_rag_final.json")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-lines", type=int, default=0)
    parser.add_argument("--risk-watch-market", default="BOTH")
    parser.add_argument("--risk-watch-top", type=int, default=3)
    parser.add_argument("--cn-hype-cap", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")

    risk_gate = RiskGate()

    market_rag = None
    try:
        from src.data.rag import MarketRAG

        market_rag = MarketRAG(data_dir=str(args.daily_dir))
    except Exception:
        market_rag = None

    total = 0
    kept = 0
    dropped = 0
    debug_state: Dict[str, int] = {"debug_print_max": 20, "rebuild_printed": 0, "fallback_printed": 0}

    samples: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if int(args.skip_lines) > 0 and i <= int(args.skip_lines):
                continue
            if int(args.max_rows) > 0 and kept >= int(args.max_rows):
                break

            if not line.strip():
                continue

            total += 1
            try:
                entry = json.loads(line)
            except Exception:
                dropped += 1
                continue

            ok, _reason = is_valid_entry(entry)
            if not ok:
                dropped += 1
                continue

            messages = build_prompt_for_entry(
                entry=entry,
                market_rag=market_rag,
                risk_gate=risk_gate,
                daily_dir=Path(str(args.daily_dir)),
                risk_watch_market=str(args.risk_watch_market),
                risk_watch_top=int(args.risk_watch_top),
                cn_hype_cap=int(args.cn_hype_cap),
                debug=bool(args.dry_run),
                debug_state=debug_state,
            )
            if not messages:
                dropped += 1
                continue

            output_obj = entry.get("output") if isinstance(entry.get("output"), dict) else {}
            if not output_obj:
                dropped += 1
                continue

            samples.append(to_chatml_sample(messages, output_obj))
            kept += 1

            if kept % 500 == 0:
                print(f"processed={total} kept={kept} dropped={dropped}")

    print(f"Total lines read: {total}")
    print(f"Valid samples: {kept}")
    print(f"Dropped: {dropped}")

    if bool(args.dry_run):
        print("[DRY_RUN] Done. No output files written.")
        return

    train_data, val_data = split_train_val(samples, float(args.val_ratio), int(args.seed))

    train_path = out_dir / str(args.train_name)
    val_path = out_dir / str(args.val_name)

    train_path.write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved Train({len(train_data)}) -> {train_path}")
    print(f"Saved Val({len(val_data)}) -> {val_path}")


if __name__ == "__main__":
    main()
