#!/usr/bin/env python

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.llm_tools import extract_json_text, repair_and_parse_json


SYSTEM_PROMPT_STOCK_STRICT_JSON = """You are a strictly compliant trading signal generator.
You must analyze the input market data and output a JSON object containing the trading decision.
The decision logic is based on maximizing T+5 returns.

Response Format (STRICT JSON ONLY, NO MARKDOWN, NO PROSE):
{"decision": "BUY" | "SELL" | "HOLD", "ticker": "SYMBOL", "analysis": "Brief reason < 30 words"}
"""


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


def build_news_context_from_signal_item(it: Dict[str, Any]) -> Optional[str]:
    sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
    if sig is None:
        return None

    et = str(sig.get("event_type") or "").strip()
    if not et:
        return None

    sent = str(sig.get("sentiment") or "").strip()
    impact_eq = sig.get("impact_equity")
    impact_bond = sig.get("impact_bond")
    impact_gold = sig.get("impact_gold")
    summary = str(sig.get("summary") or "").strip()

    if (summary == "") and (_to_float(impact_eq) == 0.0) and (_to_float(impact_bond) == 0.0) and (_to_float(impact_gold) == 0.0):
        return None

    lines = []
    lines.append("Market News Context:")
    lines.append(f"EventType: {et}")
    if sent:
        lines.append(f"Sentiment: {sent}")
    lines.append(f"ImpactEquity: {impact_eq}")
    lines.append(f"ImpactBond: {impact_bond}")
    lines.append(f"ImpactGold: {impact_gold}")
    if summary:
        lines.append(f"Summary: {summary}")

    return "\n".join(lines)


def load_daily_news_contexts(
    *,
    daily_dir: Path,
    date_str: str,
    signals_path: str,
    min_abs_impact: float,
    max_signals: int,
) -> List[str]:
    fp = Path(signals_path) if str(signals_path or "").strip() else (daily_dir / f"signals_{date_str}.json")
    if not fp.exists():
        return []

    try:
        items = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(items, list):
        return []

    candidates: List[Tuple[float, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if not it.get("parse_ok"):
            continue
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
        if sig is None:
            continue
        impact = _to_float(sig.get("impact_equity"))
        if abs(impact) < float(min_abs_impact):
            continue
        ctx = build_news_context_from_signal_item(it)
        if not ctx:
            continue
        candidates.append((abs(impact), ctx))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(max_signals))
    return [c for _score, c in candidates[:k]]


def prefix_news_contexts(user_text: str, contexts: List[str]) -> str:
    if not contexts:
        return user_text
    addon = "\n\n" + "\n\n".join(contexts) + "\n\n"
    return addon + user_text


def build_stock_messages(symbol: str, date_str: str, feats: Dict[str, Any], news_contexts: Optional[List[str]] = None) -> List[Dict[str, str]]:
    tech = feats.get("technical") if isinstance(feats.get("technical"), dict) else {}
    sig = feats.get("signal") if isinstance(feats.get("signal"), dict) else {}

    def gv(d: Dict[str, Any], k: str, default: Any = "") -> Any:
        return d.get(k, default) if isinstance(d, dict) else default

    user = (
        f"Ticker: {symbol}\n"
        f"Date: {date_str}\n"
        f"Close: {gv(tech, 'close', '')}\n"
        f"Price vs MA20: {gv(tech, 'price_vs_ma20', '')}\n"
        f"Price vs MA200: {gv(tech, 'price_vs_ma200', '')}\n"
        f"Trend alignment: {gv(tech, 'trend_alignment', '')}\n"
        f"Breakout 20d high: {gv(tech, 'breakout_20d_high', '')}\n"
        f"Breakdown 20d low: {gv(tech, 'breakdown_20d_low', '')}\n"
        f"Return 5d: {gv(tech, 'return_5d', '')}\n"
        f"Return 21d: {gv(tech, 'return_21d', '')}\n"
        f"Return 63d: {gv(tech, 'return_63d', '')}\n"
        f"Volatility 20d: {gv(tech, 'volatility_20d', '')}\n"
        f"Volume ratio: {gv(tech, 'vol_ratio', '')}\n"
        f"Drawdown: {gv(tech, 'drawdown', '')}\n"
        f"Max drawdown 20d: {gv(tech, 'max_drawdown_20d', '')}\n"
        f"Max drawdown 60d: {gv(tech, 'max_drawdown_60d', '')}\n"
        f"Composite signal: {gv(sig, 'composite', '')}\n\n"
        "Decide BUY/SELL/HOLD for the next 5 days."
    )

    if news_contexts:
        user = prefix_news_contexts(user, news_contexts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT_STOCK_STRICT_JSON},
        {"role": "user", "content": user},
    ]


def validate_stock_decision(obj: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    required = {"decision"}
    keys = set(obj.keys())
    missing = sorted(list(required - keys))
    extra = []
    decision = str(obj.get("decision") or "").strip().upper()
    if decision not in {"BUY", "SELL", "HOLD"}:
        missing = sorted(list(set(missing + ["decision(enum BUY/SELL/HOLD)"])))
    return missing, extra


def iter_feature_items(payload: Any) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []

    def push(symbol: Any, item: Any) -> None:
        if not symbol:
            return
        if not isinstance(item, dict):
            return
        out.append((str(symbol).strip(), item))

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                sym = it.get("symbol") or it.get("ticker")
                push(sym, it)
            return out

        for _k, v in payload.items():
            if isinstance(v, list):
                for it in v:
                    if not isinstance(it, dict):
                        continue
                    sym = it.get("symbol") or it.get("ticker")
                    push(sym, it)
            elif isinstance(v, dict):
                for sym, it in v.items():
                    if isinstance(it, dict):
                        push(sym, {"symbol": sym, **it})
        return out

    if isinstance(payload, list):
        for it in payload:
            if not isinstance(it, dict):
                continue
            sym = it.get("symbol") or it.get("ticker")
            push(sym, it)

    return out


def load_daily_payload(daily_dir: Path, date_str: str) -> Optional[Any]:
    fp = daily_dir / f"etf_features_{date_str}.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_model(base_model_id: str, adapter_path: str, load_4bit: bool) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = str(adapter_path or "").strip()

    adapter_p = Path(adapter_path)
    if adapter_path and adapter_p.exists() and adapter_p.is_dir():
        lw = adapter_p / "lora_weights"
        if lw.exists() and lw.is_dir():
            adapter_path = str(lw)

    tokenizer_src = adapter_path if adapter_path and Path(adapter_path, "tokenizer_config.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["low_cpu_mem_usage"] = True

    if load_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trading LoRA inference with MarketRAG + RiskGate")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--daily-dir", default="data/daily")
    parser.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter", default="", help="LoRA adapter path (output dir or lora_weights dir)")
    parser.add_argument("--model", dest="base", help="Alias of --base")
    parser.add_argument("--lora", dest="adapter", help="Alias of --adapter")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers for stock integration")
    parser.add_argument("--risk-watch-market", default="BOTH", help="CN|US|BOTH|NONE")
    parser.add_argument("--risk-watch-top", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    parser.set_defaults(load_4bit=True)
    parser.add_argument("--disable-news", action="store_true", default=False)
    parser.add_argument("--signals", dest="signals_path", default="", help="Override signals_YYYY-MM-DD.json path for stock news injection")
    parser.add_argument("--min-news-abs-impact", type=float, default=0.5)
    parser.add_argument("--max-news-signals", type=int, default=3)
    parser.add_argument("--out", default="data/decisions_inference.json")
    parser.add_argument("--output", dest="out", help="Alias of --out")
    args = parser.parse_args()

    from src.data.rag import MarketRAG
    from src.risk.gate import RiskGate
    from scripts.generate_etf_teacher_dataset import (
        build_risk_watch_summary,
        build_teacher_messages,
        validate_label,
    )

    daily_dir = Path(args.daily_dir)

    tickers = [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]

    stock_fp = daily_dir / f"stock_features_{args.date}.json"
    etf_fp = daily_dir / f"etf_features_{args.date}.json"

    use_stock = bool(tickers) and stock_fp.exists()

    if use_stock:
        payload = json.loads(stock_fp.read_text(encoding="utf-8"))
    else:
        payload = load_daily_payload(daily_dir, str(args.date))
        if payload is None:
            raise SystemExit(f"Missing daily features: {etf_fp}")

    items = iter_feature_items(payload)
    if not items:
        raise SystemExit("No symbols found in daily features payload")

    if use_stock:
        allow = set(tickers)
        items = [(sym, it) for (sym, it) in items if sym.upper() in allow]
        if not items:
            raise SystemExit(f"No requested tickers found in stock features payload: {tickers}")

    risk_watch = build_risk_watch_summary(
        daily_dir=daily_dir,
        date_str=str(args.date),
        market_mode=str(args.risk_watch_market),
        top_k=int(args.risk_watch_top),
    )

    rag = MarketRAG(data_dir=str(daily_dir))
    risk_gate = RiskGate()

    if bool(args.use_lora) and not str(args.adapter).strip():
        raise SystemExit("--use-lora requires --adapter/--lora")

    model, tokenizer = load_model(str(args.base), str(args.adapter), bool(args.load_4bit))

    decisions: Dict[str, Any] = {
        "date": str(args.date),
        "base": str(args.base),
        "adapter": str(args.adapter),
        "risk_watch": risk_watch,
        "items": {},
    }

    news_signals = extract_news_signals(risk_watch)

    stock_news_contexts: List[str] = []
    if use_stock and (not bool(args.disable_news)):
        stock_news_contexts = load_daily_news_contexts(
            daily_dir=daily_dir,
            date_str=str(args.date),
            signals_path=str(args.signals_path),
            min_abs_impact=float(args.min_news_abs_impact),
            max_signals=int(args.max_news_signals),
        )

    for symbol, etf_item in items:
        feats = extract_features(etf_item)

        similar_days: List[Dict[str, Any]] = []
        if not use_stock:
            try:
                similar_days = rag.retrieve(feats, k=3, ticker=symbol, exclude_date=str(args.date))
            except Exception:
                similar_days = []

        try:
            _, _, risk_trace = risk_gate.adjudicate(feats, news_signals, "BUY", 1.0)
        except Exception:
            risk_trace = []

        if use_stock:
            messages = build_stock_messages(str(symbol), str(args.date), etf_item, stock_news_contexts)
        else:
            messages = build_teacher_messages(
                symbol=str(symbol),
                date_str=str(args.date),
                etf_item=etf_item,
                risk_watch=risk_watch,
                history=similar_days,
                risk_constraints=risk_trace,
                include_cot=False,
                variant_index=0,
            )

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        import torch

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                do_sample=False,
            )

        gen_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        raw_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        parsed: Optional[Dict[str, Any]] = None
        parse_error = ""
        try:
            raw_json = extract_json_text(raw_text.strip())
            if raw_json is None:
                raise ValueError("no json found in model output")

            obj = repair_and_parse_json(raw_json)
            if not isinstance(obj, dict):
                raise ValueError("model output json is not object")

            parsed = obj
            if use_stock:
                if not str(parsed.get("ticker") or "").strip():
                    parsed["ticker"] = str(symbol)
                if "analysis" not in parsed:
                    parsed["analysis"] = ""
                missing, extra = validate_stock_decision(parsed)
            else:
                missing, extra = validate_label(parsed)
            if missing or extra:
                raise ValueError(f"schema mismatch missing={missing} extra={extra}")
        except Exception as e:
            parsed = None
            parse_error = str(e)

        final_action = "HOLD"
        final_pos = 0.0
        final_trace: List[str] = []
        if parsed is not None:
            if use_stock:
                proposed_action = str(parsed.get("decision") or "HOLD")
                decision = str(proposed_action).strip().upper()
                if decision == "BUY":
                    proposed_pos = 0.5
                else:
                    proposed_pos = 0.0
            else:
                label = parsed.get("label") if isinstance(parsed.get("label"), dict) else {}
                proposed_action = str(label.get("action") or "hold")
                proposed_pos = label.get("target_position", 0.0)
            try:
                final_action, final_pos, final_trace = risk_gate.adjudicate(
                    feats,
                    news_signals,
                    proposed_action,
                    proposed_pos,
                )
            except Exception as e:
                final_action = str(proposed_action).upper()
                try:
                    final_pos = float(proposed_pos)
                except Exception:
                    final_pos = 0.0
                final_trace = [f"[RISK] adjudicate error: {e}"]

        decisions["items"][symbol] = {
            "parsed": parsed,
            "parse_error": parse_error,
            "final": {"action": final_action, "target_position": final_pos, "trace": final_trace},
            "raw": raw_text,
        }

        if parsed is not None:
            print(f"{symbol}: {final_action} {final_pos}")
        else:
            print(f"{symbol}: PARSE_ERROR {parse_error}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
