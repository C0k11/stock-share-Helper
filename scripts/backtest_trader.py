#!/usr/bin/env python

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.llm_tools import extract_json_text, repair_and_parse_json


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def list_feature_days(daily_dir: Path, start_date: str, end_date: str) -> List[str]:
    out: List[str] = []
    for fp in sorted(daily_dir.glob("stock_features_????-??-??.json")):
        name = fp.name
        day = name[len("stock_features_") : len("stock_features_") + 10]
        if start_date <= day <= end_date:
            out.append(day)
    return out


def load_stock_features(daily_dir: Path, day: str) -> Optional[Dict[str, Any]]:
    fp = daily_dir / f"stock_features_{day}.json"
    if not fp.exists():
        return None
    try:
        payload = _read_json(fp)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def iter_feature_items(payload: Any, tickers: Optional[set[str]]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(payload, dict):
        return out

    items = payload.get("items")
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        sym = _safe_str(it.get("symbol") or it.get("ticker"))
        if not sym:
            continue
        sym_u = sym.upper()
        if tickers is not None and sym_u not in tickers:
            continue
        out.append((sym_u, it))

    return out


def load_price_df(raw_dir: Path, ticker: str):
    import pandas as pd

    fp = raw_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None

    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if "close" not in df.columns:
        return None

    df = df.dropna(subset=["close"])
    df = df.sort_index()
    return df


def get_forward_return(df, day: str, horizon: int) -> Optional[float]:
    if df is None or df.empty:
        return None

    try:
        import datetime as dt

        target = dt.datetime.strptime(str(day), "%Y-%m-%d").date()
    except Exception:
        return None

    try:
        idxs = [i for i, d in enumerate(df.index.date) if d == target]
    except Exception:
        idxs = []

    if not idxs:
        return None

    i = int(idxs[0])
    j = i + int(horizon)
    if j >= len(df):
        return None

    p0 = _safe_float(df["close"].iloc[i], default=0.0)
    p1 = _safe_float(df["close"].iloc[j], default=0.0)
    if p0 <= 0:
        return None
    return (p1 - p0) / p0


def build_news_context_from_signal_item(it: Dict[str, Any]) -> Optional[str]:
    sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
    if sig is None:
        return None

    et = _safe_str(sig.get("event_type"))
    if not et:
        return None

    sent = _safe_str(sig.get("sentiment"))
    impact_eq = sig.get("impact_equity")
    impact_bond = sig.get("impact_bond")
    impact_gold = sig.get("impact_gold")
    summary = _safe_str(sig.get("summary"))

    if (summary == "") and (_safe_float(impact_eq) == 0.0) and (_safe_float(impact_bond) == 0.0) and (_safe_float(impact_gold) == 0.0):
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
    min_abs_impact: float,
    max_signals: int,
) -> List[str]:
    fp = daily_dir / f"signals_{date_str}.json"
    if not fp.exists():
        return []

    try:
        items = _read_json(fp)
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
        impact = _safe_float(sig.get("impact_equity"))
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


SYSTEM_PROMPT_STOCK_STRICT_JSON = """You are a strictly compliant trading signal generator.
You must analyze the input market data and output a JSON object containing the trading decision.
The decision logic is based on maximizing T+5 returns.

Response Format (STRICT JSON ONLY, NO MARKDOWN, NO PROSE):
{"decision": "BUY" | "SELL" | "HOLD", "ticker": "SYMBOL", "analysis": "Brief reason < 30 words"}
"""


def build_stock_messages(symbol: str, date_str: str, feats: Dict[str, Any], news_contexts: Optional[List[str]]) -> List[Dict[str, str]]:
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


def parse_decision(raw_text: str, fallback_ticker: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return None, None, "empty"

    try:
        raw_json = extract_json_text(raw_text)
        if raw_json is None:
            raise ValueError("no json found")
        obj = repair_and_parse_json(raw_json)
        if not isinstance(obj, dict):
            raise ValueError("json is not object")
        decision = _safe_str(obj.get("decision")).upper()
        if decision not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("invalid decision")
        if not _safe_str(obj.get("ticker")):
            obj["ticker"] = fallback_ticker
        if "analysis" not in obj:
            obj["analysis"] = ""
        return decision, obj, ""
    except Exception as e:
        return None, None, str(e)


@dataclass
class Strategy:
    name: str
    adapter: str
    use_news: bool


def load_base_model(base_model: str, load_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = str(base_model).replace("\\", "/")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
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

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.eval()
    return model, tokenizer


def attach_adapters(base_model, adapters: Dict[str, str]):
    from peft import PeftModel

    names = list(adapters.keys())
    if not names:
        raise ValueError("no adapters")

    first = names[0]
    model = PeftModel.from_pretrained(base_model, adapters[first], adapter_name=first)

    for name in names[1:]:
        model.load_adapter(adapters[name], adapter_name=name, is_trainable=False)

    model.eval()
    return model


def generate_one(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int):
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

    gen_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, out)]
    raw_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return raw_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly backtest for Stock Trader (LoRA swap on a single base model)")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--tickers", default="AAPL,NVDA,TSLA,MSFT,GOOGL")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated strategy names to run: v1_tech,v1_1_news,v1_1_ablation. Empty means run all.",
    )
    parser.add_argument("--daily-dir", default="data/daily")
    parser.add_argument("--raw-dir", default="data/raw")

    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    parser.set_defaults(load_4bit=True)

    parser.add_argument("--lora-v1", default="models/trader_stock_v1_tech_only/lora_weights")
    parser.add_argument("--lora-v1-1", default="models/trader_stock_v1_1_tech_plus_news/lora_weights")

    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--min-news-abs-impact", type=float, default=0.5)
    parser.add_argument("--max-news-signals", type=int, default=3)

    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--out", default="data/backtest/report_2025_12.json")

    args = parser.parse_args()

    daily_dir = Path(args.daily_dir)
    raw_dir = Path(args.raw_dir)

    tickers = [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]
    ticker_set = set(tickers) if tickers else None

    days = list_feature_days(daily_dir, str(args.start_date), str(args.end_date))
    if not days:
        raise SystemExit("No stock_features_YYYY-MM-DD.json found in the specified range")

    all_strategies = [
        Strategy(name="v1_tech", adapter=str(args.lora_v1), use_news=False),
        Strategy(name="v1_1_news", adapter=str(args.lora_v1_1), use_news=True),
        Strategy(name="v1_1_ablation", adapter=str(args.lora_v1_1), use_news=False),
    ]

    allowed = {
        "v1_tech",
        "v1_1_news",
        "v1_1_ablation",
    }
    requested = [m.strip() for m in str(args.models or "").split(",") if m.strip()]
    if requested:
        unknown = sorted([m for m in requested if m not in allowed])
        if unknown:
            raise SystemExit(f"Unknown --models entries: {unknown}. Allowed: {sorted(list(allowed))}")
        strategies = [s for s in all_strategies if s.name in set(requested)]
    else:
        strategies = all_strategies

    need_v1 = any(s.name == "v1_tech" for s in strategies)
    need_v1_1 = any(s.name in {"v1_1_news", "v1_1_ablation"} for s in strategies)
    adapter_map: Dict[str, str] = {}
    if need_v1:
        adapter_map["v1_tech"] = str(args.lora_v1)
    if need_v1_1:
        adapter_map["v1_1"] = str(args.lora_v1_1)

    base_model, tokenizer = load_base_model(str(args.base_model), bool(args.load_4bit))
    model = attach_adapters(base_model, adapter_map)

    price_dfs: Dict[str, Any] = {}
    for t in tickers:
        df = load_price_df(raw_dir, t)
        if df is not None:
            price_dfs[t] = df

    report: Dict[str, Any] = {
        "range": {"start": str(args.start_date), "end": str(args.end_date)},
        "daily_dir": str(daily_dir),
        "raw_dir": str(raw_dir),
        "tickers": tickers,
        "horizon_days": int(args.horizon_days),
        "strategies": {},
    }

    for strat in strategies:
        adapter_name = "v1_tech" if strat.name == "v1_tech" else "v1_1"
        model.set_adapter(adapter_name)

        trades: List[Dict[str, Any]] = []
        total = 0
        wins = 0
        sum_ret = 0.0
        sum_ret_buy_sell = 0.0
        cnt_buy_sell = 0

        for day in days:
            payload = load_stock_features(daily_dir, day)
            if payload is None:
                continue

            news_ctx: List[str] = []
            if strat.use_news:
                news_ctx = load_daily_news_contexts(
                    daily_dir=daily_dir,
                    date_str=day,
                    min_abs_impact=float(args.min_news_abs_impact),
                    max_signals=int(args.max_news_signals),
                )

            for sym, it in iter_feature_items(payload, ticker_set):
                raw = generate_one(
                    model,
                    tokenizer,
                    build_stock_messages(sym, day, it, news_ctx),
                    int(args.max_new_tokens),
                )

                decision, parsed, parse_error = parse_decision(raw, sym)
                if decision is None:
                    decision = "HOLD"

                fwd = None
                df = price_dfs.get(sym)
                if df is not None:
                    fwd = get_forward_return(df, day, int(args.horizon_days))

                realized = None
                if fwd is not None:
                    if decision == "BUY":
                        realized = float(fwd)
                    elif decision == "SELL":
                        realized = float(-fwd)
                    else:
                        realized = 0.0

                rec = {
                    "date": day,
                    "ticker": sym,
                    "decision": decision,
                    "parsed": parsed,
                    "parse_error": parse_error,
                    "forward_return": fwd,
                    "realized_return": realized,
                    "has_news_context": bool(news_ctx),
                    "news_context_count": int(len(news_ctx)),
                }
                trades.append(rec)

                if realized is not None:
                    total += 1
                    sum_ret += float(realized)
                    if float(realized) > 0:
                        wins += 1

                if realized is not None and decision in {"BUY", "SELL"}:
                    cnt_buy_sell += 1
                    sum_ret_buy_sell += float(realized)

        metrics = {
            "samples": int(len(trades)),
            "scored": int(total),
            "win_rate": (float(wins) / float(total)) if total > 0 else 0.0,
            "cumulative_return": float(sum_ret),
            "buy_sell_trades": int(cnt_buy_sell),
            "buy_sell_cumulative_return": float(sum_ret_buy_sell),
        }

        report["strategies"][strat.name] = {
            "adapter": str(strat.adapter),
            "use_news": bool(strat.use_news),
            "metrics": metrics,
            "trades": trades,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
