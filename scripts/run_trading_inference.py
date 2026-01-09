#!/usr/bin/env python

import argparse
import math
from datetime import datetime, timedelta
import json
import re
import sys
import time
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.llm_tools import extract_json_text, repair_and_parse_json


def _infer_state_dir(out_path: Path) -> Path:
    p = Path(out_path)
    parents = [p.parent] + list(p.parents)
    for i, par in enumerate(parents):
        if str(par.name).lower() == "results":
            if i > 0:
                run_dir = parents[i - 1]
                if run_dir.exists() and run_dir.is_dir():
                    return run_dir
            break
    return p.parent


def _write_engine_state(state_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        fp = state_dir / "engine_state.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _append_engine_event(state_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        fp = state_dir / "events.jsonl"
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


SYSTEM_PROMPT_STOCK_STRICT_JSON = """You are a strictly compliant trading signal generator.
You must analyze the input market data and output a JSON object containing the trading decision.
The decision logic is based on maximizing T+5 returns.

Response Format (STRICT JSON ONLY, NO MARKDOWN, NO PROSE):
{
  "decision": "BUY" | "SELL" | "HOLD",
  "ticker": "SYMBOL",
  "analysis": "Brief reason < 30 words",
  "reasoning_trace": [
    "1. <short reason>",
    "2. <short reason>",
    "3. <short reason>"
  ]
}

Rules:
- reasoning_trace must contain exactly 3 short bullet points
- each bullet must be <= 25 words
- if any news context is provided, at least one bullet must explicitly cite it by quoting a short phrase from the news title
- if provided news context is irrelevant, explicitly say so in the trace
"""


SYSTEM_PROMPT_STOCK_FAST_JSON = """You are a strictly compliant trading signal generator.
You must analyze the input market data and output a JSON object containing the trading decision.
The decision logic is based on maximizing T+5 returns.

Response Format (STRICT JSON ONLY, NO MARKDOWN, NO PROSE):
{
  "decision": "BUY" | "SELL" | "HOLD",
  "ticker": "SYMBOL",
  "analysis": "Brief reason < 25 words"
}
"""


SYSTEM2_CRITIC_PROMPT_STOCK_JSON = """You are a strict trading decision critic.
You will be given a proposed decision and the market inputs.
Your job is to find flaws, missing risks, and overconfidence.

Response Format (STRICT JSON ONLY, NO MARKDOWN, NO PROSE):
{
  "accept": true | false,
  "suggested_decision": "BUY" | "SELL" | "HOLD" | "CLEAR",
  "reasons": ["<short>", "<short>", "<short>"]
}

Rules:
- reasons must contain exactly 3 short strings (<= 25 words each)
- suggested_decision must be one of BUY/SELL/HOLD/CLEAR
"""


SYSTEM2_JUDGE_PROMPT_STOCK_JSON = """You are a strict trading decision judge.
You will be given a proposed decision and a critic review.
Decide the final decision.

Response Format (STRICT JSON ONLY, NO MARKDOWN, NO PROSE):
{
  "final_decision": "BUY" | "SELL" | "HOLD" | "CLEAR",
  "rationale": "Brief reason < 25 words"
}

Rules:
- final_decision must be one of BUY/SELL/HOLD/CLEAR
"""


def _patch_prompt_allow_clear(prompt: str) -> str:
    p = str(prompt or "")
    p = p.replace('"decision": "BUY" | "SELL" | "HOLD"', '"decision": "BUY" | "SELL" | "HOLD" | "CLEAR"')
    return p


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _llm_generate(*, model: Any, tokenizer: Any, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float) -> str:
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    do_sample = float(temperature) > 0
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=bool(do_sample),
        )
    gen_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


def _try_parse_json_dict(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        raw_json = extract_json_text(str(raw_text or "").strip())
        if raw_json is None:
            return None, "no json found in model output"
        obj = repair_and_parse_json(raw_json)
        if not isinstance(obj, dict):
            return None, "model output json is not object"
        return obj, ""
    except Exception as e:
        return None, str(e)


def _build_system2_context_lines(*, symbol: str, date_str: str, feats: Dict[str, Any], news_contexts: List[str]) -> List[str]:
    tech = feats.get("technical") if isinstance(feats.get("technical"), dict) else {}
    sig = feats.get("signal") if isinstance(feats.get("signal"), dict) else {}

    def gv(d: Dict[str, Any], k: str, default: Any = "") -> Any:
        return d.get(k, default) if isinstance(d, dict) else default

    lines = [
        f"Ticker: {symbol}",
        f"Date: {date_str}",
        f"Close: {gv(tech, 'close', '')}",
        f"Price vs MA20: {gv(tech, 'price_vs_ma20', '')}",
        f"Price vs MA200: {gv(tech, 'price_vs_ma200', '')}",
        f"Trend alignment: {gv(tech, 'trend_alignment', '')}",
        f"Return 5d: {gv(tech, 'return_5d', '')}",
        f"Return 21d: {gv(tech, 'return_21d', '')}",
        f"Volatility 20d: {gv(tech, 'volatility_20d', '')}",
        f"Volume ratio: {gv(tech, 'vol_ratio', '')}",
        f"Max drawdown 20d: {gv(tech, 'max_drawdown_20d', '')}",
        f"Composite signal: {gv(sig, 'composite', '')}",
    ]

    ctxs = [str(x) for x in (news_contexts or []) if str(x).strip()]
    if ctxs:
        lines.append("")
        lines.append("News Context (may be empty):")
        for i, c in enumerate(ctxs[:3], start=1):
            lines.append(f"[{i}] {c}")
    return lines


def _build_system2_critic_messages(
    *,
    symbol: str,
    date_str: str,
    feats: Dict[str, Any],
    news_contexts: List[str],
    proposed_decision: str,
    proposed_analysis: str,
    allow_clear: bool,
) -> List[Dict[str, str]]:
    lines = _build_system2_context_lines(symbol=symbol, date_str=date_str, feats=feats, news_contexts=news_contexts)
    lines.extend(["", f"Proposed decision: {str(proposed_decision).strip().upper()}", f"Proposed analysis: {str(proposed_analysis or '').strip()}"])
    sys_prompt = SYSTEM2_CRITIC_PROMPT_STOCK_JSON
    if bool(allow_clear):
        sys_prompt = _patch_prompt_allow_clear(sys_prompt)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _build_system2_judge_messages(
    *,
    symbol: str,
    date_str: str,
    feats: Dict[str, Any],
    news_contexts: List[str],
    proposal: Dict[str, Any],
    critic: Dict[str, Any],
    allow_clear: bool,
) -> List[Dict[str, str]]:
    lines = _build_system2_context_lines(symbol=symbol, date_str=date_str, feats=feats, news_contexts=news_contexts)
    lines.extend(["", "Proposal JSON:", json.dumps(proposal, ensure_ascii=False), "", "Critic JSON:", json.dumps(critic, ensure_ascii=False)])
    sys_prompt = SYSTEM2_JUDGE_PROMPT_STOCK_JSON
    if bool(allow_clear):
        sys_prompt = _patch_prompt_allow_clear(sys_prompt)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _load_macro_risk_map(csv_path: str) -> Dict[str, float]:
    path_s = str(csv_path or "").strip()
    if not path_s:
        return {}
    p = Path(path_s)
    if not p.exists():
        raise SystemExit(f"macro file not found: {p}")

    out: Dict[str, float] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        field_map = {str(k or "").strip().lower(): str(k) for k in (reader.fieldnames or [])}
        date_col = field_map.get("date")
        if date_col is None:
            raise SystemExit(f"macro file missing Date column: {p}")
        score_col = field_map.get("global_risk_score")
        if score_col is None:
            raise SystemExit(f"macro file missing Global_Risk_Score column: {p}")

        for row in reader:
            if not isinstance(row, dict):
                continue
            d = str(row.get(date_col) or "").strip()
            if not d:
                continue
            d_norm = str(d).split(" ")[0].strip()
            out[d_norm] = float(_to_float(row.get(score_col)))

    return out


def _macro_gear(score: float) -> Tuple[float, str]:
    s = float(score)
    if s >= 0.5:
        return 0.0, "NEUTRAL"
    if s >= 0.3:
        return 0.5, "LOW"
    return 1.0, "DRIVE"


def _load_chartist_signals(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if path is None or (not path.exists()):
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            ok = obj.get("ok")
            if ok is False:
                continue
            ticker = str(obj.get("ticker") or "").upper().strip()
            asof = str(obj.get("asof") or obj.get("date") or "").strip()
            if not ticker or not asof:
                continue
            out[(asof, ticker)] = obj
    return out


def _chartist_score(sig: Any, conf: Any, *, threshold: float) -> int:
    s = str(sig or "").strip().upper()
    c = _to_float(conf)
    if c <= float(threshold):
        return 0
    if s == "BULLISH":
        return 1
    if s == "BEARISH":
        return -1
    return 0


def _to_pct(x: Any) -> float:
    v = _to_float(x)
    if abs(v) <= 2.5:
        return v * 100.0
    return v


def _ticker_aliases(symbol: str) -> List[str]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return []

    aliases = [sym]
    common = {
        "TSLA": ["TESLA"],
        "AAPL": ["APPLE"],
        "MSFT": ["MICROSOFT"],
        "AMZN": ["AMAZON"],
        "NVDA": ["NVIDIA"],
        "META": ["META", "FACEBOOK"],
        "NFLX": ["NETFLIX"],
        "GOOG": ["GOOGLE", "ALPHABET"],
        "GOOGL": ["GOOGLE", "ALPHABET"],
        "LOW": ["LOWE'S", "LOWES"],
        "UNH": ["UNITEDHEALTH", "UNITED HEALTH"],
        "LMT": ["LOCKHEED", "LOCKHEED MARTIN"],
        "RTX": ["RAYTHEON", "RAYTHEON TECHNOLOGIES"],
    }
    for a in common.get(sym, []):
        aliases.append(str(a).strip().upper())

    out: List[str] = []
    seen = set()
    for a in aliases:
        a2 = str(a or "").strip().upper()
        if not a2:
            continue
        if a2 in seen:
            continue
        seen.add(a2)
        out.append(a2)
    return out


def _title_mentions_ticker(title: str, symbol: str) -> bool:
    t = str(title or "").strip()
    if not t:
        return False
    t_up = t.upper()
    for a in _ticker_aliases(symbol):
        if not a:
            continue
        if re.search(rf"\b{re.escape(a)}\b", t_up):
            return True
    return False


def build_raw_headline_context(it: Dict[str, Any]) -> Optional[str]:
    title = str(it.get("title") or "").strip()
    if not title:
        return None

    source = str(it.get("source") or "").strip()
    url = str(it.get("url") or "").strip()
    published_at = str(it.get("published_at") or "").strip()

    lines = []
    lines.append("Ticker News Headline (raw):")
    if source:
        lines.append(f"Source: {source}")
    if published_at:
        lines.append(f"PublishedAt: {published_at}")
    raw_json = it.get("raw_json")
    try:
        raw_obj = json.loads(raw_json) if isinstance(raw_json, str) else (raw_json if isinstance(raw_json, dict) else {})
    except Exception:
        raw_obj = {}
    if isinstance(raw_obj, dict):
        ie = raw_obj.get("impact_equity")
        if ie is not None:
            lines.append(f"ImpactEquity: {ie}")
    lines.append(f"Title: {title}")
    if url:
        lines.append(f"URL: {url}")
    return "\n".join(lines)


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
    allow_fallback: bool = True,
    ticker: str = "",
) -> List[str]:
    sp = str(signals_path or "").strip()
    if sp:
        fp = Path(sp)
    else:
        fp_assets = daily_dir / f"signals_assets_{date_str}.json"
        fp = fp_assets if fp_assets.exists() else (daily_dir / f"signals_{date_str}.json")
    if not fp.exists():
        return []

    try:
        items = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(items, list):
        return []

    def _normalize_assets(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip().upper()
            return [s] if s else []
        if isinstance(v, list):
            out: List[str] = []
            for x in v:
                s = str(x or "").strip().upper()
                if s:
                    out.append(s)
            return out
        return []

    def _item_assets(it: Dict[str, Any]) -> List[str]:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        for key in ("subject_assets", "tickers", "symbols", "assets"):
            if key in sig:
                assets = _normalize_assets(sig.get(key))
                if assets:
                    return assets
            if key in it:
                assets = _normalize_assets(it.get(key))
                if assets:
                    return assets
        return []

    def _extract_impact_equity_from_item(it: Dict[str, Any]) -> float:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
        if sig is not None:
            return float(_to_float(sig.get("impact_equity")))
        raw_json = it.get("raw_json")
        if isinstance(raw_json, str):
            try:
                raw_obj = json.loads(raw_json)
            except Exception:
                raw_obj = {}
        elif isinstance(raw_json, dict):
            raw_obj = raw_json
        else:
            raw_obj = {}
        if isinstance(raw_obj, dict):
            return float(_to_float(raw_obj.get("impact_equity")))
        return 0.0

    def _collect_candidates(*, enable_ticker_filter: bool) -> List[Tuple[float, str]]:
        out: List[Tuple[float, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue

            title = str(it.get("title") or "").strip()
            parse_ok = bool(it.get("parse_ok"))
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else None

            # Ticker filter: only keep items whose title mentions ticker (or known aliases).
            if enable_ticker_filter and str(ticker or "").strip():
                t_u = str(ticker).strip().upper()
                assets = _item_assets(it)
                if assets:
                    if t_u not in assets:
                        continue
                else:
                    if not _title_mentions_ticker(title, str(ticker)):
                        continue

            impact = _extract_impact_equity_from_item(it)

            if parse_ok and sig is not None:
                if abs(impact) < float(min_abs_impact):
                    continue
                ctx = build_news_context_from_signal_item(it)
                if not ctx:
                    continue
                out.append((abs(impact), ctx))
            else:
                if abs(impact) < float(min_abs_impact):
                    continue
                ctx = build_raw_headline_context(it)
                if not ctx:
                    continue
                out.append((abs(impact), ctx))
        return out

    candidates = _collect_candidates(enable_ticker_filter=True)
    if (not candidates) and str(ticker or "").strip() and bool(allow_fallback):
        candidates = _collect_candidates(enable_ticker_filter=False)

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(max_signals))
    return [c for _score, c in candidates[:k]]


def _extract_max_abs_impact_equity_from_contexts(contexts: List[str]) -> float:
    best = 0.0
    for c in contexts or []:
        if not isinstance(c, str):
            continue
        for line in str(c).splitlines():
            s = line.strip()
            if not s.startswith("ImpactEquity:"):
                continue
            v = _to_float(s.split(":", 1)[1].strip() if ":" in s else "")
            best = max(best, abs(float(v)))
    return float(best)


def prefix_news_contexts(user_text: str, contexts: List[str]) -> str:
    if not contexts:
        return user_text
    addon = "\n\n" + "\n\n".join(contexts) + "\n\n"
    return addon + user_text


def build_stock_messages(
    symbol: str,
    date_str: str,
    feats: Dict[str, Any],
    news_contexts: Optional[List[str]] = None,
    *,
    allow_clear: bool = False,
) -> List[Dict[str, str]]:
    tech = feats.get("technical") if isinstance(feats.get("technical"), dict) else {}
    sig = feats.get("signal") if isinstance(feats.get("signal"), dict) else {}

    def gv(d: Dict[str, Any], k: str, default: Any = "") -> Any:
        return d.get(k, default) if isinstance(d, dict) else default

    lines: List[str] = []
    lines.append(f"Ticker: {symbol}")
    lines.append(f"Date: {date_str}")
    if news_contexts:
        lines.extend([""])
        lines.extend(news_contexts)
        lines.extend([""])
    lines.extend(
        [
            f"Close: {gv(tech, 'close', '')}",
            f"Price vs MA20: {gv(tech, 'price_vs_ma20', '')}",
            f"Price vs MA200: {gv(tech, 'price_vs_ma200', '')}",
            f"Trend alignment: {gv(tech, 'trend_alignment', '')}",
            f"Breakout 20d high: {gv(tech, 'breakout_20d_high', '')}",
            f"Breakdown 20d low: {gv(tech, 'breakdown_20d_low', '')}",
            f"Return 5d: {gv(tech, 'return_5d', '')}",
            f"Return 21d: {gv(tech, 'return_21d', '')}",
            f"Return 63d: {gv(tech, 'return_63d', '')}",
            f"Volatility 20d: {gv(tech, 'volatility_20d', '')}",
            f"Volume ratio: {gv(tech, 'vol_ratio', '')}",
            f"Drawdown: {gv(tech, 'drawdown', '')}",
            f"Max drawdown 20d: {gv(tech, 'max_drawdown_20d', '')}",
            f"Max drawdown 60d: {gv(tech, 'max_drawdown_60d', '')}",
            f"Composite signal: {gv(sig, 'composite', '')}",
            "",
            "Decide BUY/SELL/HOLD/CLEAR for the next 5 days." if bool(allow_clear) else "Decide BUY/SELL/HOLD for the next 5 days.",
        ]
    )

    if news_contexts:
        lines.extend(
            [
                "",
                "Constraint: You MUST include a short quoted phrase copied from a provided news 'Title:' line in at least one reasoning_trace bullet.",
            ]
        )
    user = "\n".join(lines)

    sys_prompt = SYSTEM_PROMPT_STOCK_STRICT_JSON
    if bool(allow_clear):
        sys_prompt = _patch_prompt_allow_clear(sys_prompt)

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]


def build_stock_messages_fast(
    symbol: str,
    date_str: str,
    feats: Dict[str, Any],
    *,
    allow_clear: bool = False,
) -> List[Dict[str, str]]:
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
        f"Return 5d: {gv(tech, 'return_5d', '')}\n"
        f"Return 21d: {gv(tech, 'return_21d', '')}\n"
        f"Volatility 20d: {gv(tech, 'volatility_20d', '')}\n"
        f"Volume ratio: {gv(tech, 'vol_ratio', '')}\n"
        f"Max drawdown 20d: {gv(tech, 'max_drawdown_20d', '')}\n"
        f"Composite signal: {gv(sig, 'composite', '')}\n\n"
        + ("Decide BUY/SELL/HOLD/CLEAR for the next 5 days." if bool(allow_clear) else "Decide BUY/SELL/HOLD for the next 5 days.")
    )

    sys_prompt = SYSTEM_PROMPT_STOCK_FAST_JSON
    if bool(allow_clear):
        sys_prompt = _patch_prompt_allow_clear(sys_prompt)

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]


def validate_stock_decision(obj: Dict[str, Any], *, allow_clear: bool = False) -> Tuple[List[str], List[str]]:
    required = {"decision"}
    keys = set(obj.keys())
    missing = sorted(list(required - keys))
    extra = []
    decision = str(obj.get("decision") or "").strip().upper()
    allowed = {"BUY", "SELL", "HOLD"}
    if bool(allow_clear):
        allowed.add("CLEAR")
    if decision not in allowed:
        missing = sorted(list(set(missing + ["decision(enum BUY/SELL/HOLD)"])))
    return missing, extra


def _normalize_reasoning_trace(obj: Dict[str, Any]) -> None:
    rt = obj.get("reasoning_trace")
    if not isinstance(rt, list):
        rt = []
    items: List[str] = []
    for x in rt:
        s = str(x or "").strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
        if len(s) > 160:
            s = s[:160]
        items.append(s)
        if len(items) >= 3:
            break
    while len(items) < 3:
        items.append("No trace provided.")
    obj["reasoning_trace"] = items


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
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        # Force single-device placement for compatibility with PEFT adapters.
        # device_map='auto' may create meta tensors and later .to() will crash.
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["low_cpu_mem_usage"] = False
        try:
            frac_raw = str(__import__("os").environ.get("TRADING_MAX_MEMORY_FRAC", "") or "").strip()
            if frac_raw:
                # When forcing single-device placement (device_map={"":0}), avoid
                # max_memory-based dispatch which may trigger meta-tensor init paths.
                pass
        except Exception:
            pass
    else:
        model_kwargs["device_map"] = "cpu"

    if load_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    try:
        if torch.cuda.is_available():
            model_kwargs.setdefault("attn_implementation", "sdpa")
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def load_model_moe(
    base_model_id: str,
    adapters: Dict[str, str],
    load_4bit: bool,
    default_adapter: str,
) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("adapters must be non-empty dict")

    def _normalize_adapter_dir(p: str) -> str:
        pp = Path(str(p or "").strip())
        if not str(pp):
            return str(pp)
        if pp.exists() and pp.is_dir():
            lw = pp / "lora_weights"
            if lw.exists() and lw.is_dir():
                return str(lw)
        # If caller passed /lora_weights but it doesn't exist, fall back to parent
        if pp.name.lower() == "lora_weights" and (not pp.exists()) and pp.parent.exists():
            return str(pp.parent)
        return str(pp)

    adapters = {str(k): _normalize_adapter_dir(v) for k, v in adapters.items() if str(v or "").strip()}

    tok_src = base_model_id
    for p in adapters.values():
        if str(p or "").strip() and Path(str(p), "tokenizer_config.json").exists():
            tok_src = str(p)
            break
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

        # Force single-device placement for compatibility with PEFT multi-adapter loading.
        # device_map='auto' can create meta tensors and then .to() crashes.
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["low_cpu_mem_usage"] = False
        try:
            frac_raw = str(__import__("os").environ.get("TRADING_MAX_MEMORY_FRAC", "") or "").strip()
            if frac_raw:
                # When forcing single-device placement (device_map={"":0}), avoid
                # max_memory-based dispatch which may trigger meta-tensor init paths.
                pass
        except Exception:
            pass
    else:
        model_kwargs["device_map"] = "cpu"

    if load_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    try:
        if torch.cuda.is_available():
            model_kwargs.setdefault("attn_implementation", "sdpa")
    except Exception:
        pass

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    from peft import PeftModel

    names = list(adapters.keys())
    first = names[0]
    model = PeftModel.from_pretrained(base_model, str(adapters[first]), adapter_name=str(first))
    for name in names[1:]:
        model.load_adapter(str(adapters[name]), adapter_name=str(name))

    if str(default_adapter) in set(names):
        model.set_adapter(str(default_adapter))
    else:
        model.set_adapter(str(first))

    model.eval()
    return model, tokenizer


def _router_choose_expert(
    *,
    symbol: str,
    feats: Dict[str, Any],
    news_contexts: List[str],
    moe_any_news: bool,
    moe_news_threshold: float,
    moe_vol_threshold: float,
) -> Tuple[str, Dict[str, Any]]:
    vol = feats.get("volatility_ann_pct")
    if vol is None:
        vol = _to_pct((feats.get("technical") or {}).get("volatility_20d")) if isinstance(feats.get("technical"), dict) else None
    vol_f = _to_float(vol)

    has_any_news = bool(news_contexts)
    news_score = _extract_max_abs_impact_equity_from_contexts(news_contexts)

    use_analyst = False
    if moe_any_news and has_any_news:
        use_analyst = True
    elif news_score >= float(moe_news_threshold):
        use_analyst = True
    elif (float(moe_vol_threshold) > 0.0) and (vol_f >= float(moe_vol_threshold)):
        use_analyst = True

    expert = "analyst" if use_analyst else "scalper"
    meta = {
        "expert": expert,
        "news_count": int(len(news_contexts)),
        "news_score": float(news_score),
        "volatility_ann_pct": float(vol_f),
    }
    return expert, meta


def _parse_ymd(s: str) -> datetime:
    return datetime.strptime(str(s).strip(), "%Y-%m-%d")


def _iter_dates_inclusive(start_ymd: str, end_ymd: str) -> List[str]:
    a = _parse_ymd(start_ymd)
    b = _parse_ymd(end_ymd)
    if b < a:
        a, b = b, a
    cur = a
    out: List[str] = []
    while cur <= b:
        out.append(cur.strftime("%Y-%m-%d"))
        cur = cur + timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trading LoRA inference with MarketRAG + RiskGate")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="YYYY-MM-DD")
    g.add_argument("--date-range", nargs=2, metavar=("START", "END"), help="YYYY-MM-DD YYYY-MM-DD")
    parser.add_argument("--daily-dir", default="data/daily")
    parser.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter", default="", help="LoRA adapter path (output dir or lora_weights dir)")
    parser.add_argument("--model", dest="base", help="Alias of --base")
    parser.add_argument("--lora", dest="adapter", help="Alias of --adapter")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers for stock integration")
    parser.add_argument("--universe", default="auto", choices=["auto", "stock", "etf"])
    parser.add_argument("--use-fast-prompt", action="store_true", default=False)
    parser.add_argument(
        "--planner-mode",
        default="off",
        choices=["off", "rule", "sft"],
        help="Global strategy planner. If enabled, can gate MoE expert choice (defensive/cash_preservation => disable analyst).",
    )
    parser.add_argument(
        "--planner-policy",
        default="",
        choices=["", "rule", "sft"],
        help="Planner policy override. If set, takes precedence over --planner-mode.",
    )
    parser.add_argument("--planner-sft-model", default="models/planner_sft_v1.pt")
    parser.add_argument("--planner-rl-model", default="", help="Phase 19.1: RL Gatekeeper model path (Q(s,allow) regressor bundle)")
    parser.add_argument(
        "--planner-rl-threshold",
        type=float,
        default=0.0,
        help="Phase 19.1: deny if q_allow <= threshold. Example: 0.0 blocks negative expected reward.",
    )
    parser.add_argument("--planner-signals-csv", default="paper_trading/daily_signals.csv")
    parser.add_argument(
        "--risk-max-drawdown",
        type=float,
        default=0.08,
        help="Max drawdown limit (e.g. 0.08 for 8%%). If >1, treated as percent points (e.g. 8 for 8%%).",
    )
    parser.add_argument(
        "--risk-vol-limit",
        type=float,
        default=0.0,
        help="Daily volatility limit (e.g. 0.03 for 3%% daily). If 0, uses default gate settings.",
    )
    parser.add_argument("--risk-watch-market", default="BOTH", help="CN|US|BOTH|NONE")
    parser.add_argument("--risk-watch-top", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument("--load-4bit", dest="load_4bit", action="store_true")
    parser.add_argument("--load-in-4bit", dest="load_4bit", action="store_true")
    parser.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    parser.set_defaults(load_4bit=True)
    parser.add_argument("--disable-news", action="store_true", default=False)
    parser.add_argument(
        "--allow-clear",
        action="store_true",
        default=False,
        help="Allow model to output CLEAR (cash) in decision schema. Useful for DPO-trained adapters.",
    )
    parser.add_argument("--signals", dest="signals_path", default="", help="Override signals_YYYY-MM-DD.json path for stock news injection")
    parser.add_argument("--min-news-abs-impact", type=float, default=0.5)
    parser.add_argument("--max-news-signals", type=int, default=3)
    parser.add_argument(
        "--system2-debate",
        default="off",
        choices=["off", "buy_only", "all"],
        help="System-2 debate: proposal->critic->judge. off=disabled, buy_only=run only when proposal is BUY, all=run for all proposals.",
    )
    parser.add_argument("--system2-max-new-tokens", type=int, default=256)
    parser.add_argument("--system2-temperature", type=float, default=0.0)
    parser.add_argument(
        "--chart-signals-file",
        default="",
        help="Optional Chartist signals jsonl (ticker/asof/signal/confidence) keyed by (date,ticker)",
    )
    parser.add_argument(
        "--chart-confidence",
        type=float,
        default=0.7,
        help="Chartist confidence threshold for overlay (default: 0.7)",
    )
    parser.add_argument(
        "--chart-mode",
        default="standard",
        choices=["standard", "conservative"],
        help="Chartist overlay mode: standard=upgrade+block, conservative=block only",
    )
    parser.add_argument(
        "--macro-file",
        default="",
        help="Optional macro features CSV (Date,Global_Risk_Score). Applies Macro Governor to clamp long exposure.",
    )
    parser.add_argument("--out", default="data/decisions_inference.json")
    parser.add_argument("--output", dest="out", help="Alias of --out")
    parser.add_argument("--moe-mode", action="store_true", default=False)
    parser.add_argument("--moe-scalper", default="models/trader_stock_v1_1_tech_plus_news/lora_weights")
    parser.add_argument("--moe-analyst", default="models/trader_v2_cot_scaleup/lora_weights")
    parser.add_argument("--adapter-scalper", dest="moe_scalper", help="Alias of --moe-scalper")
    parser.add_argument("--adapter-analyst", dest="moe_analyst", help="Alias of --moe-analyst")
    parser.add_argument("--moe-any-news", action="store_true", default=True)
    parser.add_argument("--no-moe-any-news", dest="moe_any_news", action="store_false")
    parser.add_argument("--moe-news-threshold", type=float, default=0.8)
    parser.add_argument("--moe-vol-threshold", type=float, default=-1.0)
    args = parser.parse_args()

    from src.data.rag import MarketRAG
    from src.risk.gate import RiskGate
    from src.agent.gatekeeper import Gatekeeper
    from src.agent.planner import Planner
    from scripts.generate_etf_teacher_dataset import (
        build_risk_watch_summary,
        build_teacher_messages,
        validate_label,
    )

    daily_dir = Path(args.daily_dir)

    tickers = [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]
    use_stock = (str(args.universe).lower() == "stock") or (
        (str(args.universe).lower() == "auto") and bool(tickers)
    )

    date_list: List[str] = []
    if getattr(args, "date_range", None):
        date_list = _iter_dates_inclusive(str(args.date_range[0]), str(args.date_range[1]))
    else:
        date_list = [str(args.date)]

    chart_signals_path = str(getattr(args, "chart_signals_file", "") or "").strip()
    chart_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if chart_signals_path:
        try:
            chart_map = _load_chartist_signals(Path(chart_signals_path))
        except Exception:
            chart_map = {}

    macro_file = str(getattr(args, "macro_file", "") or "").strip()
    macro_risk_map: Dict[str, float] = {}
    if macro_file:
        macro_risk_map = _load_macro_risk_map(macro_file)

    rag = MarketRAG(data_dir=str(daily_dir))
    dd_in = float(getattr(args, "risk_max_drawdown", 0.08) or 0.08)
    dd_limit_pct = -abs(dd_in) * 100.0 if abs(dd_in) <= 1.0 else -abs(dd_in)

    vol_in = float(getattr(args, "risk_vol_limit", 0.0) or 0.0)
    if vol_in <= 0:
        vol_trigger_ann_pct = 30.0
    else:
        daily_vol = abs(vol_in) if abs(vol_in) <= 1.0 else abs(vol_in) / 100.0
        vol_trigger_ann_pct = daily_vol * 100.0 * math.sqrt(252.0)

    risk_gate = RiskGate(
        max_drawdown_limit_pct=float(dd_limit_pct),
        vol_reduce_trigger_ann_pct=float(vol_trigger_ann_pct),
    )

    planner: Optional[Planner] = None
    gatekeeper: Optional[Gatekeeper] = None
    planner_mode = str(getattr(args, "planner_mode", "off") or "off").strip().lower()
    planner_policy = str(getattr(args, "planner_policy", "") or "").strip().lower()
    if not planner_policy:
        planner_policy = "rule" if planner_mode == "rule" else ("sft" if planner_mode == "sft" else "")
    if planner_policy in {"rule", "sft"}:
        planner = Planner(policy=planner_policy, sft_model_path=str(args.planner_sft_model))

    rl_model_path = str(getattr(args, "planner_rl_model", "") or "").strip()
    if rl_model_path:
        gatekeeper = Gatekeeper(model_path=str(rl_model_path), threshold=float(getattr(args, "planner_rl_threshold", 0.0) or 0.0))

    if bool(args.moe_mode):
        scalper = str(args.moe_scalper).strip()
        analyst = str(args.moe_analyst).strip()
        if not scalper or not analyst:
            raise SystemExit("--moe-mode requires --moe-scalper and --moe-analyst")
        model, tokenizer = load_model_moe(
            str(args.base),
            {"scalper": scalper, "analyst": analyst},
            bool(args.load_4bit),
            default_adapter="scalper",
        )
    else:
        if bool(args.use_lora) and not str(args.adapter).strip():
            raise SystemExit("--use-lora requires --adapter/--lora")
        model, tokenizer = load_model(str(args.base), str(args.adapter), bool(args.load_4bit))

    multi_day = len(date_list) > 1
    if multi_day:
        out: Dict[str, Any] = {
            "date_range": {"start": date_list[0], "end": date_list[-1]},
            "base": str(args.base),
            "adapter": str(args.adapter),
            "days": {},
        }
        if bool(args.moe_mode):
            out["moe"] = {
                "enabled": True,
                "scalper": str(args.moe_scalper),
                "analyst": str(args.moe_analyst),
                "any_news": bool(args.moe_any_news),
                "news_threshold": float(args.moe_news_threshold),
                "vol_threshold": float(args.moe_vol_threshold),
            }
    else:
        out = {}

    total_dates = len(date_list)
    out_path_for_state = Path(str(getattr(args, "out", "data/decisions_inference.json") or "data/decisions_inference.json")).resolve()
    state_dir = _infer_state_dir(out_path_for_state)
    run_id = str(state_dir.name) if str(state_dir.parent.name).lower() == "results" else ""
    run_started_ts = float(time.time())
    _write_engine_state(
        state_dir,
        {
            "kind": "trading",
            "run_id": run_id,
            "status": "running",
            "stage": "init",
            "out": str(out_path_for_state),
            "updated_at": float(time.time()),
        },
    )
    _append_engine_event(
        state_dir,
        {
            "ts": float(time.time()),
            "type": "run_start",
            "run_id": run_id,
            "out": str(out_path_for_state),
        },
    )

    for i, date_str in enumerate(date_list, start=1):
        _write_engine_state(
            state_dir,
            {
                "kind": "trading",
                "run_id": run_id,
                "status": "running",
                "stage": "date_start",
                "date_index": int(i),
                "date_total": int(total_dates),
                "current_date": str(date_str),
                "out": str(out_path_for_state),
                "updated_at": float(time.time()),
            },
        )
        _append_engine_event(
            state_dir,
            {
                "ts": float(time.time()),
                "type": "date_start",
                "run_id": run_id,
                "date": str(date_str),
                "date_index": int(i),
                "date_total": int(total_dates),
            },
        )
        if bool(args.progress) and total_dates > 1:
            print(f"[date {i}/{total_dates}] {date_str}", flush=True)

        macro_score = float(macro_risk_map.get(str(date_str), 0.0)) if macro_risk_map else 0.0
        macro_multiplier, macro_gear = _macro_gear(macro_score)

        stock_fp = daily_dir / f"stock_features_{date_str}.json"
        etf_fp = daily_dir / f"etf_features_{date_str}.json"

        missing_tickers: List[str] = []
        planner_decision: Optional[Dict[str, Any]] = None
        gate_decision: Optional[Dict[str, Any]] = None
        gate_allow: Optional[bool] = None
        news_cache: Dict[str, List[str]] = {}
        if use_stock:
            if not stock_fp.exists():
                items = []
            else:
                stock_payload = json.loads(stock_fp.read_text(encoding="utf-8"))
                stock_items = iter_feature_items(stock_payload)
                if tickers:
                    stock_map = {str(sym).upper(): it for sym, it in stock_items}
                    etf_payload = load_daily_payload(daily_dir, str(date_str))
                    etf_map: Dict[str, Dict[str, Any]] = {}
                    if etf_payload is not None:
                        for sym, it in iter_feature_items(etf_payload):
                            etf_map[str(sym).upper()] = it

                    items = []
                    for t in tickers:
                        if t in stock_map:
                            items.append((t, stock_map[t]))
                        elif t in etf_map:
                            items.append((t, etf_map[t]))
                        else:
                            missing_tickers.append(t)
                else:
                    items = stock_items
        else:
            payload = load_daily_payload(daily_dir, str(date_str))
            if payload is None:
                continue
            items = iter_feature_items(payload)

        if not isinstance(items, list):
            try:
                items = list(items)
            except Exception:
                items = []

        if not items:
            decisions: Dict[str, Any] = {
                "date": str(date_str),
                "base": str(args.base),
                "adapter": str(args.adapter),
                "risk_watch": {"available": False, "signals_path": str(daily_dir / f"signals_{date_str}.json")},
                "items": {},
            }
            if missing_tickers:
                decisions["missing_tickers"] = missing_tickers
            if multi_day:
                out["days"][str(date_str)] = decisions
            else:
                out = decisions
            continue

        risk_watch = build_risk_watch_summary(
            daily_dir=daily_dir,
            date_str=str(date_str),
            market_mode=str(args.risk_watch_market),
            top_k=int(args.risk_watch_top),
        )

        if planner is not None:
            try:
                mr = None
                if use_stock:
                    if isinstance(stock_payload.get("items"), list) and stock_payload.get("items"):
                        first = stock_payload.get("items")[0]
                        if isinstance(first, dict):
                            mr = first.get("market_regime")
                else:
                    if isinstance(payload, dict):
                        its = payload.get("items")
                        if isinstance(its, list) and its:
                            first = its[0]
                            if isinstance(first, dict):
                                mr = first.get("market_regime")

                vols: List[float] = []
                news_counts: List[int] = []
                news_scores: List[float] = []

                if use_stock and (not bool(args.disable_news)):
                    for sym, _it in items:
                        sym_u = str(sym).upper().strip()
                        ctxs = load_daily_news_contexts(
                            daily_dir=daily_dir,
                            date_str=str(date_str),
                            signals_path=str(args.signals_path),
                            min_abs_impact=float(args.min_news_abs_impact),
                            max_signals=int(args.max_news_signals),
                            allow_fallback=False,
                            ticker=str(sym_u),
                        )
                        news_cache[sym_u] = ctxs
                        news_counts.append(int(len(ctxs)))
                        news_scores.append(_extract_max_abs_impact_equity_from_contexts(ctxs))

                for sym, it in items:
                    feats = extract_features(it)
                    v = feats.get("volatility_ann_pct")
                    vols.append(float(_to_float(v)))
                    if not use_stock:
                        news_counts.append(0)
                        news_scores.append(0.0)

                vol_mean = float(sum(vols) / max(1, len(vols))) if vols else 0.0
                vol_max = float(max(vols)) if vols else 0.0

                nc_sum = float(sum(news_counts)) if news_counts else 0.0
                nc_mean = float(sum(news_counts) / max(1, len(news_counts))) if news_counts else 0.0
                ns_mean = float(sum(news_scores) / max(1, len(news_scores))) if news_scores else 0.0
                ns_max = float(max(news_scores)) if news_scores else 0.0

                def _strong_from_risk_watch(rw: Dict[str, Any], thr: float) -> bool:
                    if not isinstance(rw, dict):
                        return False
                    if not bool(rw.get("available")):
                        return False
                    for it in rw.get("us_top_events") if isinstance(rw.get("us_top_events"), list) else []:
                        if isinstance(it, dict) and abs(_to_float(it.get("impact_equity"))) >= float(thr):
                            return True
                    cn = rw.get("cn_regulation_crackdown") if isinstance(rw.get("cn_regulation_crackdown"), dict) else {}
                    top = cn.get("top") if isinstance(cn.get("top"), list) else []
                    for it in top:
                        if isinstance(it, dict) and abs(_to_float(it.get("impact_equity"))) >= float(thr):
                            return True
                    return False

                has_strong = 1 if _strong_from_risk_watch(risk_watch, float(args.moe_news_threshold)) else 0

                planner_feats = {
                    "n_tickers": float(len(items)),
                    "vol_mean": float(vol_mean),
                    "vol_max": float(vol_max),
                    "news_count_sum": float(nc_sum),
                    "news_count_mean": float(nc_mean),
                    "news_score_mean": float(ns_mean),
                    "news_score_max": float(ns_max),
                    "has_strong_news_day": float(has_strong),
                }

                planner_decision = planner.decide(
                    market_regime=mr,
                    features=planner_feats,
                    date_str=str(date_str),
                    signals_csv=str(args.planner_signals_csv),
                ).to_dict()
            except Exception:
                planner_decision = None

        if (gatekeeper is not None) and isinstance(planner_decision, dict):
            try:
                p_inputs = planner_decision.get("inputs") if isinstance(planner_decision.get("inputs"), dict) else {}
                p_feats = p_inputs.get("features") if isinstance(p_inputs.get("features"), dict) else {}
                probs = p_inputs.get("probs") if isinstance(p_inputs.get("probs"), dict) else {}
                s = str(planner_decision.get("strategy") or "").strip().lower()
                conf = 0.0
                try:
                    conf = float(max([float(x) for x in probs.values()] or [0.0]))
                except Exception:
                    conf = 0.0

                gate_feats = {str(k): float(v) for k, v in p_feats.items() if k}
                gate_feats.update(
                    {
                        "sft_is_aggressive_long": 1.0 if s == "aggressive_long" else 0.0,
                        "sft_is_defensive": 1.0 if s == "defensive" else 0.0,
                        "sft_is_cash_preservation": 1.0 if s == "cash_preservation" else 0.0,
                        "sft_confidence": float(conf),
                    }
                )

                gd = gatekeeper.decide(feats=gate_feats, threshold=float(getattr(args, "planner_rl_threshold", 0.0) or 0.0))
                gate_decision = gd.to_dict()
                gate_allow = bool(gd.allow)
            except Exception:
                gate_decision = None
                gate_allow = None

        decisions = {
            "date": str(date_str),
            "base": str(args.base),
            "adapter": str(args.adapter),
            "risk_watch": risk_watch,
            "items": {},
        }
        if planner_decision is not None:
            decisions["planner"] = planner_decision
        if gate_decision is not None:
            decisions["gatekeeper"] = gate_decision

        if missing_tickers:
            decisions["missing_tickers"] = missing_tickers

        if bool(args.moe_mode):
            decisions["moe"] = {
                "enabled": True,
                "scalper": str(args.moe_scalper),
                "analyst": str(args.moe_analyst),
                "any_news": bool(args.moe_any_news),
                "news_threshold": float(args.moe_news_threshold),
                "vol_threshold": float(args.moe_vol_threshold),
            }

        news_signals = extract_news_signals(risk_watch)

        # Phase 19.1: If gatekeeper denies, force CLEAR across all symbols and skip model inference.
        if gate_allow is False:
            for symbol, _item in items:
                if use_stock:
                    parsed = {"decision": "HOLD", "ticker": str(symbol), "analysis": ""}
                    _normalize_reasoning_trace(parsed)
                else:
                    parsed = {
                        "role_aggressive": "",
                        "role_risk": "",
                        "role_quant": "",
                        "synthesis": "",
                        "label": {
                            "action": "clear",
                            "target_position": 0.0,
                            "risk_notes": [],
                            "rationale": "gatekeeper_deny",
                        },
                    }

                rec: Dict[str, Any] = {
                    "parsed": parsed,
                    "parse_error": "",
                    "final": {
                        "action": "CLEAR",
                        "target_position": 0.0,
                        "trace": ["[GATEKEEPER] Deny => FORCE CLEAR."],
                    },
                    "raw": "",
                }
                decisions["items"][symbol] = rec
                print(f"{symbol}: CLEAR 0.0")
                _append_engine_event(
                    state_dir,
                    {
                        "ts": float(time.time()),
                        "type": "ticker_end",
                        "run_id": run_id,
                        "date": str(date_str),
                        "ticker": str(symbol),
                        "final_action": "CLEAR",
                        "final_position": 0.0,
                        "reason": "gatekeeper_deny",
                    },
                )

            if multi_day:
                out["days"][str(date_str)] = decisions
            else:
                out = decisions
            _append_engine_event(
                state_dir,
                {
                    "ts": float(time.time()),
                    "type": "date_end",
                    "run_id": run_id,
                    "date": str(date_str),
                    "date_index": int(i),
                    "date_total": int(total_dates),
                },
            )
            continue

        for symbol, etf_item in items:
            feats = extract_features(etf_item)

            similar_days: List[Dict[str, Any]] = []
            if not use_stock:
                try:
                    similar_days = rag.retrieve(feats, k=3, ticker=symbol, exclude_date=str(date_str))
                except Exception:
                    similar_days = []

            try:
                _, _, risk_trace = risk_gate.adjudicate(feats, news_signals, "BUY", 1.0)
            except Exception:
                risk_trace = []

            expert = "default"
            router_meta: Dict[str, Any] = {}

            if use_stock:
                stock_news_contexts: List[str] = []
                route_news_contexts: List[str] = []
                if not bool(args.disable_news):
                    sym_u = str(symbol).upper().strip()
                    if sym_u and sym_u in news_cache:
                        route_news_contexts = list(news_cache.get(sym_u) or [])
                    else:
                        route_news_contexts = load_daily_news_contexts(
                            daily_dir=daily_dir,
                            date_str=str(date_str),
                            signals_path=str(args.signals_path),
                            min_abs_impact=float(args.min_news_abs_impact),
                            max_signals=int(args.max_news_signals),
                            allow_fallback=False,
                            ticker=str(symbol),
                        )
                        if sym_u:
                            news_cache[sym_u] = list(route_news_contexts)

                if bool(args.moe_mode):
                    expert, router_meta = _router_choose_expert(
                        symbol=str(symbol),
                        feats=feats,
                        news_contexts=route_news_contexts,
                        moe_any_news=bool(args.moe_any_news),
                        moe_news_threshold=float(args.moe_news_threshold),
                        moe_vol_threshold=float(args.moe_vol_threshold),
                    )

                    if planner_decision is not None:
                        strategy = str(planner_decision.get("strategy") or "").strip()
                        router_meta = dict(router_meta)
                        router_meta["planner_strategy"] = strategy
                        if strategy and strategy != "aggressive_long" and str(expert) == "analyst" and (not route_news_contexts):
                            router_meta["expert_before_planner_gate"] = str(router_meta.get("expert") or str(expert))
                            expert = "scalper"
                            router_meta["expert"] = "scalper"
                            router_meta["planner_gate"] = "disabled_analyst"

                    model.set_adapter(str(expert))
                    if str(expert) == "analyst":
                        if not bool(args.disable_news):
                            stock_news_contexts = load_daily_news_contexts(
                                daily_dir=daily_dir,
                                date_str=str(date_str),
                                signals_path=str(args.signals_path),
                                min_abs_impact=float(args.min_news_abs_impact),
                                max_signals=int(args.max_news_signals),
                                allow_fallback=True,
                                ticker=str(symbol),
                            )
                        messages = build_stock_messages(
                            str(symbol),
                            str(date_str),
                            etf_item,
                            stock_news_contexts,
                            allow_clear=bool(args.allow_clear),
                        )
                    else:
                        messages = build_stock_messages_fast(
                            str(symbol),
                            str(date_str),
                            etf_item,
                            allow_clear=bool(args.allow_clear),
                        )
                else:
                    if bool(args.use_fast_prompt):
                        messages = build_stock_messages_fast(
                            str(symbol),
                            str(date_str),
                            etf_item,
                            allow_clear=bool(args.allow_clear),
                        )
                    else:
                        messages = build_stock_messages(
                            str(symbol),
                            str(date_str),
                            etf_item,
                            stock_news_contexts,
                            allow_clear=bool(args.allow_clear),
                        )
            else:
                messages = build_teacher_messages(
                    symbol=str(symbol),
                    date_str=str(date_str),
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

            gen_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            raw_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

            parsed: Optional[Dict[str, Any]] = None
            parse_error = ""
            try:
                parsed, parse_error = _try_parse_json_dict(raw_text)
                if parsed is None:
                    raise ValueError(parse_error or "parse failed")
                if use_stock:
                    if not str(parsed.get("ticker") or "").strip():
                        parsed["ticker"] = str(symbol)
                    if "analysis" not in parsed:
                        parsed["analysis"] = ""
                    _normalize_reasoning_trace(parsed)
                    missing, extra = validate_stock_decision(parsed, allow_clear=bool(args.allow_clear))
                else:
                    if isinstance(parsed, dict) and ("reasoning_trace" in parsed):
                        parsed.pop("reasoning_trace", None)
                    missing, extra = validate_label(parsed)
                if missing or extra:
                    raise ValueError(f"schema mismatch missing={missing} extra={extra}")
            except Exception as e:
                parsed = None
                parse_error = str(e)

            final_action = "HOLD"
            final_pos = 0.0
            final_trace: List[str] = []
            proposed_action = "HOLD"
            proposed_pos: Any = 0.0
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

            system2_meta: Optional[Dict[str, Any]] = None
            s2_mode = str(getattr(args, "system2_debate", "off") or "off").strip().lower()
            if use_stock and parsed is not None and s2_mode in {"buy_only", "all"}:
                decision_up = str(proposed_action).strip().upper()
                if (s2_mode == "all") or (decision_up == "BUY"):
                    proposal_obj = dict(parsed)
                    critic_obj: Optional[Dict[str, Any]] = None
                    judge_obj: Optional[Dict[str, Any]] = None
                    critic_raw = ""
                    judge_raw = ""
                    critic_err = ""
                    judge_err = ""

                    ctx_for_s2 = []
                    if isinstance(stock_news_contexts, list) and stock_news_contexts:
                        ctx_for_s2 = list(stock_news_contexts)
                    elif isinstance(route_news_contexts, list) and route_news_contexts:
                        ctx_for_s2 = list(route_news_contexts)

                    critic_msgs = _build_system2_critic_messages(
                        symbol=str(symbol),
                        date_str=str(date_str),
                        feats=etf_item,
                        news_contexts=ctx_for_s2,
                        proposed_decision=str(proposal_obj.get("decision") or "HOLD"),
                        proposed_analysis=str(proposal_obj.get("analysis") or ""),
                        allow_clear=bool(args.allow_clear),
                    )
                    critic_raw = _llm_generate(
                        model=model,
                        tokenizer=tokenizer,
                        messages=critic_msgs,
                        max_new_tokens=int(getattr(args, "system2_max_new_tokens", 256) or 256),
                        temperature=float(getattr(args, "system2_temperature", 0.0) or 0.0),
                    )
                    critic_obj, critic_err = _try_parse_json_dict(critic_raw)
                    if isinstance(critic_obj, dict):
                        if not isinstance(critic_obj.get("accept"), bool):
                            critic_err = "critic schema missing/invalid accept"
                            critic_obj = None
                        sd = str(critic_obj.get("suggested_decision") or "").strip().upper()
                        if sd not in {"BUY", "SELL", "HOLD", "CLEAR"}:
                            critic_err = "critic schema missing/invalid suggested_decision"
                            critic_obj = None
                        rs = critic_obj.get("reasons")
                        if critic_obj is not None and (not isinstance(rs, list) or len(rs) != 3):
                            critic_err = "critic schema missing/invalid reasons"
                            critic_obj = None

                    if critic_obj is not None:
                        judge_msgs = _build_system2_judge_messages(
                            symbol=str(symbol),
                            date_str=str(date_str),
                            feats=etf_item,
                            news_contexts=ctx_for_s2,
                            proposal=proposal_obj,
                            critic=critic_obj,
                            allow_clear=bool(args.allow_clear),
                        )
                        judge_raw = _llm_generate(
                            model=model,
                            tokenizer=tokenizer,
                            messages=judge_msgs,
                            max_new_tokens=int(getattr(args, "system2_max_new_tokens", 256) or 256),
                            temperature=float(getattr(args, "system2_temperature", 0.0) or 0.0),
                        )
                        judge_obj, judge_err = _try_parse_json_dict(judge_raw)
                        if isinstance(judge_obj, dict):
                            fd = str(judge_obj.get("final_decision") or "").strip().upper()
                            if fd not in {"BUY", "SELL", "HOLD", "CLEAR"}:
                                judge_err = "judge schema missing/invalid final_decision"
                                judge_obj = None
                            if judge_obj is not None and not str(judge_obj.get("rationale") or "").strip():
                                judge_err = "judge schema missing/invalid rationale"
                                judge_obj = None

                    if judge_obj is not None:
                        final_from_s2 = str(judge_obj.get("final_decision") or "HOLD").strip().upper()
                        proposed_action = final_from_s2
                        proposed_pos = 0.5 if final_from_s2 == "BUY" else 0.0

                    system2_meta = {
                        "mode": s2_mode,
                        "proposal": proposal_obj,
                        "critic": critic_obj,
                        "judge": judge_obj,
                        "raw": {"critic": critic_raw, "judge": judge_raw},
                        "errors": {"critic": critic_err, "judge": judge_err},
                    }

            chartist_meta: Optional[Dict[str, Any]] = None
            ensemble_trace: List[str] = []
            if use_stock and chart_map:
                sym_u = str(symbol).upper().strip()
                key = (str(date_str), sym_u)
                cs = chart_map.get(key)
                if isinstance(cs, dict):
                    sig = cs.get("signal")
                    conf = cs.get("confidence")
                    thr = float(getattr(args, "chart_confidence", 0.7) or 0.7)
                    mode = str(getattr(args, "chart_mode", "standard") or "standard").strip().lower()
                    score = _chartist_score(sig, conf, threshold=float(thr))
                    chartist_meta = {
                        "signal": sig,
                        "confidence": conf,
                        "reasoning": cs.get("reasoning"),
                        "score": int(score),
                    }
                    decision_up = str(proposed_action).strip().upper()
                    if score >= 1 and mode == "standard" and decision_up != "BUY":
                        ensemble_trace.append("[CHARTIST] strong BULLISH => upgrade to BUY")
                        proposed_action = "BUY"
                        proposed_pos = 0.5
                    elif score <= -1 and decision_up == "BUY":
                        ensemble_trace.append("[CHARTIST] strong BEARISH => block BUY => HOLD")
                        proposed_action = "HOLD"
                        proposed_pos = 0.0

            try:
                final_action, final_pos, final_trace = risk_gate.adjudicate(
                    feats,
                    news_signals,
                    proposed_action,
                    proposed_pos,
                )
                if ensemble_trace:
                    final_trace = list(ensemble_trace) + list(final_trace)
            except Exception as e:
                final_action = str(proposed_action).upper()
                try:
                    final_pos = float(proposed_pos)
                except Exception:
                    final_pos = 0.0
                final_trace = (list(ensemble_trace) if ensemble_trace else []) + [f"[RISK] adjudicate error: {e}"]

            if macro_file and float(final_pos) > 0.0 and float(macro_multiplier) < 1.0:
                capped = min(float(final_pos), float(macro_multiplier))
                if capped < float(final_pos) - 1e-12:
                    final_pos = float(capped)
                    if float(final_pos) <= 1e-12:
                        final_action = "HOLD"
                    final_trace = list(final_trace) + [
                        f"[MACRO] gear={macro_gear} risk={macro_score:.4f} => cap={float(macro_multiplier):.4f}"
                    ]

            rec: Dict[str, Any] = {
                "parsed": parsed,
                "parse_error": parse_error,
                "final": {"action": final_action, "target_position": final_pos, "trace": final_trace},
                "raw": raw_text,
            }

            if system2_meta is not None:
                rec["system2"] = system2_meta

            if chartist_meta is not None:
                rec["chartist"] = chartist_meta

            if bool(args.moe_mode) and use_stock:
                rec["expert"] = str(expert)
                rec["router"] = router_meta

            decisions["items"][symbol] = rec

            if parsed is not None:
                print(f"{symbol}: {final_action} {final_pos}")
            else:
                print(f"{symbol}: PARSE_ERROR {parse_error}")

            _append_engine_event(
                state_dir,
                {
                    "ts": float(time.time()),
                    "type": "ticker_end",
                    "run_id": run_id,
                    "date": str(date_str),
                    "ticker": str(symbol),
                    "expert": str(expert) if bool(args.moe_mode) and use_stock else "",
                    "final_action": str(final_action),
                    "final_position": float(final_pos),
                    "parse_error": str(parse_error)[:200] if parse_error else "",
                },
            )

        if multi_day:
            if macro_file:
                decisions["macro"] = {
                    "risk_score": float(macro_score),
                    "gear": str(macro_gear),
                    "multiplier": float(macro_multiplier),
                    "source": str(macro_file),
                }
            out["days"][str(date_str)] = decisions
        else:
            out = decisions

        _append_engine_event(
            state_dir,
            {
                "ts": float(time.time()),
                "type": "date_end",
                "run_id": run_id,
                "date": str(date_str),
                "date_index": int(i),
                "date_total": int(total_dates),
            },
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")

    _write_engine_state(
        state_dir,
        {
            "kind": "trading",
            "run_id": run_id,
            "status": "done",
            "stage": "complete",
            "out": str(out_path.resolve()),
            "elapsed_sec": float(time.time() - run_started_ts),
            "updated_at": float(time.time()),
        },
    )
    _append_engine_event(
        state_dir,
        {
            "ts": float(time.time()),
            "type": "run_end",
            "run_id": run_id,
            "out": str(out_path.resolve()),
            "elapsed_sec": float(time.time() - run_started_ts),
        },
    )


if __name__ == "__main__":
    main()
