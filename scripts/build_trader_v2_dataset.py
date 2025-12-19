#!/usr/bin/env python
"""Build Trader v2 SFT dataset from Teacher CoT corrections.

Converts cot_mistakes_*.jsonl (Teacher format) to SFT conversations format,
with optional replay buffer mixing from existing trader data to prevent
catastrophic forgetting.

Usage:
  .\\venv311\\Scripts\\python.exe scripts\\build_trader_v2_dataset.py \\
      --cot data/finetune/cot_mistakes_100_v4.jsonl \\
      --out-dir data/finetune \\
      --val-ratio 0.2

With replay buffer:
  .\\venv311\\Scripts\\python.exe scripts\\build_trader_v2_dataset.py \\
      --cot data/finetune/cot_mistakes_100_v4.jsonl \\
      --replay data/finetune/trader_stock_sft_v1_plus_news.json \\
      --replay-ratio 0.5 \\
      --out-dir data/finetune \\
      --val-ratio 0.2
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


SYSTEM_PROMPT_TRADER_V2 = """You are a strictly compliant trading signal generator.
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
- if provided news context is irrelevant, explicitly say so in the trace
"""


def _format_technical(tech: Dict[str, Any]) -> str:
    if not isinstance(tech, dict):
        return ""
    lines = []
    for k in [
        "close",
        "price_vs_ma20",
        "price_vs_ma200",
        "trend_alignment",
        "breakout_20d_high",
        "breakdown_20d_low",
        "return_5d",
        "return_21d",
        "return_63d",
        "volatility_20d",
        "vol_ratio",
        "drawdown",
        "max_drawdown_20d",
        "max_drawdown_60d",
    ]:
        v = tech.get(k)
        if v is not None:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _format_signal(sig: Dict[str, Any]) -> str:
    if not isinstance(sig, dict):
        return ""
    lines = []
    for k in ["strength", "trend", "momentum", "ma_cross", "breakout", "composite"]:
        v = sig.get(k)
        if v is not None:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _format_market_regime(mr: Dict[str, Any]) -> str:
    if not isinstance(mr, dict):
        return ""
    regime = mr.get("regime", "")
    score = mr.get("score", "")
    return f"regime: {regime}, score: {score}"


def _format_news_context(news_top: List[Dict[str, Any]]) -> str:
    if not news_top:
        return "No news context available."
    lines = ["News Context:"]
    for i, n in enumerate(news_top[:3], 1):
        title = str(n.get("title") or "").strip()
        source = str(n.get("source") or "").strip()
        summary = str(n.get("summary") or "").strip()
        if title:
            line = f"{i}. [{source}] {title}"
            if summary:
                line += f" - {summary}"
            lines.append(line)
    if len(lines) == 1:
        return "No news context available."
    return "\n".join(lines)


def _build_user_prompt(rec: Dict[str, Any]) -> str:
    ticker = rec.get("ticker", "")
    date_str = rec.get("date", "")
    features = rec.get("features") if isinstance(rec.get("features"), dict) else {}
    news_top = rec.get("news_top") if isinstance(rec.get("news_top"), list) else []

    tech = features.get("technical") if isinstance(features.get("technical"), dict) else {}
    sig = features.get("signal") if isinstance(features.get("signal"), dict) else {}
    mr = features.get("market_regime") if isinstance(features.get("market_regime"), dict) else {}

    parts = [
        f"Ticker: {ticker}",
        f"Date: {date_str}",
        "",
        "Technical Snapshot:",
        _format_technical(tech),
        "",
        "Signal:",
        _format_signal(sig),
        "",
        "Market Regime:",
        _format_market_regime(mr),
        "",
        _format_news_context(news_top),
        "",
        "Decide BUY/SELL/HOLD for the next 5 days.",
    ]
    return "\n".join(parts)


def _build_assistant_output(corrected: Dict[str, Any], ticker: str) -> str:
    decision = str(corrected.get("decision") or "HOLD").strip().upper()
    analysis = str(corrected.get("analysis") or "").strip()
    rt = corrected.get("reasoning_trace")
    if not isinstance(rt, list):
        rt = []
    rt = [str(x).strip() for x in rt if str(x).strip()]
    while len(rt) < 3:
        rt.append("No trace provided.")
    rt = rt[:3]

    obj = {
        "decision": decision,
        "ticker": ticker,
        "analysis": analysis,
        "reasoning_trace": rt,
    }
    return json.dumps(obj, ensure_ascii=False)


def convert_cot_to_sft(cot_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(cot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            corrected = rec.get("corrected") if isinstance(rec.get("corrected"), dict) else {}
            if not corrected:
                continue

            ticker = rec.get("ticker", "")
            user_prompt = _build_user_prompt(rec)
            assistant_output = _build_assistant_output(corrected, ticker)

            item = {
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT_TRADER_V2},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ],
                "meta": {
                    "source": "cot_teacher_v4",
                    "date": rec.get("date"),
                    "ticker": ticker,
                    "original_decision": rec.get("original_decision"),
                    "corrected_decision": corrected.get("decision"),
                },
            }
            samples.append(item)

    return samples


def _augment_assistant_with_reasoning_trace(content: str) -> str:
    """Add reasoning_trace to old-format assistant outputs."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "reasoning_trace" not in obj:
            decision = obj.get("decision", "HOLD")
            analysis = obj.get("analysis", "")
            obj["reasoning_trace"] = [
                f"1. Technical analysis supports {decision} decision.",
                f"2. {analysis}" if analysis else "2. Market conditions align with signal.",
                "3. Risk-reward profile acceptable for position.",
            ]
            return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return content


def load_replay_buffer(replay_path: Path, max_samples: int = 0, adapt_format: bool = True) -> List[Dict[str, Any]]:
    if not replay_path.exists():
        return []

    try:
        data = json.loads(replay_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    # Normalize format: convert from/value to role/content if needed
    normalized: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        convs = item.get("conversations")
        if not isinstance(convs, list):
            continue

        new_convs = []
        for msg in convs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value")
            if role and content:
                # Adapt format for v2 compatibility
                if adapt_format:
                    if role == "system":
                        content = SYSTEM_PROMPT_TRADER_V2
                    elif role == "assistant":
                        content = _augment_assistant_with_reasoning_trace(content)
                new_convs.append({"role": str(role), "content": str(content)})

        if new_convs:
            new_item = {"conversations": new_convs}
            if "meta" in item:
                new_item["meta"] = item["meta"]
            else:
                new_item["meta"] = {"source": "replay_buffer"}
            normalized.append(new_item)

    if max_samples > 0 and len(normalized) > max_samples:
        normalized = random.sample(normalized, max_samples)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Trader v2 SFT dataset from Teacher CoT")
    parser.add_argument("--cot", required=True, help="Input CoT JSONL (e.g. cot_mistakes_100_v4.jsonl)")
    parser.add_argument("--replay", default="", help="Replay buffer JSON for mixing (optional)")
    parser.add_argument("--replay-ratio", type=float, default=0.5, help="Ratio of replay samples to CoT samples")
    parser.add_argument("--out-dir", default="data/finetune", help="Output directory")
    parser.add_argument("--out-prefix", default="trader_v2", help="Output file prefix")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    cot_path = Path(args.cot)
    if not cot_path.exists():
        raise SystemExit(f"CoT file not found: {cot_path}")

    print(f"Loading CoT data from {cot_path}")
    cot_samples = convert_cot_to_sft(cot_path)
    print(f"  CoT samples: {len(cot_samples)}")

    replay_samples: List[Dict[str, Any]] = []
    if args.replay:
        replay_path = Path(args.replay)
        if replay_path.exists():
            max_replay = int(len(cot_samples) * args.replay_ratio)
            print(f"Loading replay buffer from {replay_path} (max {max_replay})")
            replay_samples = load_replay_buffer(replay_path, max_samples=max_replay)
            print(f"  Replay samples: {len(replay_samples)}")

    all_samples = cot_samples + replay_samples
    random.shuffle(all_samples)

    val_n = int(len(all_samples) * args.val_ratio)
    val_n = max(1, val_n) if len(all_samples) >= 2 else 0

    val_samples = all_samples[:val_n]
    train_samples = all_samples[val_n:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{args.out_prefix}_train.json"
    val_path = out_dir / f"{args.out_prefix}_val.json"

    train_path.write_text(json.dumps(train_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(val_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nOutput:")
    print(f"  Train: {train_path} ({len(train_samples)} samples)")
    print(f"  Val:   {val_path} ({len(val_samples)} samples)")

    # Print sample preview
    if train_samples:
        print("\n--- Sample Preview (first train sample) ---")
        sample = train_samples[0]
        for msg in sample["conversations"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                print(f"[{role}]")
                print(content)
            elif role == "user":
                print(f"[{role}]")
                preview = content[:500] + "..." if len(content) > 500 else content
                print(preview)


if __name__ == "__main__":
    main()
