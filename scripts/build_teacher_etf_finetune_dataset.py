#!/usr/bin/env python

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _load_jsonl(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _compact_input(etf_features: Dict[str, Any], risk_watch: Dict[str, Any]) -> str:
    payload = {
        "etf_features": etf_features,
        "risk_watch": risk_watch,
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_context(*, date: str, symbol: str, variant_index: int, etf_features: Dict[str, Any], risk_watch: Dict[str, Any]) -> str:
    return (
        "你是一个专业的交易与风险管理助手。你会得到ETF/指数的结构化特征与可选的风险新闻摘要。\n"
        "你的任务：在风险约束下给出一个可执行的仓位动作建议。\n"
        "你必须输出严格JSON（不要markdown，不要额外文字，不要多余字段）。\n\n"
        f"date={date}\n"
        f"symbol={symbol}\n"
        f"variant_index={variant_index}\n\n"
        "INPUT_JSON=" + _compact_input(etf_features, risk_watch)
    )


def _build_output_json(label: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": label.get("action"),
        "target_position": label.get("target_position"),
        "risk_notes": label.get("risk_notes"),
        "rationale": label.get("rationale"),
    }


def _split_train_val(data: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(int(seed))
    items = list(data)
    rng.shuffle(items)
    if not items:
        return [], []
    val_n = int(len(items) * float(val_ratio))
    if len(items) >= 2:
        val_n = max(1, val_n)
    else:
        val_n = 0
    val = items[:val_n] if val_n > 0 else []
    train = items[val_n:] if val_n > 0 else items
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Build finetune conversations dataset from ETF teacher JSONL")
    parser.add_argument("--in", dest="in_path", default="data/finetune/teacher_etf/teacher_etf.jsonl")
    parser.add_argument("--outdir", default="data/finetune/teacher_etf")
    parser.add_argument("--train-name", default="train.json")
    parser.add_argument("--val-name", default="val.json")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dedup", action="store_true")
    args = parser.parse_args()

    from src.llm.finetune.dataset import FineTuneDataset

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(in_path, max_samples=int(args.max_samples))
    logger.info(f"Loaded teacher rows: {len(rows)}")

    ds = FineTuneDataset(data_path=str(outdir))

    seen = set()
    kept = 0
    skipped = 0

    for r in rows:
        sid = str(r.get("id") or "")
        if args.dedup:
            if sid and sid in seen:
                skipped += 1
                continue
            if sid:
                seen.add(sid)

        output = r.get("output") if isinstance(r.get("output"), dict) else {}
        label = output.get("label") if isinstance(output.get("label"), dict) else None
        if not isinstance(label, dict):
            skipped += 1
            continue

        date = str(r.get("date") or "")
        symbol = str(r.get("symbol") or "")
        vi = int(r.get("variant_index") or 0)

        inp = r.get("input") if isinstance(r.get("input"), dict) else {}
        etf_features = inp.get("etf_features") if isinstance(inp.get("etf_features"), dict) else {}
        risk_watch = inp.get("risk_watch") if isinstance(inp.get("risk_watch"), dict) else {}

        context = _build_context(date=date, symbol=symbol, variant_index=vi, etf_features=etf_features, risk_watch=risk_watch)
        out_json = _build_output_json(label)

        meta = {
            "id": sid,
            "date": date,
            "symbol": symbol,
            "variant_index": vi,
            "teacher_model": (r.get("teacher") or {}).get("model") if isinstance(r.get("teacher"), dict) else "",
        }

        ds.add_trading_case_sample(context=context, output_json=out_json, meta=meta)
        kept += 1

    convs = ds.to_conversation_format()
    logger.info(f"Built conversations: kept={kept} skipped={skipped} total={len(convs)}")

    train, val = _split_train_val(convs, float(args.val_ratio), int(args.seed))
    train_path = outdir / str(args.train_name)
    val_path = outdir / str(args.val_name)

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved train={len(train)} -> {train_path}")
    logger.info(f"Saved val={len(val)} -> {val_path}")


if __name__ == "__main__":
    main()
