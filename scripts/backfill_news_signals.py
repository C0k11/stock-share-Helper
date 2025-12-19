#!/usr/bin/env python

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_daily_inference import (  # noqa: E402
    build_messages_news,
    post_process_cn_signals,
    prepare_signals_for_save,
    try_parse_json,
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    out: List[Dict[str, Any]] = []
    for it in data:
        if isinstance(it, dict):
            out.append(it)
    return out


def _extract_trader_dates(trader_path: Path) -> Set[str]:
    dataset = _load_json_list(trader_path)
    out: Set[str] = set()

    for item in dataset:
        convs = item.get("conversations")
        if not isinstance(convs, list) or len(convs) < 2:
            continue
        user = convs[1]
        if not isinstance(user, dict):
            continue
        txt = str(user.get("value") or user.get("content") or "")

        marker = " on "
        pos = txt.find(marker)
        if pos < 0:
            continue
        day = txt[pos + len(marker) : pos + len(marker) + 10]
        if len(day) == 10 and day[4] == "-" and day[7] == "-":
            out.add(day)

    return out


def _normalize_market(market: Any) -> str:
    m = str(market or "US").strip().upper()
    if m not in ("US", "CN"):
        return "US"
    return m


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _spool_news_by_day(
    *,
    news_jsonl: Path,
    needed_days: Set[str],
    spool_dir: Path,
    max_open_files: int,
) -> Tuple[int, Dict[str, int]]:
    spool_dir.mkdir(parents=True, exist_ok=True)

    handles: "OrderedDict[str, Any]" = OrderedDict()
    counts: Dict[str, int] = {}
    total = 0

    def get_handle(day: str):
        nonlocal handles
        if day in handles:
            fh = handles.pop(day)
            handles[day] = fh
            return fh

        if len(handles) >= int(max_open_files):
            _old_day, old_fh = handles.popitem(last=False)
            try:
                old_fh.close()
            except Exception:
                pass

        fp = spool_dir / f"news_{day}.jsonl"
        fh = open(fp, "a", encoding="utf-8")
        handles[day] = fh
        return fh

    for it in _iter_jsonl(news_jsonl):
        pub = _safe_str(it.get("published_at"))
        if len(pub) < 10:
            continue
        day = pub[:10]
        if day not in needed_days:
            continue

        fh = get_handle(day)
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        total += 1
        counts[day] = counts.get(day, 0) + 1

    for _day, fh in list(handles.items()):
        try:
            fh.close()
        except Exception:
            pass

    return total, counts


def _load_day_news(spool_dir: Path, day: str) -> List[Dict[str, Any]]:
    fp = spool_dir / f"news_{day}.jsonl"
    if not fp.exists():
        return []
    return list(_iter_jsonl(fp))


def _infer_day(
    *,
    day: str,
    items: List[Dict[str, Any]],
    out_path: Path,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    deterministic: bool,
    max_input_chars: int,
    min_content_chars: int,
    cn_hype_cap: int,
) -> Dict[str, Any]:
    import torch

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": not bool(deterministic),
    }
    if not deterministic:
        gen_kwargs.update({"temperature": 0.7, "top_p": 0.9})

    signals: List[Dict[str, Any]] = []
    et_counts: Dict[str, int] = {}
    market_counts: Dict[str, int] = {"US": 0, "CN": 0}
    skipped_noise = 0
    skipped_empty_content = 0

    bsz = max(1, int(batch_size))

    for start in range(0, len(items), bsz):
        batch = items[start : start + bsz]

        prompts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for it in batch:
            title = _safe_str(it.get("title"))
            content = _safe_str(it.get("content"))

            max_chars = max(0, int(max_input_chars))
            min_chars = max(0, int(min_content_chars))

            if min_chars and len(content) < min_chars:
                skipped_empty_content += 1
                market = _normalize_market(it.get("market"))
                market_counts[market] = market_counts.get(market, 0) + 1

                meta = {
                    "id": it.get("id"),
                    "market": market,
                    "source": _safe_str(it.get("source")),
                    "url": _safe_str(it.get("url")),
                    "published_at": it.get("published_at"),
                    "title": title,
                    "input_chars": len(content),
                }
                signals.append({**meta, "parse_ok": False, "signal": None, "raw_json": None})
                continue

            if max_chars and len(content) > max_chars:
                content = content[:max_chars] + "...(truncated)"

            market = _normalize_market(it.get("market"))

            messages = build_messages_news(market=market, title=title, content=content)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            metas.append(
                {
                    "id": it.get("id"),
                    "market": market,
                    "source": _safe_str(it.get("source")),
                    "url": _safe_str(it.get("url")),
                    "published_at": it.get("published_at"),
                    "title": title,
                    "input_chars": len(content),
                }
            )

        if not prompts:
            continue

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for i in range(len(metas)):
            pl = int(prompt_lens[i])
            out_text = tokenizer.decode(outputs[i][pl:], skip_special_tokens=True)
            parse_ok, obj, raw_json = try_parse_json(out_text)

            meta = metas[i]
            market = meta["market"]
            market_counts[market] = market_counts.get(market, 0) + 1

            record = {**meta, "parse_ok": parse_ok, "signal": obj, "raw_json": raw_json}
            record = post_process_cn_signals(record)

            sig = record.get("signal") if isinstance(record.get("signal"), dict) else None
            if parse_ok and sig:
                et = _safe_str(sig.get("event_type"))
                if et == "noise":
                    skipped_noise += 1
                    record["parse_ok"] = False
                    record["signal"] = None
                else:
                    et_counts[et] = et_counts.get(et, 0) + 1

            signals.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prepare_signals_for_save(signals, int(cn_hype_cap)), f, ensure_ascii=False, indent=2)

    return {
        "date": day,
        "out": str(out_path),
        "input_items": int(len(items)),
        "saved_items": int(len(signals)),
        "parse_ok": int(sum(1 for x in signals if x.get("parse_ok"))),
        "skipped_noise": int(skipped_noise),
        "skipped_empty_content": int(skipped_empty_content),
        "market_counts": market_counts,
        "top_event_types": sorted(et_counts.items(), key=lambda x: x[1], reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill daily signals_YYYY-MM-DD.json from historical news JSONL")
    parser.add_argument("--news-data", default="data/raw/news_us_raw.jsonl")
    parser.add_argument("--trader-data", default="data/finetune/trader_stock_sft_v1.json")
    parser.add_argument("--out-dir", default="data/daily")
    parser.add_argument("--spool-dir", default="data/tmp/news_by_day")

    parser.add_argument("--base-model", dest="model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model", dest="model", help="Alias of --base-model")
    parser.add_argument("--lora", default="models/news_final_3b_v1_1_noise_killer_retry2/lora_weights")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--max-input-chars", type=int, default=6000)
    parser.add_argument("--min-content-chars", type=int, default=50)
    parser.add_argument("--cn-hype-cap", type=int, default=30)

    parser.add_argument("--max-open-files", type=int, default=32)
    parser.add_argument("--limit-days", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    news_path = Path(args.news_data)
    trader_path = Path(args.trader_data)
    out_dir = Path(args.out_dir)
    spool_dir = Path(args.spool_dir)

    if not news_path.exists():
        raise SystemExit(f"Missing news jsonl: {news_path}")
    if not trader_path.exists():
        raise SystemExit(f"Missing trader dataset: {trader_path}")

    logger.info(f"Scanning Trader dates from: {trader_path}")
    needed_days = _extract_trader_dates(trader_path)
    days_sorted = sorted(needed_days)

    if int(args.limit_days) > 0:
        days_sorted = days_sorted[: int(args.limit_days)]
        needed_days = set(days_sorted)

    logger.info(f"Trader days: {len(needed_days)}")

    logger.info(f"Spooling news by day from: {news_path}")
    total, counts = _spool_news_by_day(
        news_jsonl=news_path,
        needed_days=needed_days,
        spool_dir=spool_dir,
        max_open_files=int(args.max_open_files),
    )
    logger.info(f"Spool done: matched_items={total} matched_days={len(counts)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    if bool(args.load_in_4bit):
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    logger.info(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(str(args.model), **model_kwargs)

    if bool(args.use_lora):
        from peft import PeftModel

        lora_path = Path(args.lora)
        if not lora_path.exists():
            raise SystemExit(f"Missing LoRA weights: {lora_path}")
        logger.info(f"Loading LoRA weights: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))

    model.eval()

    processed = 0
    skipped_existing = 0

    for day in days_sorted:
        out_path = out_dir / f"signals_{day}.json"
        if out_path.exists() and (not bool(args.overwrite)):
            skipped_existing += 1
            continue

        items = _load_day_news(spool_dir, day)
        stats = _infer_day(
            day=day,
            items=items,
            out_path=out_path,
            model=model,
            tokenizer=tokenizer,
            batch_size=int(args.batch_size),
            max_new_tokens=int(args.max_new_tokens),
            deterministic=bool(args.deterministic),
            max_input_chars=int(args.max_input_chars),
            min_content_chars=int(args.min_content_chars),
            cn_hype_cap=int(args.cn_hype_cap),
        )

        processed += 1
        if processed % 10 == 0:
            logger.info(
                f"days_done={processed}/{len(days_sorted)} skipped_existing={skipped_existing} "
                f"last_day={day} last_stats={stats}"
            )

    logger.info(
        f"Backfill finished: days_total={len(days_sorted)} days_processed={processed} skipped_existing={skipped_existing} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
