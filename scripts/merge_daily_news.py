#!/usr/bin/env python

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger


def stable_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def iter_daily_news_files(daily_dir: Path) -> Iterable[Path]:
    if not daily_dir.exists():
        return []
    return sorted(daily_dir.glob("news_*.json"))


def load_json_any(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    return _TAG_RE.sub(" ", s or "").strip()


def extract_news_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ["news", "items", "articles", "data"]:
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def normalize_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = str(it.get("title") or "").strip()
    content = str(it.get("content") or it.get("description") or it.get("summary") or "").strip()
    url = str(it.get("url") or "").strip()
    source = str(it.get("source") or "").strip()
    market = str(it.get("market") or "").strip()
    published_at = str(it.get("published_at") or it.get("published") or it.get("date") or "").strip()

    if not title and not content:
        return None

    if content:
        content = strip_html(content)

    out = {
        "id": str(it.get("id") or "").strip(),
        "title": title,
        "content": content,
        "url": url,
        "source": source,
        "market": market,
        "published_at": published_at,
    }

    if not out["id"]:
        out["id"] = stable_key(title, url, published_at)[:40]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge data/daily/news_*.json into a single US raw news jsonl")
    parser.add_argument("--daily-dir", default="data/daily")
    parser.add_argument("--out", default="data/raw/news_us_raw.jsonl")
    parser.add_argument("--market", default="US")
    parser.add_argument("--min-content-len", type=int, default=30)
    parser.add_argument("--include-empty-content", action="store_true")
    parser.add_argument("--max", type=int, default=0)

    args = parser.parse_args()

    daily_dir = Path(args.daily_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_daily_news_files(daily_dir))
    if not files:
        raise SystemExit(f"No news_*.json found under: {daily_dir}")

    wanted_market = str(args.market).strip().upper()
    min_len = int(args.min_content_len)

    seen: set = set()
    merged: List[Dict[str, Any]] = []

    for fp in files:
        try:
            payload = load_json_any(fp)
        except Exception as e:
            logger.warning(f"Skip unreadable file: {fp} err={e}")
            continue

        items = extract_news_items(payload)
        for it in items:
            norm = normalize_item(it)
            if norm is None:
                continue

            mkt = str(norm.get("market") or "").strip().upper()
            if wanted_market and mkt != wanted_market:
                continue

            content = str(norm.get("content") or "")
            if not bool(args.include_empty_content):
                if len(content.strip()) < min_len:
                    continue

            title = str(norm.get("title") or "")
            url = str(norm.get("url") or "")
            published_at = str(norm.get("published_at") or "")

            key = stable_key(title, url, published_at)
            if key in seen:
                continue
            seen.add(key)

            merged.append(norm)

            if int(args.max) > 0 and len(merged) >= int(args.max):
                break

        if int(args.max) > 0 and len(merged) >= int(args.max):
            break

    with open(out_path, "w", encoding="utf-8") as f:
        for it in merged:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    logger.info(f"Saved: {out_path} items={len(merged)} sources={len(files)}")


if __name__ == "__main__":
    main()
