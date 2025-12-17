#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from loguru import logger
from tqdm import tqdm


def stable_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:24]


def load_existing_urls(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    urls: Set[str] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                u = str(obj.get("url") or "").strip()
                if u:
                    urls.add(u)
    return urls


def to_iso8601(ts: Any) -> str:
    try:
        t = int(ts)
        return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def fetch_article_text(url: str, timeout: int) -> Tuple[str, str]:
    """Return (text, title). Empty text on failure."""
    try:
        from newspaper import Article  # type: ignore

        a = Article(url)
        a.download()
        a.parse()
        text = (a.text or "").strip()
        title = (a.title or "").strip()
        return text, title
    except Exception as e:
        _ = timeout
        logger.debug(f"article_parse_failed url={url} err={e}")
        return "", ""


def iter_tickers(args_tickers: Optional[str]) -> List[str]:
    if args_tickers:
        raw = [x.strip().upper() for x in str(args_tickers).split(",") if x.strip()]
        return sorted(list(dict.fromkeys(raw)))

    return [
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "NVDA",
        "TSLA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "AMD",
        "INTC",
        "SMCI",
        "COIN",
        "MSTR",
        "JPM",
        "GS",
        "XOM",
        "CVX",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical US news with content: yfinance for links + newspaper3k for article text",
    )
    parser.add_argument("--out", default="data/raw/news_us_raw.jsonl")
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers, e.g. SPY,QQQ,NVDA. If empty, uses a default list.",
    )
    parser.add_argument("--max-items-per-ticker", type=int, default=20)
    parser.add_argument("--min-content-len", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--append", action="store_true")

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_urls = load_existing_urls(out_path)
    logger.info(f"Loaded existing urls: {len(existing_urls)}")

    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency yfinance. Install with: pip install yfinance"
        ) from e

    tickers = iter_tickers(args.tickers)
    max_items = int(max(1, args.max_items_per_ticker))
    min_len = int(max(0, args.min_content_len))

    new_items: List[Dict[str, Any]] = []

    pbar = tqdm(tickers, desc="tickers")
    for sym in pbar:
        pbar.set_description(f"{sym}")
        try:
            t = yf.Ticker(sym)
            news = t.news
        except Exception as e:
            logger.warning(f"yfinance_failed ticker={sym} err={e}")
            continue

        if not isinstance(news, list):
            continue

        taken = 0
        for it in news:
            if not isinstance(it, dict):
                continue

            url = str(it.get("link") or it.get("url") or "").strip()
            if not url:
                continue
            if url in existing_urls:
                continue

            title = str(it.get("title") or "").strip()
            publisher = str(it.get("publisher") or it.get("provider") or "").strip()
            published_at = to_iso8601(it.get("providerPublishTime"))

            text, parsed_title = fetch_article_text(url, timeout=int(args.timeout))
            if not title and parsed_title:
                title = parsed_title

            if len(text) < min_len:
                continue

            obj: Dict[str, Any] = {
                "id": stable_id(url, title, published_at),
                "title": title,
                "content": text,
                "published_at": published_at,
                "source": publisher,
                "url": url,
                "market": "US",
                "related_ticker": sym,
            }

            new_items.append(obj)
            existing_urls.add(url)
            taken += 1

            if args.sleep and float(args.sleep) > 0:
                time.sleep(float(args.sleep))

            if taken >= max_items:
                break

    mode = "a" if bool(args.append) and out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for obj in new_items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {out_path} new_items={len(new_items)}")


if __name__ == "__main__":
    main()
