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


def parse_yahoo_news_item(it: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Return (url, title, publisher, published_at). Empty url if not found."""
    c = it.get("content") if isinstance(it.get("content"), dict) else {}
    if not isinstance(c, dict):
        c = {}

    def _get_url(d: Dict[str, Any], k: str) -> str:
        v = d.get(k)
        if isinstance(v, dict):
            return str(v.get("url") or "").strip()
        return str(v or "").strip()

    url = _get_url(c, "canonicalUrl") or _get_url(c, "clickThroughUrl") or str(it.get("link") or it.get("url") or "").strip()
    title = str(c.get("title") or it.get("title") or "").strip()

    provider = c.get("provider") if isinstance(c.get("provider"), dict) else {}
    if not isinstance(provider, dict):
        provider = {}
    publisher = str(provider.get("displayName") or c.get("publisher") or it.get("publisher") or "").strip()

    pub = str(c.get("pubDate") or c.get("displayTime") or "").strip()
    published_at = pub
    if not published_at:
        published_at = to_iso8601(it.get("providerPublishTime"))

    return url, title, publisher, published_at


def fetch_article_text(url: str, timeout: int) -> Tuple[str, str]:
    """Return (text, title). Empty text on failure."""
    try:
        from newspaper import Article, Config  # type: ignore

        config = Config()
        config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        config.request_timeout = int(timeout)
        a = Article(url, config=config)
        a.download()
        a.parse()
        text = (a.text or "").strip()
        title = (a.title or "").strip()
        return text, title
    except Exception as e:
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
        "DIA",
        "TLT",
        "GLD",
        "SLV",
        "USO",
        "HYG",
        "EEM",
        "UNH",
        "JNJ",
        "LLY",
        "MRK",
        "ABBV",
        "PFE",
        "TMO",
        "DHR",
        "BMY",
        "AMGN",
        "JPM",
        "BAC",
        "WFC",
        "C",
        "GS",
        "MS",
        "BLK",
        "AXP",
        "V",
        "MA",
        "PYPL",
        "WMT",
        "COST",
        "PG",
        "KO",
        "PEP",
        "MCD",
        "SBUX",
        "NKE",
        "HD",
        "LOW",
        "TGT",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "CAT",
        "DE",
        "HON",
        "GE",
        "UNP",
        "UPS",
        "BA",
        "LMT",
        "RTX",
        "AVGO",
        "ORCL",
        "CRM",
        "ADBE",
        "CSCO",
        "ACN",
        "IBM",
        "QCOM",
        "TXN",
        "INTC",
        "AMD",
        "NFLX",
        "DIS",
        "CMCSA",
        "TMUS",
        "VZ",
        "T",
        "BABA",
        "PDD",
        "JD",
        "NIO",
        "COIN",
        "MSTR",
        "PLTR",
        "UBER",
        "ABNB",
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

    links_seen = 0
    skipped_existing = 0
    parse_failed = 0
    too_short = 0
    saved = 0

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

            url, title, publisher, published_at = parse_yahoo_news_item(it)
            if not url:
                continue
            if url in existing_urls:
                skipped_existing += 1
                continue

            links_seen += 1

            text, parsed_title = fetch_article_text(url, timeout=int(args.timeout))
            if not title and parsed_title:
                title = parsed_title

            if len(text) < min_len:
                if len(text) == 0:
                    parse_failed += 1
                else:
                    too_short += 1
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
            saved += 1

            if args.sleep and float(args.sleep) > 0:
                time.sleep(float(args.sleep))

            if taken >= max_items:
                break

    mode = "a" if bool(args.append) and out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for obj in new_items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info(
        "summary "
        f"tickers={len(tickers)} links_seen={links_seen} saved={saved} "
        f"skipped_existing={skipped_existing} parse_failed={parse_failed} too_short={too_short} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()
