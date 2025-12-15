#!/usr/bin/env python

import argparse
import datetime as dt
import hashlib
import json
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import yaml
from loguru import logger


CONFIG_PATH = Path("config/sources.yaml")


def _now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def _to_iso_zoned(t: dt.datetime) -> str:
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.isoformat()


def _parse_entry_time(entry: Any) -> Optional[dt.datetime]:
    # feedparser may provide structured_time in published_parsed / updated_parsed
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return dt.datetime.fromtimestamp(time.mktime(st)).astimezone()
            except Exception:
                pass

    # fall back to RFC822 strings
    for key in ("published", "updated"):
        s = entry.get(key)
        if s:
            try:
                d = parsedate_to_datetime(s)
                return d.astimezone() if d.tzinfo else d.replace(tzinfo=dt.timezone.utc).astimezone()
            except Exception:
                pass

    return None


def _within_hours(ts: Optional[dt.datetime], *, hours: float) -> bool:
    if ts is None:
        return True
    delta = _now_local() - ts
    return delta.total_seconds() <= hours * 3600


def _extract_text(entry: Any) -> str:
    # Prefer summary / description, then content blocks
    for key in ("summary", "description"):
        v = entry.get(key)
        if v:
            return str(v)

    content = entry.get("content")
    if isinstance(content, list) and content:
        for c in content:
            if isinstance(c, dict) and c.get("value"):
                return str(c.get("value"))
    return ""


def _stable_id(*, title: str, url: str, source: str, market: str) -> str:
    basis = "|".join([(market or "").strip(), (source or "").strip(), (url or "").strip(), (title or "").strip()])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("sources.yaml must be a mapping")
    return data


def _normalize_sources(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    us_sources = cfg.get("us_sources") or []
    cn_sources = cfg.get("cn_sources") or []

    if isinstance(us_sources, list) and isinstance(cn_sources, list) and (us_sources or cn_sources):
        return us_sources, cn_sources

    # Backward-compatible fallback: use existing `sources:` list, treat all as US by default.
    legacy = cfg.get("sources") or []
    if not isinstance(legacy, list):
        return [], []

    us: List[Dict[str, Any]] = []
    for s in legacy:
        if isinstance(s, dict) and (s.get("type") == "rss") and s.get("url"):
            us.append({
                "name": s.get("name") or s.get("id") or "unknown",
                "url": s.get("url"),
                "weight": float(s.get("weight") or 1.0),
            })
    return us, []


def fetch_feed(url: str) -> List[Any]:
    # feedparser handles HTTP/redirects; keep it simple.
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", False):
        # bozo_exception can be informative, but don't fail pipeline
        logger.warning(f"Feed parse bozo for {url}: {getattr(feed, 'bozo_exception', None)}")
    return list(getattr(feed, "entries", []) or [])


def main():
    parser = argparse.ArgumentParser(description="Fetch daily RSS news for US+CN markets")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--hours", type=float, default=26.0, help="Lookback window in hours")
    parser.add_argument("--outdir", default="data/daily")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between feeds")
    parser.add_argument("--date", default=None, help="Override output date YYYY-MM-DD (default: today local)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    us_sources, cn_sources = _normalize_sources(cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_date = args.date or _now_local().strftime("%Y-%m-%d")
    out_path = outdir / f"news_{out_date}.json"

    collected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def process_sources(sources: Iterable[Dict[str, Any]], market: str):
        nonlocal collected, seen
        for src in sources:
            if not isinstance(src, dict):
                continue
            name = str(src.get("name") or "unknown")
            url = str(src.get("url") or "").strip()
            if not url:
                continue
            weight = float(src.get("weight") or 1.0)

            logger.info(f"Fetching {market} - {name}: {url}")
            try:
                entries = fetch_feed(url)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                continue

            logger.info(f"  -> entries={len(entries)}")
            for e in entries:
                title = str(getattr(e, "title", None) or e.get("title") or "").strip()
                link = str(getattr(e, "link", None) or e.get("link") or "").strip()
                content = _extract_text(e).strip()
                published_dt = _parse_entry_time(e)

                if not _within_hours(published_dt, hours=args.hours):
                    continue

                item_id = _stable_id(title=title, url=link, source=name, market=market)
                if item_id in seen:
                    continue
                seen.add(item_id)

                collected.append({
                    "id": item_id,
                    "title": title,
                    "content": content,
                    "url": link,
                    "source": name,
                    "market": market,
                    "weight": weight,
                    "published_at": _to_iso_zoned(published_dt) if published_dt else out_date,
                    "fetched_at": _to_iso_zoned(_now_local()),
                })

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

    process_sources(us_sources, "US")
    process_sources(cn_sources, "CN")

    logger.info(f"Total news collected (last {args.hours}h): {len(collected)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
