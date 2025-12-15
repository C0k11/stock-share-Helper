#!/usr/bin/env python

import argparse
import datetime as dt
import hashlib
import json
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests
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


def _decode_http_content(content: bytes) -> str:
    if not content:
        return ""
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return content.decode(enc)
        except Exception:
            pass
    return content.decode("utf-8", errors="replace")


def _mojibake_penalty(s: str) -> int:
    if not s:
        return 0
    bad = 0
    bad += s.count("\ufffd") * 10
    for ch in ("鍗", "鈥", "锟"):
        bad += s.count(ch) * 3
    return bad


def _repair_mojibake(s: str) -> str:
    if not s:
        return s
    s = str(s)
    base_penalty = _mojibake_penalty(s)
    best = s
    best_penalty = base_penalty

    for enc in ("gb18030", "gbk"):
        try:
            repaired = s.encode(enc).decode("utf-8", errors="replace")
        except Exception:
            continue
        p = _mojibake_penalty(repaired)
        if p < best_penalty:
            best = repaired
            best_penalty = p

    return best


def _parse_dt_guess(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None

    # unix timestamp seconds
    if s.isdigit():
        try:
            return dt.datetime.fromtimestamp(int(s)).astimezone()
        except Exception:
            return None

    # common CN formats
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            d = dt.datetime.strptime(s, fmt)
            return d.replace(tzinfo=_now_local().tzinfo)
        except Exception:
            pass

    # last resort: try RFC822
    try:
        d = parsedate_to_datetime(s)
        return d.astimezone() if d.tzinfo else d.replace(tzinfo=dt.timezone.utc).astimezone()
    except Exception:
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
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
        "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.6,zh;q=0.4",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
    except Exception as e:
        logger.warning(f"HTTP fetch failed for {url}: {e}")
        return []

    ct = (resp.headers.get("content-type") or "").lower()
    if resp.status_code >= 400:
        logger.warning(f"HTTP {resp.status_code} for {url} content_type={ct}")
        return []

    content = resp.content or b""
    # Some feeds include BOM or stray bytes before XML which can trigger "invalid token".
    content = content.lstrip(b"\xef\xbb\xbf")
    for marker in (b"<?xml", b"<rss", b"<feed"):
        idx = content.find(marker)
        if idx > 0:
            content = content[idx:]
            break

    feed = feedparser.parse(content)
    if getattr(feed, "bozo", False):
        logger.warning(
            f"Feed parse bozo for {url}: {getattr(feed, 'bozo_exception', None)} status={resp.status_code} content_type={ct}"
        )

    return list(getattr(feed, "entries", []) or [])


def _cn_fetch_sina_roll(*, page: int, per_page: int) -> List[Dict[str, Any]]:
    # JSONP-ish API used in scripts/fetch_cn_news.py (more stable than RSS in many envs)
    base = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516&k=&num={num}&page={page}&r=0.1&callback="
    url = base.format(num=int(per_page), page=int(page))
    try:
        resp = requests.get(url, timeout=15)
        text = _decode_http_content(resp.content)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group())
        items = data.get("result", {}).get("data", []) or []
    except Exception as e:
        logger.warning(f"CN sina roll fetch failed: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for it in items:
        title = _repair_mojibake((it.get("title") or "").strip())
        content = _repair_mojibake((it.get("intro") or it.get("summary") or "").strip())
        pub = it.get("ctime") or it.get("create_time") or ""
        if pub and str(pub).isdigit():
            pub = str(pub)
        out.append(
            {
                "title": title,
                "content": content,
                "url": it.get("url") or "",
                "source": "sina_api",
                "published_dt": _parse_dt_guess(str(pub)),
            }
        )
    return out


def _cn_fetch_eastmoney_kuaixun(*, page: int, per_page: int) -> List[Dict[str, Any]]:
    # Example: https://newsapi.eastmoney.com/kuaixun/v1/getlist_101_ajaxResult_50_1_.html
    url = f"https://newsapi.eastmoney.com/kuaixun/v1/getlist_101_ajaxResult_{int(per_page)}_{int(page)}_.html"
    try:
        resp = requests.get(url, timeout=15)
        text = _decode_http_content(resp.content)
        m = re.search(r"var defined[^=]*=\s*(\[.*?\]);", text, re.DOTALL)
        if not m:
            m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        raw = m.group(1) if m.lastindex else m.group()
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
    except Exception as e:
        logger.warning(f"CN eastmoney fetch failed: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for it in items:
        title = _repair_mojibake((it.get("title") or "").strip())
        content = _repair_mojibake((it.get("digest") or it.get("content") or "").strip())
        pub = (it.get("showtime") or it.get("time") or "").strip()
        out.append(
            {
                "title": title,
                "content": content,
                "url": it.get("url") or "",
                "source": "eastmoney_api",
                "published_dt": _parse_dt_guess(pub),
            }
        )
    return out


def _cn_fetch_cls_telegraph(*, limit: int) -> List[Dict[str, Any]]:
    url = "https://www.cls.cn/nodeapi/updateTelegraphList"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.cls.cn/telegraph",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        items = data.get("data", {}).get("roll_data", []) or []
    except Exception as e:
        logger.warning(f"CN cls telegraph fetch failed: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for it in items[: int(limit)]:
        title = _repair_mojibake((it.get("title") or it.get("brief") or "").strip())
        content = _repair_mojibake((it.get("content") or it.get("brief") or "").strip())
        pub = it.get("ctime") or ""
        pub_dt = _parse_dt_guess(str(pub))
        out.append(
            {
                "title": title,
                "content": content,
                "url": f"https://www.cls.cn/detail/{it.get('id', '')}" if it.get("id") else "",
                "source": "cls_telegraph",
                "published_dt": pub_dt,
            }
        )
    return out


def fetch_cn_fallback_items(*, hours: float, max_pages: int = 3, per_page: int = 50) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # CLS first (usually stable)
    items.extend(_cn_fetch_cls_telegraph(limit=120))

    for page in range(1, int(max_pages) + 1):
        items.extend(_cn_fetch_sina_roll(page=page, per_page=per_page))

    for page in range(1, int(max_pages) + 1):
        items.extend(_cn_fetch_eastmoney_kuaixun(page=page, per_page=per_page))

    filtered: List[Dict[str, Any]] = []
    for it in items:
        pub_dt = it.get("published_dt")
        if not _within_hours(pub_dt, hours=hours):
            continue
        filtered.append(it)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Fetch daily RSS news for US+CN markets")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--hours", type=float, default=26.0, help="Lookback window in hours")
    parser.add_argument("--outdir", default="data/daily")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between feeds")
    parser.add_argument("--date", default=None, help="Override output date YYYY-MM-DD (default: today local)")
    parser.add_argument(
        "--health-out",
        default=None,
        help="Optional path to write health JSON; use 'auto' to write to outdir/health_YYYY-MM-DD.json",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    us_sources, cn_sources = _normalize_sources(cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_date = args.date or _now_local().strftime("%Y-%m-%d")
    out_path = outdir / f"news_{out_date}.json"

    health: Dict[str, Any] = {
        "date": out_date,
        "hours": float(args.hours),
        "generated_at": _to_iso_zoned(_now_local()),
        "by_source": {},
        "cn": {
            "rss_items": 0,
            "fallback_used": False,
            "fallback_candidates": 0,
            "fallback_added": 0,
        },
        "totals": {
            "rss_added": 0,
            "rss_errors": 0,
            "fallback_added": 0,
            "final_total": 0,
        },
    }

    collected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    cn_seen = 0

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

            key = f"{market}:{name}"
            if key not in health["by_source"]:
                health["by_source"][key] = {
                    "market": market,
                    "source": name,
                    "rss_entries": 0,
                    "rss_in_window": 0,
                    "rss_added": 0,
                    "rss_errors": 0,
                    "fallback_candidates": 0,
                    "fallback_added": 0,
                }

            logger.info(f"Fetching {market} - {name}: {url}")
            try:
                entries = fetch_feed(url)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                health["by_source"][key]["rss_errors"] += 1
                health["totals"]["rss_errors"] += 1
                continue

            logger.info(f"  -> entries={len(entries)}")
            health["by_source"][key]["rss_entries"] += int(len(entries))
            for e in entries:
                title = _repair_mojibake(str(getattr(e, "title", None) or e.get("title") or "").strip())
                link = str(getattr(e, "link", None) or e.get("link") or "").strip()
                content = _repair_mojibake(_extract_text(e).strip())
                published_dt = _parse_entry_time(e)

                if not _within_hours(published_dt, hours=args.hours):
                    continue

                health["by_source"][key]["rss_in_window"] += 1

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

                health["by_source"][key]["rss_added"] += 1
                health["totals"]["rss_added"] += 1

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

    process_sources(us_sources, "US")
    process_sources(cn_sources, "CN")

    # CN fallback: if RSS sources are blocked/return HTML, use stable CN JSON APIs.
    cn_seen = sum(1 for x in collected if x.get("market") == "CN")
    health["cn"]["rss_items"] = int(cn_seen)
    if cn_seen == 0:
        logger.warning("CN RSS returned 0 items; falling back to CN JSON APIs (sina/eastmoney/cls)")
        health["cn"]["fallback_used"] = True
        cn_items = fetch_cn_fallback_items(hours=args.hours)
        logger.info(f"CN fallback candidates within window: {len(cn_items)}")
        health["cn"]["fallback_candidates"] = int(len(cn_items))
        for it in cn_items:
            title = _repair_mojibake((it.get("title") or "").strip())
            link = (it.get("url") or "").strip()
            content = _repair_mojibake((it.get("content") or "").strip())
            src = it.get("source") or "cn_api"
            pub_dt = it.get("published_dt")

            item_id = _stable_id(title=title, url=link, source=str(src), market="CN")
            if item_id in seen:
                continue
            seen.add(item_id)
            collected.append(
                {
                    "id": item_id,
                    "title": title,
                    "content": content,
                    "url": link,
                    "source": str(src),
                    "market": "CN",
                    "weight": 1.0,
                    "published_at": _to_iso_zoned(pub_dt) if isinstance(pub_dt, dt.datetime) else out_date,
                    "fetched_at": _to_iso_zoned(_now_local()),
                }
            )

            health["cn"]["fallback_added"] += 1
            health["totals"]["fallback_added"] += 1
            key = f"CN:{str(src)}"
            if key not in health["by_source"]:
                health["by_source"][key] = {
                    "market": "CN",
                    "source": str(src),
                    "rss_entries": 0,
                    "rss_in_window": 0,
                    "rss_added": 0,
                    "rss_errors": 0,
                    "fallback_candidates": 0,
                    "fallback_added": 0,
                }
            health["by_source"][key]["fallback_candidates"] += 1
            health["by_source"][key]["fallback_added"] += 1

    logger.info(f"Total news collected (last {args.hours}h): {len(collected)}")
    health["totals"]["final_total"] = int(len(collected))

    logger.info("News health summary:")
    zero_sources: List[str] = []
    for k, st in sorted(health["by_source"].items(), key=lambda x: (x[1].get("market"), -int(x[1].get("rss_added", 0) + x[1].get("fallback_added", 0)), x[0])):
        added = int(st.get("rss_added", 0)) + int(st.get("fallback_added", 0))
        rss_entries = int(st.get("rss_entries", 0))
        rss_err = int(st.get("rss_errors", 0))
        logger.info(
            f"  - {st.get('market')}:{st.get('source')}: added={added} rss_entries={rss_entries} rss_in_window={int(st.get('rss_in_window', 0))} rss_errors={rss_err} fallback_added={int(st.get('fallback_added', 0))}"
        )
        if rss_entries == 0 and rss_err == 0 and int(st.get("fallback_added", 0)) == 0:
            zero_sources.append(str(k))

    if zero_sources:
        logger.warning(f"Sources with 0 entries (and no errors): {zero_sources}")

    if args.health_out:
        health_out = str(args.health_out).strip()
        if health_out.lower() == "auto":
            health_path = outdir / f"health_{out_date}.json"
        else:
            health_path = Path(health_out)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        with open(health_path, "w", encoding="utf-8") as f:
            json.dump(health, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved health to {health_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
