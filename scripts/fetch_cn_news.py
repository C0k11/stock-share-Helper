#!/usr/bin/env python
"""抓取中文财经新闻，输出为 teacher 生成脚本需要的格式。

用法：
  .\venv311\Scripts\python.exe scripts\fetch_cn_news.py --out data\cn_news_400.json --limit 400

输出格式：
  [{"title": "...", "content": "...", "published_at": "...", "source": "..."}, ...]
"""

import sys
import argparse
import json
import hashlib
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import feedparser
import requests
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


CN_RSS_SOURCES = [
    {
        "id": "sina_finance_focus",
        "name": "新浪财经焦点",
        "url": "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2516&k=&num=50&page=1&r=0.1&callback=",
        "type": "sina_api",
        "category": "market_news",
    },
    {
        "id": "eastmoney_news",
        "name": "东方财富要闻",
        "url": "https://newsapi.eastmoney.com/kuaixun/v1/getlist_101_ajaxResult_50_1_.html",
        "type": "eastmoney_api",
        "category": "market_news",
    },
    {
        "id": "cls_telegraph",
        "name": "财联社电报",
        "url": "https://www.cls.cn/nodeapi/updateTelegraphList",
        "type": "cls_api",
        "category": "flash_news",
    },
]


def _set_query_param(url: str, key: str, value: str) -> str:
    if f"{key}=" not in url:
        sep = "&" if "?" in url else "?"
        return url + sep + f"{key}={value}"
    return re.sub(rf"({re.escape(key)}=)([^&]*)", rf"\g<1>{value}", url)


def _eastmoney_set_page(url: str, page: int, per_page: int) -> str:
    # Example: ...getlist_101_ajaxResult_50_1_.html
    if re.search(r"_\d+_\d+_\.html$", url):
        return re.sub(r"_(\d+)_(\d+)_\.html$", f"_{per_page}_{page}_.html", url)
    return url


def stable_id(title: str, content: str, published_at: str) -> str:
    basis = "|".join([published_at or "", title or "", content or ""]).strip()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def fetch_sina_api(url: str, limit: int) -> List[Dict[str, Any]]:
    """从新浪财经 API 抓取新闻"""
    results = []
    json_match = None
    try:
        resp = requests.get(url, timeout=15)
        text = resp.text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
    except Exception as e:
        logger.warning(f"Sina API fetch failed: {e}")
        return results

    if not json_match:
        return results

    try:
        data = json.loads(json_match.group())
        items = data.get("result", {}).get("data", [])
        for item in items[:limit]:
            title = (item.get("title") or "").strip()
            content = (item.get("intro") or item.get("summary") or "").strip()
            pub_time = item.get("ctime") or item.get("create_time") or ""
            if pub_time and str(pub_time).isdigit():
                pub_time = datetime.fromtimestamp(int(pub_time)).isoformat()
            results.append(
                {
                    "title": title,
                    "content": content,
                    "published_at": pub_time,
                    "source": "sina_finance",
                    "url": item.get("url") or "",
                }
            )
    except Exception as e:
        logger.warning(f"Sina API parse failed: {e}")

    return results


def fetch_sina_api_paged(base_url: str, max_pages: int, per_page: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        url = _set_query_param(base_url, "page", str(page))
        url = _set_query_param(url, "num", str(per_page))
        items = fetch_sina_api(url, per_page)
        if not items:
            break
        results.extend(items)
        if len(items) < max(5, per_page // 3):
            break
    return results


def fetch_eastmoney_api(url: str, limit: int) -> List[Dict[str, Any]]:
    """从东方财富快讯 API 抓取"""
    results = []
    try:
        resp = requests.get(url, timeout=15)
        text = resp.text
        json_match = re.search(r"var defined[^=]*=\s*(\[.*?\]);", text, re.DOTALL)
        if not json_match:
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return results
        items = json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
        for item in items[:limit]:
            title = (item.get("title") or "").strip()
            content = (item.get("digest") or item.get("content") or "").strip()
            pub_time = item.get("showtime") or item.get("time") or ""
            results.append({
                "title": title,
                "content": content,
                "published_at": pub_time,
                "source": "eastmoney",
                "url": item.get("url") or "",
            })
    except Exception as e:
        logger.warning(f"Eastmoney API fetch failed: {e}")
    return results


def fetch_eastmoney_api_paged(base_url: str, max_pages: int, per_page: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        url = _eastmoney_set_page(base_url, page=page, per_page=per_page)
        items = fetch_eastmoney_api(url, per_page)
        if not items:
            break
        results.extend(items)
        if len(items) < max(5, per_page // 3):
            break
    return results


def fetch_cls_api(url: str, limit: int) -> List[Dict[str, Any]]:
    """从财联社电报 API 抓取"""
    results = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.cls.cn/telegraph",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        items = data.get("data", {}).get("roll_data", [])
        for item in items[:limit]:
            title = (item.get("title") or item.get("brief") or "").strip()
            content = (item.get("content") or item.get("brief") or "").strip()
            pub_time = item.get("ctime") or ""
            if pub_time and str(pub_time).isdigit():
                pub_time = datetime.fromtimestamp(int(pub_time)).isoformat()
            results.append({
                "title": title,
                "content": content,
                "published_at": pub_time,
                "source": "cls_telegraph",
                "url": f"https://www.cls.cn/detail/{item.get('id', '')}" if item.get("id") else "",
            })
    except Exception as e:
        logger.warning(f"CLS API fetch failed: {e}")
    return results


def fetch_rss_generic(url: str, limit: int, source_name: str) -> List[Dict[str, Any]]:
    """通用 RSS 抓取"""
    results = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            title = (entry.get("title") or "").strip()
            content = ""
            if hasattr(entry, "summary"):
                content = (entry.summary or "").strip()
            elif hasattr(entry, "description"):
                content = (entry.description or "").strip()
            content = re.sub(r"<[^>]+>", "", content)
            pub_time = ""
            if hasattr(entry, "published"):
                pub_time = entry.published
            elif hasattr(entry, "updated"):
                pub_time = entry.updated
            results.append({
                "title": title,
                "content": content,
                "published_at": pub_time,
                "source": source_name,
                "url": entry.get("link") or "",
            })
    except Exception as e:
        logger.warning(f"RSS fetch failed for {source_name}: {e}")
    return results


def fetch_all_cn_news(limit: int = 400, per_page: int = 50, max_pages: int = 10) -> List[Dict[str, Any]]:
    """从所有 CN 源抓取新闻"""
    # Notes:
    # - 一些源按“翻页”才能拿到更多条数；这里做尽量兼容的分页抓取。
    # - 目标是尽可能接近 limit，最终仍以去重后的数量为准。
    all_news = []
    per_page = max(10, int(per_page or 50))
    max_pages = max(1, int(max_pages or 10))
    desired_per_source = max(80, int(math.ceil(limit / max(1, len(CN_RSS_SOURCES))) + 40))

    for src in CN_RSS_SOURCES:
        src_type = src.get("type", "rss")
        url = src["url"]
        name = src["name"]

        logger.info(f"Fetching from {name}...")

        if src_type == "sina_api":
            items = fetch_sina_api_paged(url, max_pages=max_pages, per_page=per_page)
        elif src_type == "eastmoney_api":
            items = fetch_eastmoney_api_paged(url, max_pages=max_pages, per_page=per_page)
        elif src_type == "cls_api":
            items = fetch_cls_api(url, desired_per_source)
        else:
            items = fetch_rss_generic(url, desired_per_source, name)

        logger.info(f"  Got {len(items)} items from {name}")
        all_news.extend(items)

    seen_ids = set()
    deduped = []
    for item in all_news:
        item_id = stable_id(item["title"], item["content"], item["published_at"])
        if item_id in seen_ids:
            continue
        if not item["title"] and not item["content"]:
            continue
        if len(item.get("title", "")) < 5 and len(item.get("content", "")) < 10:
            continue
        seen_ids.add(item_id)
        item["id"] = item_id
        deduped.append(item)

    deduped = deduped[:limit]
    logger.info(f"Total unique CN news: {len(deduped)}")
    return deduped


def main():
    parser = argparse.ArgumentParser(description="抓取中文财经新闻")
    parser.add_argument("--out", default="data/cn_news_400.json", help="输出文件路径")
    parser.add_argument("--limit", type=int, default=400, help="最大新闻条数")
    parser.add_argument("--per-page", type=int, default=50, help="每页条数（部分源支持）")
    parser.add_argument("--max-pages", type=int, default=10, help="最大翻页数（部分源支持）")

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    news = fetch_all_cn_news(limit=args.limit, per_page=args.per_page, max_pages=args.max_pages)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(news, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(news)} CN news items to {out_path}")


if __name__ == "__main__":
    main()
