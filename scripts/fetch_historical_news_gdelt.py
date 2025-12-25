import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests


GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

TARGET_QUERY = (
    '(domain:reuters.com OR domain:finance.yahoo.com OR domain:marketwatch.com OR domain:cnbc.com) '
    '(inflation OR recession OR interest OR rates OR federal OR reserve)'
)


def _normalize_published_at(seen: str, fallback_date: datetime) -> str:
    s = str(seen or "").strip()
    if not s:
        return fallback_date.strftime("%Y-%m-%d 00:00:00")

    # Common GDELT formats:
    # - 20220601000000
    # - 2022-06-01T19:15:50Z
    if len(s) >= 14 and s[:14].isdigit():
        return (
            f"{s[:4]}-{s[4:6]}-{s[6:8]} "
            f"{s[8:10]}:{s[10:12]}:{s[12:14]}"
        )

    if len(s) >= 19 and s[4] == "-" and s[7] == "-" and ("T" in s[:11]):
        return s[:19].replace("T", " ")

    # Best-effort fallback: keep date prefix if present
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10] + " 00:00:00"

    return fallback_date.strftime("%Y-%m-%d 00:00:00")


def fetch_gdelt_news(start_date: datetime, end_date: datetime, output_file: str) -> None:
    current_date = start_date
    delta = timedelta(days=1)

    records = []
    print(f"Starting GDELT fetch from {start_date.date()} to {end_date.date()}...")

    while current_date <= end_date:
        date_str_start = current_date.strftime("%Y%m%d000000")
        date_str_end = current_date.strftime("%Y%m%d235959")

        params = {
            "query": TARGET_QUERY,
            "mode": "artlist",
            "maxrecords": "250",
            "format": "json",
            "startdatetime": date_str_start,
            "enddatetime": date_str_end,
        }

        try:
            response = requests.get(
                GDELT_API_URL,
                params=params,
                timeout=20,
                headers={"User-Agent": "stock-share-Helper/1.0"},
            )
            if response.status_code == 200:
                ct = str(response.headers.get("content-type") or "")
                if "json" not in ct.lower():
                    snippet = (response.text or "")[:300].replace("\n", "\\n")
                    print(f"  {current_date.date()}: Non-JSON response ct={ct} text={snippet}")
                    current_date += delta
                    time.sleep(1)
                    continue

                data = response.json()
                articles = data.get("articles", [])
                print(f"  {current_date.date()}: Found {len(articles)} articles")

                for art in articles:
                    pub_fmt = _normalize_published_at(str(art.get("seendate") or ""), current_date)

                    title = str(art.get("title") or "").strip()
                    url = str(art.get("url") or "").strip()
                    domain = str(art.get("domain") or "gdelt").strip()

                    content = f"{title} Source: {domain}. URL: {url}".strip()

                    record = {
                        "published_at": pub_fmt,
                        "title": title,
                        "content": content,
                        "url": url,
                        "source": domain,
                        "market": "US",
                    }
                    records.append(record)
            else:
                snippet = (response.text or "")[:200].replace("\n", "\\n")
                print(f"  Error {response.status_code} for {current_date.date()} text={snippet}")

        except Exception as e:
            print(f"  Exception for {current_date.date()}: {e}")

        time.sleep(1)
        current_date += delta

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} historical news items to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="data/raw/news_us_raw_2022_06.jsonl")
    args = parser.parse_args()

    s = datetime.strptime(args.start, "%Y-%m-%d")
    e = datetime.strptime(args.end, "%Y-%m-%d")
    fetch_gdelt_news(s, e, args.output)


if __name__ == "__main__":
    main()
