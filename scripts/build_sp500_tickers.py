#!/usr/bin/env python

import argparse
import re
from pathlib import Path
from typing import List

import requests
from loguru import logger


def parse_sp500_from_wikipedia(html: str) -> List[str]:
    # Wikipedia table contains symbols like BRK.B, BF.B
    symbols = re.findall(r"<td>([A-Z\.]{1,6})</td>", html)
    out: List[str] = []
    for s in symbols:
        s = s.strip().upper()
        if not s:
            continue
        out.append(s.replace(".", "-"))
    # Wikipedia has other tables; this regex can over-match. Filter by plausible tickers.
    out2: List[str] = []
    for s in out:
        if 1 <= len(s) <= 7 and re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", s):
            out2.append(s)
    # de-dupe while preserving order
    seen = set()
    final: List[str] = []
    for s in out2:
        if s not in seen:
            seen.add(s)
            final.append(s)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S&P 500 ticker list into a text file")
    parser.add_argument("--out", default="data/tickers/sp500.txt")
    parser.add_argument(
        "--url",
        default="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        help="Source URL for S&P 500 constituents",
    )
    args = parser.parse_args()

    url = str(args.url)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching: {url}")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    tickers = parse_sp500_from_wikipedia(r.text)
    if len(tickers) < 400:
        raise SystemExit(f"Parsed too few tickers ({len(tickers)}). Source format may have changed.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# S&P 500 tickers (generated)\n")
        for t in tickers:
            f.write(t + "\n")

    logger.info(f"Saved tickers={len(tickers)} to {out_path}")


if __name__ == "__main__":
    main()
