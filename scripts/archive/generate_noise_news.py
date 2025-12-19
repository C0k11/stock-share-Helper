#!/usr/bin/env python

import argparse
import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

US_EVENT_TYPES = [
    "rate_hike",
    "rate_cut",
    "hawkish_signal",
    "dovish_signal",
    "inflation_data",
    "employment_data",
    "gdp_growth",
    "earnings_beat",
    "earnings_miss",
    "guidance_update",
    "merger_acquisition",
    "liquidity_event",
    "short_squeeze",
    "technical_breakout",
    "geopolitical_tension",
    "trade_policy",
    "noise",
    "concept_hype",
]

TEMPLATES: List[Dict[str, Any]] = [
    {
        "type": "hard_product",
        "templates": [
            "Check out the top 10 {item} of 2024. We tested {brand} against {brand2}.",
            "{company} announces new color options for its best-selling {product}.",
            "Review: Why the {product} is the best gift for {holiday}.",
            "{company} launches a limited edition sneaker collaboration with {celebrity}.",
        ],
        "vars": {
            "item": ["vacuum cleaners", "air fryers", "gaming mice", "yoga mats"],
            "brand": ["Dyson", "Ninja", "Logitech", "Lululemon", "Sony", "Samsung"],
            "brand2": ["Shark", "Philips", "Razer", "Alo", "LG", "Bose"],
            "company": ["Apple", "Nike", "Sony", "Starbucks", "Canon"],
            "product": ["iPhone Case", "Air Jordan", "Lens", "Coffee Blend"],
            "holiday": ["Christmas", "Valentine's Day"],
            "celebrity": ["Travis Scott", "Taylor Swift"],
        },
    },
    {
        "type": "sports_ent",
        "templates": [
            "{team} defeats {team2} 3-1 in the {league} playoffs.",
            "{player} signs a 5-year contract extension with {team}.",
            "{movie} tops the box office this weekend with $50M opening.",
            "{celebrity} spotted in {city} wearing {brand}.",
        ],
        "vars": {
            "team": ["Lakers", "Warriors", "Manchester United", "Yankees"],
            "team2": ["Celtics", "Bulls", "Liverpool", "Red Sox"],
            "league": ["NBA", "Premier League", "NFL"],
            "player": ["LeBron James", "Messi", "Curry"],
            "movie": ["Dune Part 2", "Barbie", "Oppenheimer"],
            "celebrity": ["Kim Kardashian", "Tom Cruise"],
            "city": ["Paris", "Tokyo", "New York"],
            "brand": ["Gucci", "Prada"],
        },
    },
    {
        "type": "life",
        "templates": [
            "Heavy rain expected in {city} this weekend, causing potential flooding.",
            "5 tips to improve your {activity} skills immediately.",
            "The best recipe for {food} that takes only 20 minutes.",
            "Traffic alert: {road} closed due to construction.",
        ],
        "vars": {
            "city": ["London", "Florida", "California"],
            "activity": ["coding", "sleeping", "cooking"],
            "food": ["pasta", "pancakes", "salad"],
            "road": ["I-95", "Highway 1", "Main Street"],
        },
    },
]


def _gen_text(rng: random.Random) -> str:
    cat = rng.choice(TEMPLATES)
    tpl = rng.choice(cat["templates"])
    for key, vals in cat["vars"].items():
        needle = "{" + key + "}"
        if needle in tpl:
            tpl = tpl.replace(needle, rng.choice(vals))
    return tpl


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic noise samples for News Tower (v1.1 Noise Killer)")
    parser.add_argument("--out", default="data/finetune/news_final_3b/noise_data.json")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system = "You are a US financial market intelligence analyst. Output STRICT JSON only."

    rows: List[Dict[str, Any]] = []
    for i in range(int(args.n)):
        content = _gen_text(rng)
        item_id = f"noise_{args.seed}_{i:05d}"

        user = (
            f"US_EVENT_TYPE_ENUMS: {', '.join(US_EVENT_TYPES)}\n\n"
            "HISTORICAL_CONTEXT_TOP3_JSONL:\n\n"
            f"CURRENT_NEWS_JSON={{\"id\": \"{item_id}\", \"market\": \"US\", \"published_at\": \"{_now_iso()}\", "
            f"\"source\": \"synthetic_noise\", \"url\": \"\", \"title\": \"\", \"content\": {json.dumps(content)} }}\n\n"
            "Output exactly ONE JSON object with keys: market, event_type, subject_assets, sentiment_score, confidence, summary, reasoning, rag_evidence_ids. "
            "market MUST be 'US'. event_type MUST be one of the predefined enums. "
            "subject_assets MUST be an array of uppercase tickers (e.g., ['SPY','QQQ']); it MAY be an empty array if the news has no clear tradable target. "
            "sentiment_score MUST be a float between -1 and 1. confidence MUST be a float between 0 and 1. rag_evidence_ids MUST be an array of ids from the provided historical context (or empty array). "
            "Do not include any additional keys. If the news is non-financial, off-topic, or contains insufficient information, set event_type='noise' and use subject_assets=[] with low confidence."
        )

        assistant_obj = {
            "market": "US",
            "event_type": "noise",
            "subject_assets": [],
            "sentiment_score": 0.0,
            "confidence": 0.95,
            "summary": "Non-financial/off-topic content with no tradable market impact.",
            "reasoning": "The content is unrelated to financial markets and does not provide actionable information for tradable assets.",
            "rag_evidence_ids": [],
        }

        rows.append(
            {
                "conversations": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
                ],
                "meta": {
                    "market": "US",
                    "published_at": _now_iso(),
                    "source": "synthetic_noise",
                    "url": "",
                },
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"saved={len(rows)} out={out_path}")


if __name__ == "__main__":
    main()
