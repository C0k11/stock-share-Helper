#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path
from typing import Any, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix train_news_v1.json with synthetic noise samples")
    parser.add_argument("--train", default="data/finetune/news_final_3b/train_news_v1.json")
    parser.add_argument("--noise", default="data/finetune/news_final_3b/noise_data.json")
    parser.add_argument("--out", default="data/finetune/news_final_3b/train_news_v1_noise_mix.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-max",
        type=int,
        default=0,
        help="Optional cap on number of noise samples used (0 = use all).",
    )
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train)
    noise_path = Path(args.noise)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "r", encoding="utf-8") as f:
        train: List[Any] = json.load(f)

    with open(noise_path, "r", encoding="utf-8") as f:
        noise: List[Any] = json.load(f)

    if int(args.noise_max) > 0:
        noise = noise[: int(args.noise_max)]

    mixed: List[Any] = list(train) + list(noise)

    if bool(args.shuffle):
        rng = random.Random(int(args.seed))
        rng.shuffle(mixed)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "train": str(train_path),
                "train_items": len(train),
                "noise": str(noise_path),
                "noise_items": len(noise),
                "out": str(out_path),
                "out_items": len(mixed),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
