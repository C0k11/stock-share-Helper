import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_list(path: Path) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")
    for i, x in enumerate(data[:3]):
        if not isinstance(x, dict):
            raise ValueError(f"{path} item[{i}] must be an object")
    return data


def sample_to_size(items: List[Dict[str, Any]], size: int, rng: random.Random) -> List[Dict[str, Any]]:
    if size <= 0:
        return []
    if not items:
        return []
    if len(items) >= size:
        return rng.sample(items, k=size)
    # sample with replacement (deterministic via seed)
    return [rng.choice(items) for _ in range(size)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", default="data/finetune/train.json", help="US dataset JSON list")
    ap.add_argument("--cn", default="data/finetune/phase2_cn_teacher_clean.json", help="CN clean dataset JSON list")
    ap.add_argument("--out", default="data/finetune/phase2_hybrid_1000.json", help="Output JSON list")
    ap.add_argument("--target", type=int, default=1000, help="Target total items (0=just concat)")
    ap.add_argument("--us-ratio", type=float, default=0.6, help="US ratio when sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    us_path = Path(args.us)
    cn_path = Path(args.cn)
    out_path = Path(args.out)

    us = load_list(us_path)
    cn = load_list(cn_path)

    if args.target and args.target > 0:
        target = int(args.target)
        us_n = int(round(target * float(args.us_ratio)))
        cn_n = target - us_n
        rng = random.Random(int(args.seed))
        us_s = sample_to_size(us, us_n, rng)
        cn_s = sample_to_size(cn, cn_n, rng)
        out = us_s + cn_s
        if args.shuffle:
            rng.shuffle(out)
    else:
        out = us + cn

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("us_items", len(us))
    print("cn_items", len(cn))
    print("out_items", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
