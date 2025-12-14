import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_KEYS = {
    "event_type",
    "sentiment",
    "impact_equity",
    "impact_bond",
    "impact_gold",
    "summary",
}

ALLOWED_EVENT_TYPES = {
    "policy_stimulus",
    "regulation_crackdown",
    "market_intervention",
    "corporate_restructuring",
    "concept_hype",
}


def clamp_impact(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        v = 0
    return max(-1, min(1, v))


def normalize_sentiment(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return max(-1, min(1, x))
    if isinstance(x, float):
        return max(-1, min(1, int(x)))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"-1", "0", "1"}:
            return int(s)
        if s in {"neg", "negative", "bearish"}:
            return -1
        if s in {"neu", "neutral", "mixed"}:
            return 0
        if s in {"pos", "positive", "bullish"}:
            return 1
    return None


def extract_assistant_obj(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conv = item.get("conversations", [])
    assistant = None
    for m in conv:
        if m.get("role") == "assistant":
            assistant = m
    if not assistant:
        return None
    try:
        obj = json.loads(assistant.get("content", ""))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def apply_cn_hard_rules(obj: Dict[str, Any]) -> Dict[str, Any]:
    et = str(obj.get("event_type") or "").strip()
    if et not in ALLOWED_EVENT_TYPES:
        # keep original, but still sanitize numeric fields
        pass

    if et == "regulation_crackdown":
        obj["impact_equity"] = -1
    elif et in {"policy_stimulus", "market_intervention", "corporate_restructuring", "concept_hype"}:
        obj["impact_equity"] = 1

    if et == "policy_stimulus":
        obj["impact_bond"] = 1
    else:
        obj["impact_bond"] = 0

    # Note: impact_gold not hard-bound by current design; keep teacher output.
    obj["impact_equity"] = clamp_impact(obj.get("impact_equity"))
    obj["impact_bond"] = clamp_impact(obj.get("impact_bond"))
    obj["impact_gold"] = clamp_impact(obj.get("impact_gold"))

    ns = normalize_sentiment(obj.get("sentiment"))
    if ns is None:
        ns = 0
    obj["sentiment"] = ns

    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/finetune/phase2_cn_teacher.json")
    ap.add_argument("--out", dest="out", default="data/finetune/phase2_cn_teacher_clean.json")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    data = json.load(open(in_path, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Dataset must be a JSON list")

    stats = Counter()

    for item in data:
        obj = extract_assistant_obj(item)
        if obj is None:
            stats["assistant_json_parse_fail"] += 1
            continue

        before_sent = obj.get("sentiment")
        before_bond = obj.get("impact_bond")

        obj = apply_cn_hard_rules(obj)

        # validate required keys exist (should already)
        missing = REQUIRED_KEYS - set(obj.keys())
        if missing:
            stats["missing_required_keys"] += 1

        if before_sent != obj.get("sentiment"):
            stats["sentiment_changed"] += 1
        if before_bond != obj.get("impact_bond"):
            stats["impact_bond_changed"] += 1

        # write back
        for m in item.get("conversations", []):
            if m.get("role") == "assistant":
                m["content"] = json.dumps(obj, ensure_ascii=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("saved", str(out_path))
    print("stats", dict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
