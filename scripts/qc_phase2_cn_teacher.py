import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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


def normalize_sentiment(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
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


def extract_assistant_json(item: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    conv = item.get("conversations", [])
    assistant = None
    for m in conv:
        if m.get("role") == "assistant":
            assistant = m
    if not assistant:
        return False, None

    txt = assistant.get("content", "")
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return True, obj
        return False, None
    except Exception:
        return False, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/finetune/phase2_cn_teacher.json")
    args = ap.parse_args()

    path = Path(args.path)
    data = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Dataset must be a JSON list")

    parse_ok = 0
    parse_fail = 0

    missing = Counter()
    extra = Counter()

    event_type = Counter()
    sentiment_raw = Counter()
    sentiment_norm = Counter()

    impact_equity = Counter()
    impact_bond = Counter()
    impact_gold = Counter()

    rule_viol = Counter()

    for item in data:
        ok, obj = extract_assistant_json(item)
        if not ok or obj is None:
            parse_fail += 1
            continue

        parse_ok += 1

        ks = set(obj.keys())
        for k in (REQUIRED_KEYS - ks):
            missing[k] += 1
        for k in (ks - REQUIRED_KEYS):
            extra[k] += 1

        et = obj.get("event_type")
        event_type[et] += 1

        s = obj.get("sentiment")
        sentiment_raw[s] += 1
        ns = normalize_sentiment(s)
        if ns is None:
            rule_viol["sentiment_unrecognized"] += 1
        else:
            sentiment_norm[ns] += 1

        ie = obj.get("impact_equity")
        ib = obj.get("impact_bond")
        ig = obj.get("impact_gold")

        impact_equity[ie] += 1
        impact_bond[ib] += 1
        impact_gold[ig] += 1

        if et not in ALLOWED_EVENT_TYPES:
            rule_viol["bad_event_type"] += 1

        if et == "regulation_crackdown" and ie != -1:
            rule_viol["crackdown_equity_not_-1"] += 1
        if et in {"policy_stimulus", "market_intervention", "corporate_restructuring", "concept_hype"} and ie != 1:
            rule_viol["non_crackdown_equity_not_+1"] += 1

        if et == "policy_stimulus" and ib != 1:
            rule_viol["stimulus_bond_not_+1"] += 1

        # Note: gold/bond are not hard-bound for other event types in current pipeline.

    print("N", len(data))
    print("parse_ok", parse_ok, "parse_fail", parse_fail)
    print("missing", dict(missing))
    print("extra", dict(extra))
    print("event_type", dict(event_type.most_common()))
    print("impact_equity", dict(impact_equity.most_common()))
    print("impact_bond", dict(impact_bond.most_common()))
    print("impact_gold", dict(impact_gold.most_common()))
    print("sentiment_raw_top10", dict(sentiment_raw.most_common(10)))
    print("sentiment_norm", dict(sentiment_norm.most_common()))
    print("sentiment_unrecognized_count", int(rule_viol.get("sentiment_unrecognized", 0)))
    print("rule_violations", dict(rule_viol))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
