#!/usr/bin/env python

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


def _today_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    return data


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get_signal(it: Dict[str, Any]) -> Dict[str, Any]:
    sig = it.get("signal")
    return sig if isinstance(sig, dict) else {}


def _impact_line(sig: Dict[str, Any]) -> str:
    ie = _safe_int(sig.get("impact_equity"), 0)
    ib = _safe_int(sig.get("impact_bond"), 0)
    ig = _safe_int(sig.get("impact_gold"), 0)
    return f"EQ:{ie} BOND:{ib} GOLD:{ig}"


def _md_escape(s: str) -> str:
    return (s or "").replace("\n", " ").strip()


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
    base = _mojibake_penalty(s)
    best = s
    best_pen = base
    for enc in ("gb18030", "gbk"):
        try:
            repaired = s.encode(enc).decode("utf-8", errors="replace")
        except Exception:
            continue
        pen = _mojibake_penalty(repaired)
        if pen < best_pen:
            best = repaired
            best_pen = pen
    return best


def _rank_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(it: Dict[str, Any]) -> Tuple[int, int, int]:
        sig = _get_signal(it)
        # prioritize absolute equity impact, then bond/gold
        ie = abs(_safe_int(sig.get("impact_equity"), 0))
        ib = abs(_safe_int(sig.get("impact_bond"), 0))
        ig = abs(_safe_int(sig.get("impact_gold"), 0))
        # prefer parsed items
        ok = 1 if it.get("parse_ok") else 0
        return (ok, ie, ib + ig)

    return sorted(items, key=score, reverse=True)


def _split_markets(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    us: List[Dict[str, Any]] = []
    cn: List[Dict[str, Any]] = []
    for it in items:
        m = str(it.get("market") or "US").upper()
        if m == "CN":
            cn.append(it)
        else:
            us.append(it)
    return us, cn


def _event_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        if not it.get("parse_ok"):
            continue
        et = str(_get_signal(it).get("event_type") or "").strip() or "unknown"
        out[et] = out.get(et, 0) + 1
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def _section_risk_watch_cn(items: List[Dict[str, Any]], *, k: int = 10) -> str:
    ok_items = [it for it in items if it.get("parse_ok")]
    flagged: List[Dict[str, Any]] = []
    for it in ok_items:
        et = str(_get_signal(it).get("event_type") or "").strip()
        if et == "regulation_crackdown":
            flagged.append(it)

    total_ok = len(ok_items)
    n = len(flagged)
    ratio = (n / total_ok * 100.0) if total_ok > 0 else 0.0

    lines: List[str] = ["## Risk Watch (CN: regulation_crackdown)"]
    lines.append(f"- **count**: {n}")
    lines.append(f"- **share_of_cn_parse_ok**: {ratio:.2f}%")
    lines.append("")
    if not flagged:
        lines.append("(none)")
        return "\n".join(lines)

    # prioritize strongest absolute equity impacts
    def score(it: Dict[str, Any]) -> int:
        sig = _get_signal(it)
        return abs(_safe_int(sig.get("impact_equity"), 0))

    ranked = sorted(flagged, key=score, reverse=True)[:k]
    for it in ranked:
        sig = _get_signal(it)
        impact = _impact_line(sig)
        src = _md_escape(str(it.get("source") or ""))
        url = _md_escape(str(it.get("url") or ""))
        title_line = _md_escape(_repair_mojibake(str(it.get("title") or "")))
        summary = _md_escape(_repair_mojibake(str(sig.get("summary") or "")))
        lines.append(f"- **regulation_crackdown** ({impact}) | {src} | [{title_line}]({url})")
        if summary:
            lines.append(f"  - **summary**: {summary}")
    return "\n".join(lines)


def _section_top(items: List[Dict[str, Any]], *, title: str, k: int = 10) -> str:
    ranked = _rank_items([it for it in items if it.get("parse_ok")])[:k]
    lines: List[str] = [f"## {title}"]
    if not ranked:
        lines.append("(none)")
        return "\n".join(lines)

    for it in ranked:
        sig = _get_signal(it)
        event_type = _md_escape(_repair_mojibake(str(sig.get("event_type") or "")))
        summary = _md_escape(_repair_mojibake(str(sig.get("summary") or "")))
        impact = _impact_line(sig)
        src = _md_escape(str(it.get("source") or ""))
        url = _md_escape(str(it.get("url") or ""))
        title_line = _md_escape(_repair_mojibake(str(it.get("title") or "")))
        lines.append(f"- **{event_type}** ({impact}) | {src} | [{title_line}]({url})")
        if summary:
            lines.append(f"  - **summary**: {summary}")
    return "\n".join(lines)


def _section_failures(items: List[Dict[str, Any]], k: int = 10) -> str:
    failed = [it for it in items if not it.get("parse_ok")]
    lines: List[str] = ["## Parse failures (sample)"]
    if not failed:
        lines.append("(none)")
        return "\n".join(lines)

    for it in failed[:k]:
        src = _md_escape(str(it.get("source") or ""))
        url = _md_escape(str(it.get("url") or ""))
        title_line = _md_escape(str(it.get("title") or ""))
        lines.append(f"- **FAIL** | {src} | [{title_line}]({url})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate daily markdown report from signals JSON")
    parser.add_argument("--in", dest="in_path", default=None, help="Input signals JSON (default: data/daily/signals_YYYY-MM-DD.json)")
    parser.add_argument("--out", dest="out_path", default=None, help="Output markdown path (default: data/daily/report_YYYY-MM-DD.md)")
    parser.add_argument("--date", default=None, help="Override date YYYY-MM-DD used for default in/out")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    date_str = args.date or _today_str()
    in_path = Path(args.in_path) if args.in_path else Path("data/daily") / f"signals_{date_str}.json"
    out_path = Path(args.out_path) if args.out_path else Path("data/daily") / f"report_{date_str}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = _load_json_list(in_path)

    total = len(items)
    ok = sum(1 for it in items if it.get("parse_ok"))
    us, cn = _split_markets(items)

    header = [
        f"# Daily Market Signals ({date_str})",
        "",
        f"- **items_total**: {total}",
        f"- **parse_ok**: {ok}/{total}",
        f"- **market_us**: {len(us)}",
        f"- **market_cn**: {len(cn)}",
        "",
    ]

    us_counts = _event_counts(us)
    cn_counts = _event_counts(cn)

    def fmt_counts(title: str, counts: Dict[str, int]) -> str:
        lines = [f"## {title}"]
        if not counts:
            lines.append("(none)")
            return "\n".join(lines)
        for k, v in list(counts.items())[:15]:
            lines.append(f"- **{_md_escape(k)}**: {v}")
        return "\n".join(lines)

    parts: List[str] = []
    parts.append("\n".join(header))

    parts.append(fmt_counts("US event_type distribution (top)", us_counts))
    parts.append(_section_top(us, title="US top signals", k=args.top))

    parts.append(fmt_counts("CN event_type distribution (top)", cn_counts))
    parts.append(_section_risk_watch_cn(cn, k=min(args.top, 12)))
    parts.append(_section_top(cn, title="CN top signals", k=args.top))

    parts.append(_section_failures(items, k=10))

    content = "\n\n".join(parts).strip() + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
