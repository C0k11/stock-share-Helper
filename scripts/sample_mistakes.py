#!/usr/bin/env python

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _load_signals(daily_dir: Path, date_str: str) -> List[Dict[str, Any]]:
    fp_assets = daily_dir / f"signals_assets_{date_str}.json"
    fp = fp_assets if fp_assets.exists() else (daily_dir / f"signals_{date_str}.json")
    if not fp.exists():
        return []
    try:
        data = _read_json(fp)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [it for it in data if isinstance(it, dict)]


def _signal_impact(it: Dict[str, Any]) -> float:
    if not isinstance(it, dict):
        return 0.0
    if not it.get("parse_ok"):
        return 0.0
    sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
    v = _to_float(sig.get("impact_equity"))
    return float(v) if v is not None else 0.0


def _day_news_score(signals: List[Dict[str, Any]]) -> float:
    return float(sum(abs(_signal_impact(it)) for it in signals))


def _top_news(signals: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in signals:
        imp = abs(_signal_impact(it))
        if imp <= 0:
            continue
        scored.append((imp, it))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    for _imp, it in scored[: max(0, int(k))]:
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        out.append(
            {
                "market": it.get("market"),
                "source": it.get("source"),
                "title": it.get("title"),
                "published_at": it.get("published_at"),
                "event_type": sig.get("event_type"),
                "impact_equity": sig.get("impact_equity"),
                "summary": sig.get("summary"),
                "url": it.get("url"),
            }
        )
    return out


def _format_news_items(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    kk = max(0, int(k))
    for it in items[:kk]:
        if not isinstance(it, dict):
            continue
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        out.append(
            {
                "market": it.get("market"),
                "source": it.get("source"),
                "title": it.get("title"),
                "published_at": it.get("published_at"),
                "event_type": sig.get("event_type"),
                "impact_equity": sig.get("impact_equity"),
                "summary": sig.get("summary"),
                "url": it.get("url"),
            }
        )
    return out


def _news_text(it: Dict[str, Any]) -> str:
    if not isinstance(it, dict):
        return ""
    title = str(it.get("title") or "").strip()
    sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
    summary = str(sig.get("summary") or "").strip()
    text = (title + "\n" + summary).strip()
    return text


def _top_ticker_news(signals: List[Dict[str, Any]], k: int, *, min_news_chars: int) -> List[Dict[str, Any]]:
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for it in signals:
        if not isinstance(it, dict):
            continue
        txt = _news_text(it)
        if len(txt) < int(min_news_chars):
            continue
        candidates.append((len(txt), it))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    kk = max(0, int(k))
    return [it for _len, it in candidates[:kk]]


def _ticker_news_candidates(
    signals: List[Dict[str, Any]],
    ticker: str,
    *,
    min_news_chars: int,
) -> Tuple[List[Dict[str, Any]], str]:
    matched, join_mode = _select_ticker_signals(signals, ticker)
    if not matched:
        return [], join_mode

    out: List[Dict[str, Any]] = []
    for it in matched:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        if not title:
            continue
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        _ = str(sig.get("summary") or "").strip()
        if len(_news_text(it)) < int(min_news_chars):
            continue
        out.append(it)

    return out, join_mode


def _normalize_assets(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip().upper()
        return [s] if s else []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            s = str(x or "").strip().upper()
            if s:
                out.append(s)
        return out
    return []


def _item_assets(it: Dict[str, Any]) -> List[str]:
    sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
    for key in ("subject_assets", "tickers", "symbols", "assets"):
        if key in sig:
            assets = _normalize_assets(sig.get(key))
            if assets:
                return assets
        if key in it:
            assets = _normalize_assets(it.get(key))
            if assets:
                return assets
    return []


def _match_ticker_symbol_in_text(text: str, ticker: str) -> bool:
    if not text:
        return False
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    pat = re.compile(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])")
    return pat.search(str(text)) is not None


def _select_ticker_signals(signals: List[Dict[str, Any]], ticker: str) -> Tuple[List[Dict[str, Any]], str]:
    t = str(ticker or "").strip().upper()
    if not t:
        return [], "none"

    by_assets: List[Dict[str, Any]] = []
    for it in signals:
        if not isinstance(it, dict):
            continue
        assets = _item_assets(it)
        if assets and t in assets:
            by_assets.append(it)
    if by_assets:
        return by_assets, "assets"

    by_symbol: List[Dict[str, Any]] = []
    for it in signals:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "")
        sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
        summary = str(sig.get("summary") or "")
        if _match_ticker_symbol_in_text(title, t) or _match_ticker_symbol_in_text(summary, t):
            by_symbol.append(it)
    if by_symbol:
        return by_symbol, "symbol"

    return [], "none"


def _load_stock_features(daily_dir: Path, date_str: str, ticker: str) -> Dict[str, Any]:
    fp = daily_dir / f"stock_features_{date_str}.json"
    if not fp.exists():
        return {}
    try:
        data = _read_json(fp)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    items = data.get("items")
    if isinstance(items, dict):
        it = items.get(ticker)
        return it if isinstance(it, dict) else {}

    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or it.get("ticker") or "").strip().upper()
            if sym == str(ticker).strip().upper():
                return it
    return {}


def _is_wrong(decision: str, fwd: float, min_abs_move: float) -> bool:
    d = str(decision or "").strip().upper()
    if d == "BUY":
        return fwd <= -float(min_abs_move)
    if d == "SELL":
        return fwd >= float(min_abs_move)
    return False


def _loss_score(decision: str, fwd: float) -> float:
    d = str(decision or "").strip().upper()
    if d == "BUY" and fwd < 0:
        return abs(fwd)
    if d == "SELL" and fwd > 0:
        return abs(fwd)
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Sample worst mistakes under strong-news days for CoT distillation")
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="mistake_only", choices=["mistake_only", "all_news"])
    p.add_argument("--strategy", default="v1_1_news")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--min-abs-move", type=float, default=0.01)
    p.add_argument("--news-score-threshold", type=float, default=1.0)
    p.add_argument("--news-topk", type=int, default=3)
    p.add_argument("--min-news-chars", type=int, default=40)
    p.add_argument("--require-ticker-news", action="store_true")
    p.add_argument("--no-require-ticker-news", dest="require_ticker_news", action="store_false")
    p.set_defaults(require_ticker_news=True)
    args = p.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    report = _read_json(report_path)
    if not isinstance(report, dict):
        raise SystemExit("Report must be a JSON object")

    strategies = report.get("strategies") if isinstance(report.get("strategies"), dict) else {}
    strat = strategies.get(str(args.strategy)) if isinstance(strategies, dict) else None
    if not isinstance(strat, dict):
        raise SystemExit(f"Strategy not found in report: {args.strategy}")

    trades = strat.get("trades") if isinstance(strat.get("trades"), list) else []
    daily_dir = Path(str(report.get("daily_dir") or "data/daily"))

    day_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
    join_mode_counts: Dict[str, int] = {"assets": 0, "symbol": 0, "none": 0}

    picked: List[Tuple[float, Dict[str, Any]]] = []

    for tr in trades:
        if not isinstance(tr, dict):
            continue
        if not tr.get("has_news_context"):
            continue

        date_str = str(tr.get("date") or "").strip()
        ticker = str(tr.get("ticker") or "").strip().upper()
        decision = str(tr.get("decision") or "").strip().upper()

        fwd = _to_float(tr.get("forward_return"))
        if fwd is None:
            continue

        if str(args.mode) == "mistake_only":
            if not _is_wrong(decision, float(fwd), float(args.min_abs_move)):
                continue

        if date_str not in day_cache:
            sigs = _load_signals(daily_dir, date_str)
            day_cache[date_str] = (_day_news_score(sigs), sigs)

        day_score, sigs = day_cache[date_str]
        if float(day_score) < float(args.news_score_threshold):
            continue

        ticker_sigs, join_mode = _ticker_news_candidates(sigs, ticker, min_news_chars=int(args.min_news_chars))
        join_mode_counts[join_mode] = join_mode_counts.get(join_mode, 0) + 1
        if bool(args.require_ticker_news) and not ticker_sigs:
            continue

        if str(args.mode) == "mistake_only":
            score = _loss_score(decision, float(fwd))
            if score <= 0:
                continue
        else:
            score = abs(float(fwd))

        parsed = tr.get("parsed") if isinstance(tr.get("parsed"), dict) else {}
        feat = _load_stock_features(daily_dir, date_str, ticker)
        ticker_news_top = _top_ticker_news(ticker_sigs, int(args.news_topk), min_news_chars=int(args.min_news_chars))
        if bool(args.require_ticker_news) and not ticker_news_top:
            continue
        rec = {
            "date": date_str,
            "ticker": ticker,
            "mode": str(args.mode),
            "strategy": str(args.strategy),
            "decision": decision,
            "analysis": parsed.get("analysis", ""),
            "forward_return": float(fwd),
            "news_score": float(day_score),
            "news_top": _format_news_items(ticker_news_top, int(args.news_topk)) if ticker_news_top else [],
            "news_join_mode": join_mode,
            "features": feat,
        }
        picked.append((float(score), float(day_score), rec))

    picked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    topk = picked[: max(0, int(args.top_k))]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _score, _day_score, rec in topk:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "report": str(report_path),
                "strategy": str(args.strategy),
                "mode": str(args.mode),
                "trades_total": len(trades),
                "picked": len(picked),
                "written": len(topk),
                "out": str(out_path),
                "news_score_threshold": float(args.news_score_threshold),
                "min_abs_move": float(args.min_abs_move),
                "min_news_chars": int(args.min_news_chars),
                "require_ticker_news": bool(args.require_ticker_news),
                "join_mode_counts": join_mode_counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
