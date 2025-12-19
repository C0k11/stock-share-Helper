import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def detect_strong_news_days(signals_dir: Path, min_abs_impact: float) -> Set[str]:
    days: Set[str] = set()
    for fp in sorted(signals_dir.glob("signals_????-??-??.json")):
        day = fp.name[len("signals_") : len("signals_") + 10]
        try:
            items = _read_json(fp)
        except Exception:
            continue
        if not isinstance(items, list):
            continue

        strong = False
        for it in items:
            if not isinstance(it, dict):
                continue
            if not it.get("parse_ok"):
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
            if sig is None:
                continue
            impact = _safe_float(sig.get("impact_equity"))
            if impact is None:
                continue
            if abs(impact) >= float(min_abs_impact):
                strong = True
                break

        if strong:
            days.add(day)

    return days


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [t for t in trades if _safe_float(t.get("realized_return")) is not None]
    if not scored:
        return {
            "trades": len(trades),
            "scored": 0,
            "win_rate": 0.0,
            "cumulative_return": 0.0,
            "buy_sell_trades": 0,
            "buy_sell_cumulative_return": 0.0,
        }

    rets = [float(t["realized_return"]) for t in scored]
    wins = sum(1 for r in rets if r > 0)

    bs = [t for t in scored if str(t.get("decision") or "").upper() in {"BUY", "SELL"}]
    bs_rets = [float(t["realized_return"]) for t in bs]

    return {
        "trades": len(trades),
        "scored": len(scored),
        "win_rate": wins / len(scored),
        "cumulative_return": sum(rets),
        "buy_sell_trades": len(bs),
        "buy_sell_cumulative_return": sum(bs_rets) if bs_rets else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="data/backtest/report_2025_12_final.json")
    parser.add_argument("--signals-dir", default="data/daily")
    parser.add_argument("--min-news-abs-impact", type=float, default=0.5)
    args = parser.parse_args()

    report_path = Path(args.report)
    signals_dir = Path(args.signals_dir)

    report = _read_json(report_path)
    strategies = report.get("strategies") if isinstance(report, dict) else None
    if not isinstance(strategies, dict):
        raise SystemExit("Invalid report schema: missing top-level 'strategies'")

    news_days = detect_strong_news_days(signals_dir, float(args.min_news_abs_impact))

    print(f"Report: {report_path}")
    print(f"SignalsDir: {signals_dir}")
    print(f"StrongNewsDays: {len(news_days)} (min_abs_impact={float(args.min_news_abs_impact)})")

    for strat_name, content in strategies.items():
        if not isinstance(content, dict):
            continue
        trades = content.get("trades")
        if not isinstance(trades, list):
            continue

        news_trades = [t for t in trades if str(t.get("date") or "") in news_days]
        quiet_trades = [t for t in trades if str(t.get("date") or "") not in news_days]

        total_s = summarize(trades)
        news_s = summarize(news_trades)
        quiet_s = summarize(quiet_trades)

        print("\nStrategy:", strat_name)
        print("  Total:")
        print(
            "    trades={trades} scored={scored} win_rate={win_rate:.4f} cum_ret={cumulative_return:.6f} buy_sell={buy_sell_trades} buy_sell_ret={buy_sell_cumulative_return:.6f}".format(
                **total_s
            )
        )
        print("  StrongNewsDays:")
        print(
            "    trades={trades} scored={scored} win_rate={win_rate:.4f} cum_ret={cumulative_return:.6f} buy_sell={buy_sell_trades} buy_sell_ret={buy_sell_cumulative_return:.6f}".format(
                **news_s
            )
        )
        print("  QuietDays:")
        print(
            "    trades={trades} scored={scored} win_rate={win_rate:.4f} cum_ret={cumulative_return:.6f} buy_sell={buy_sell_trades} buy_sell_ret={buy_sell_cumulative_return:.6f}".format(
                **quiet_s
            )
        )


if __name__ == "__main__":
    main()
