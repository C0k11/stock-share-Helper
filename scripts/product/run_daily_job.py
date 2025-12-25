import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


_LIVE_LEDGER_HEADER = [
    "date",
    "ticker",
    "action",
    "target_position",
    "filled_price",
    "shares",
    "commission",
    "system_tag",
]


def _ensure_live_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LIVE_LEDGER_HEADER)
        w.writeheader()


def _append_live_ledger_from_orders(*, ledger_path: Path, orders_csv: Path, system_tag: str) -> int:
    _ensure_live_ledger(ledger_path)
    if not orders_csv.exists():
        return 0

    rows_written = 0
    with open(orders_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        with open(ledger_path, "a", encoding="utf-8", newline="") as out_f:
            w = csv.DictWriter(out_f, fieldnames=_LIVE_LEDGER_HEADER)
            for row in r:
                trade_value = _safe_float(row.get("trade_value"), 0.0)
                if abs(trade_value) <= 1e-12:
                    continue

                ticker = str(row.get("ticker") or "").strip().upper()
                date_str = str(row.get("date") or "").strip()
                target_pos = _safe_float(row.get("target_pos"), 0.0)
                px = _safe_float(row.get("price"), 0.0)
                shares_delta = _safe_float(row.get("shares_delta"), 0.0)
                fee = _safe_float(row.get("fee"), 0.0)

                if abs(target_pos) <= 1e-12:
                    action = "CLEAR"
                else:
                    action = "BUY" if shares_delta > 0 else "SELL"

                w.writerow(
                    {
                        "date": date_str,
                        "ticker": ticker,
                        "action": action,
                        "target_position": target_pos,
                        "filled_price": px,
                        "shares": shares_delta,
                        "commission": fee,
                        "system_tag": system_tag,
                    }
                )
                rows_written += 1

    return rows_written


def _format_md_table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return ""
    out: List[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _write_report_from_decision(*, decision_path: Path, out_path: Path) -> None:
    if not decision_path.exists():
        raise SystemExit(f"Decision file not found: {decision_path}")

    import json

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise SystemExit(f"Invalid decision json: {decision_path}")

    date_str = str(decision.get("date") or "").strip() or out_path.stem
    items = decision.get("items") if isinstance(decision.get("items"), dict) else {}

    alerts: List[str] = []
    action_rows: List[List[str]] = []
    for ticker, it in items.items():
        if not isinstance(it, dict):
            continue
        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        action = str(final.get("action") or "").strip().upper()
        target_pos = _safe_float(final.get("target_position"), 0.0)
        trace = final.get("trace") if isinstance(final.get("trace"), list) else []
        for t in trace:
            s = str(t or "").strip()
            if not s:
                continue
            if "FORCE CLEAR" in s or "[RISK]" in s or "Drawdown" in s:
                alerts.append(f"{str(ticker).upper()}: {s}")
        action_rows.append([str(ticker).upper(), action, f"{target_pos:+.3f}"])

    action_rows = sorted(action_rows, key=lambda r: abs(_safe_float(r[2], 0.0)), reverse=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Daily Report ({date_str})")
    lines.append("")
    lines.append("## Risk Alert")
    if alerts:
        for a in alerts[:20]:
            lines.append(f"- {a}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Action Plan")
    if action_rows:
        lines.append(_format_md_table(action_rows[:50], ["ticker", "action", "target_position"]))
    else:
        lines.append("(no decisions)")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_single_day_decision_from_results(*, run_dir: Path, system: str, date_str: str, out_path: Path) -> Path:
    sys_dir = run_dir / str(system)
    if not sys_dir.exists():
        raise SystemExit(f"Results system dir not found: {sys_dir}")

    cands = sorted(sys_dir.glob("decisions_*.json"))
    if not cands:
        raise SystemExit(f"No decisions_*.json under: {sys_dir}")
    src = cands[-1]

    import json

    obj = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"Invalid results decisions json: {src}")

    days = obj.get("days") if isinstance(obj.get("days"), dict) else None
    if not isinstance(days, dict):
        raise SystemExit(f"Results decisions json is not multi-day format (missing days): {src}")

    day = days.get(str(date_str))
    if not isinstance(day, dict):
        available = sorted([str(k) for k in days.keys()])
        raise SystemExit(f"Date not found in results decisions: {date_str} (available sample: {available[:5]})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(day, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 16.2: one-click daily job (fetch -> signals -> features -> trading inference -> report -> paper log)")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")

    p.add_argument("--venv-python", default="", help="Optional explicit python executable (default: current interpreter)")

    p.add_argument("--daily-dir", default="data/daily")
    p.add_argument("--reports-dir", default="reports/daily")

    p.add_argument("--risk-watch-market", default="BOTH")
    p.add_argument("--planner-mode", default="off", choices=["off", "rule"])

    p.add_argument("--moe-mode", action="store_true", default=False)
    p.add_argument("--no-moe-mode", dest="moe_mode", action="store_false")

    p.add_argument("--universe", default="etf", choices=["etf", "stock", "auto"])
    p.add_argument("--tickers", default="", help="Comma-separated tickers (required when universe=stock)")

    p.add_argument("--out-decision", default="data/decisions_inference.json")

    p.add_argument(
        "--results-run-dir",
        default="",
        help="Optional replay mode: extract decisions from results/<run>/<system>/decisions_*.json (multi-day) for --date.",
    )
    p.add_argument("--results-system", default="golden_strict", choices=["golden_strict", "baseline_fast"])

    p.add_argument("--run-paper", action="store_true", default=True)
    p.add_argument("--no-run-paper", dest="run_paper", action="store_false")
    p.add_argument("--paper-dir", default="data/paper")

    p.add_argument("--live-ledger", default="paper_trading/live_ledger.csv")
    p.add_argument("--system-tag", default="daily_job_v1")

    p.add_argument("--skip-fetch", action="store_true", default=False)
    p.add_argument("--skip-news-parse", action="store_true", default=False)
    p.add_argument("--skip-features", action="store_true", default=False)
    p.add_argument("--skip-trading-inference", action="store_true", default=False)
    p.add_argument("--skip-report", action="store_true", default=False)

    args = p.parse_args()

    date_str = str(args.date or _today_str()).strip()
    daily_dir = Path(args.daily_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    py = str(args.venv_python).strip() or sys.executable

    results_run_dir = str(args.results_run_dir or "").strip()
    if results_run_dir:
        extracted = _extract_single_day_decision_from_results(
            run_dir=Path(results_run_dir),
            system=str(args.results_system),
            date_str=date_str,
            out_path=Path(args.out_decision),
        )
        print(f"Extracted decision from results: {extracted}")

    tickers = str(args.tickers or "").strip()
    universe = str(args.universe).strip().lower()
    if universe == "stock" and not tickers:
        raise SystemExit("--tickers is required when --universe stock")

    if not bool(args.skip_fetch):
        _run([py, "scripts/fetch_daily_rss.py", "--date", date_str, "--health-out", "auto"])

    if not bool(args.skip_news_parse):
        _run([py, "scripts/run_daily_inference.py", "--date", date_str, "--use-lora", "--load-in-4bit", "--batch-size", "4", "--max-input-chars", "6000", "--save-every", "20"])

    if not bool(args.skip_features):
        _run([py, "scripts/build_daily_etf_features.py", "--date", date_str])

    if (not results_run_dir) and (not bool(args.skip_trading_inference)):
        cmd = [
            py,
            "scripts/run_trading_inference.py",
            "--date",
            date_str,
            "--daily-dir",
            str(daily_dir),
            "--universe",
            str(args.universe),
            "--planner-mode",
            str(args.planner_mode),
            "--risk-watch-market",
            str(args.risk_watch_market),
            "--out",
            str(args.out_decision),
        ]
        if tickers:
            cmd.extend(["--tickers", tickers])
        if bool(args.moe_mode):
            cmd.append("--moe-mode")
        _run(cmd)

    if bool(args.run_paper):
        _run([
            py,
            "scripts/run_paper_trading.py",
            "--date",
            date_str,
            "--decision",
            str(args.out_decision),
            "--paper-dir",
            str(args.paper_dir),
        ])

    if not bool(args.skip_report):
        report_out = reports_dir / f"{date_str}.md"
        signals_path = daily_dir / f"signals_{date_str}.json"
        if signals_path.exists():
            _run([py, "scripts/generate_daily_report.py", "--date", date_str, "--out", str(report_out)])
        else:
            _write_report_from_decision(decision_path=Path(args.out_decision), out_path=report_out)
            print(f"Saved report: {report_out}")

    orders_csv = Path(args.paper_dir) / "orders" / f"orders_{date_str}.csv"
    n = _append_live_ledger_from_orders(
        ledger_path=Path(args.live_ledger),
        orders_csv=orders_csv,
        system_tag=str(args.system_tag),
    )
    if n > 0:
        print(f"Appended live ledger: {args.live_ledger} (+{n} rows)")
    else:
        print(f"Live ledger ready (no trades appended): {args.live_ledger}")


if __name__ == "__main__":
    main()
