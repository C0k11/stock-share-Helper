import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _run_paper_trading(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode == 0:
        if p.stdout:
            print(p.stdout.rstrip())
        return
    if "State already updated for date=" in out:
        print("[warn] Paper trading state already updated for this date; continuing without rerun.")
        return
    raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _append_csv_row(path: Path, header: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})


def _csv_any_row_with_date(path: Path, date_str: str) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("date") or "").strip() == str(date_str):
                    return True
    except Exception:
        return False
    return False


def _csv_tickers_for_date(path: Path, date_str: str) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("date") or "").strip() != str(date_str):
                    continue
                t = str(row.get("ticker") or "").strip().upper()
                if t:
                    out.add(t)
    except Exception:
        return set()
    return out


def _summarize_orders_csv(orders_csv: Path) -> Dict[str, Any]:
    if not orders_csv.exists():
        return {"trade_count": 0, "turnover": 0.0, "fee": 0.0}

    trade_count = 0
    turnover = 0.0
    fee_sum = 0.0
    with open(orders_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tv = _safe_float(row.get("trade_value"), 0.0)
            fee = _safe_float(row.get("fee"), 0.0)
            turnover += abs(tv)
            fee_sum += fee
            if abs(tv) > 1e-12:
                trade_count += 1

    return {"trade_count": int(trade_count), "turnover": float(turnover), "fee": float(fee_sum)}


def _append_nav(*, date_str: str, system_tag: str, portfolio_path: Path, orders_csv: Path, out_path: Path) -> bool:
    if _csv_any_row_with_date(out_path, date_str):
        return False
    if not portfolio_path.exists():
        return False
    try:
        port = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(port, dict):
        return False

    cash = _safe_float(port.get("cash"), 0.0)
    equity = _safe_float(port.get("equity"), 0.0)
    positions = port.get("positions") if isinstance(port.get("positions"), dict) else {}
    pos_count = len([k for k, v in positions.items() if abs(_safe_float(v, 0.0)) > 1e-12])

    order_stats = _summarize_orders_csv(orders_csv)

    header = ["date", "equity", "cash", "positions_count", "turnover", "fee", "trade_count", "system_tag"]
    _append_csv_row(
        out_path,
        header,
        {
            "date": str(date_str),
            "equity": float(equity),
            "cash": float(cash),
            "positions_count": int(pos_count),
            "turnover": float(order_stats.get("turnover", 0.0)),
            "fee": float(order_stats.get("fee", 0.0)),
            "trade_count": int(order_stats.get("trade_count", 0)),
            "system_tag": str(system_tag),
        },
    )
    return True


def _append_daily_signals(*, date_str: str, decision_path: Path, out_path: Path) -> int:
    if _csv_any_row_with_date(out_path, date_str):
        return 0
    if not decision_path.exists():
        return 0
    try:
        obj = json.loads(decision_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    if not isinstance(obj, dict):
        return 0

    items = obj.get("items") if isinstance(obj.get("items"), dict) else {}
    if not items:
        return 0

    header = [
        "date",
        "ticker",
        "action",
        "target_position",
        "expert",
        "planner_strategy",
        "risk_flags",
    ]

    n = 0
    for ticker, it in items.items():
        if not isinstance(it, dict):
            continue
        final = it.get("final") if isinstance(it.get("final"), dict) else {}
        router = it.get("router") if isinstance(it.get("router"), dict) else {}
        trace = final.get("trace") if isinstance(final.get("trace"), list) else []
        risk_flags = ";".join([str(x).strip() for x in trace if str(x).strip()])
        _append_csv_row(
            out_path,
            header,
            {
                "date": str(date_str),
                "ticker": str(ticker).strip().upper(),
                "action": str(final.get("action") or "").strip().upper(),
                "target_position": float(_safe_float(final.get("target_position"), 0.0)),
                "expert": str(it.get("expert") or ""),
                "planner_strategy": str(router.get("planner_strategy") or ""),
                "risk_flags": risk_flags,
            },
        )
        n += 1
    return int(n)


def _export_compat_log(*, live_ledger_path: Path, nav_path: Path, out_path: Path) -> bool:
    if not live_ledger_path.exists():
        return False

    agg: Dict[str, Dict[str, Any]] = {}
    with open(live_ledger_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = str(row.get("date") or "").strip()
            if not d:
                continue
            if d not in agg:
                agg[d] = {"date": d, "trade_count": 0, "turnover_sum": 0.0, "fee_sum": 0.0}
            px = _safe_float(row.get("filled_price"), 0.0)
            sh = _safe_float(row.get("shares"), 0.0)
            fee = _safe_float(row.get("commission"), 0.0)
            tv = abs(px * sh)
            if tv > 1e-12:
                agg[d]["trade_count"] = int(agg[d]["trade_count"]) + 1
            agg[d]["turnover_sum"] = float(agg[d]["turnover_sum"]) + float(tv)
            agg[d]["fee_sum"] = float(agg[d]["fee_sum"]) + float(fee)

    nav_by_date: Dict[str, Dict[str, Any]] = {}
    if nav_path.exists():
        try:
            with open(nav_path, "r", encoding="utf-8", newline="") as f:
                r2 = csv.DictReader(f)
                for row in r2:
                    d = str(row.get("date") or "").strip()
                    if not d:
                        continue
                    nav_by_date[d] = dict(row)
        except Exception:
            nav_by_date = {}

    all_dates = set(agg.keys()) | set(nav_by_date.keys())
    if not all_dates:
        return False

    header = [
        "date",
        "equity",
        "cash",
        "positions_count",
        "trade_count",
        "turnover_sum",
        "fee_sum",
        "system_tag",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for d in sorted(all_dates):
            nav = nav_by_date.get(d, {})
            day = agg.get(d, {"trade_count": 0, "turnover_sum": 0.0, "fee_sum": 0.0})
            w.writerow(
                {
                    "date": d,
                    "equity": _safe_float(nav.get("equity"), 0.0),
                    "cash": _safe_float(nav.get("cash"), 0.0),
                    "positions_count": int(_safe_float(nav.get("positions_count"), 0.0)),
                    "trade_count": int(_safe_float(day.get("trade_count"), 0.0)),
                    "turnover_sum": float(_safe_float(day.get("turnover_sum"), 0.0)),
                    "fee_sum": float(_safe_float(day.get("fee_sum"), 0.0)),
                    "system_tag": str(nav.get("system_tag") or ""),
                }
            )

    return True


def _generate_charts(*, date_str: str, nav_path: Path, assets_dir: Path) -> List[Path]:
    if not nav_path.exists():
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return []

    try:
        df = pd.read_csv(nav_path)
    except Exception:
        return []

    if "date" not in df.columns:
        return []

    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    except Exception:
        return []

    if df.empty:
        return []

    assets_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []

    if "equity" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df["date"], df["equity"], marker="o", linewidth=1.5)
        plt.title("Portfolio NAV")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        p1 = assets_dir / f"{date_str}_nav.png"
        plt.savefig(p1)
        plt.close()
        out.append(p1)

    if ("equity" in df.columns) and ("cash" in df.columns):
        try:
            mv = (df["equity"] - df["cash"]).clip(lower=0)
            cash = df["cash"].clip(lower=0)
        except Exception:
            mv = None
            cash = None
        if mv is not None and cash is not None:
            plt.figure(figsize=(10, 4))
            plt.stackplot(df["date"], mv, cash, labels=["Market Value", "Cash"], alpha=0.8)
            plt.title("Asset Allocation")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            p2 = assets_dir / f"{date_str}_allocation.png"
            plt.savefig(p2)
            plt.close()
            out.append(p2)

    return out


def _append_chart_links_to_report(*, report_path: Path, chart_paths: List[Path]) -> bool:
    if (not report_path.exists()) or (not chart_paths):
        return False
    try:
        txt = report_path.read_text(encoding="utf-8")
    except Exception:
        return False

    if "## Charts" in txt:
        return False

    lines: List[str] = []
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for p in chart_paths:
        rel = f"assets/{p.name}"
        lines.append(f"![{p.stem}]({rel})")
        lines.append("")

    try:
        report_path.write_text(txt.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")
    except Exception:
        return False
    return True


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
    existing = _csv_tickers_for_date(ledger_path, str(orders_csv.stem).replace("orders_", ""))
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
                if date_str and ticker and ticker in existing:
                    continue
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

    p.add_argument("--news-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--news-lora", default="models/llm_qwen14b_lora_c_hybrid/lora_weights")
    p.add_argument("--news-use-lora", action="store_true", default=False)
    p.add_argument("--news-no-lora", dest="news_use_lora", action="store_false")
    p.add_argument("--news-limit", type=int, default=0)
    p.add_argument("--news-offset", type=int, default=0)
    p.add_argument("--news-sample-us", type=int, default=0)
    p.add_argument("--news-sample-cn", type=int, default=0)

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
    p.add_argument("--paper-allow-same-day", action="store_true", default=False)

    p.add_argument("--live-ledger", default="paper_trading/live_ledger.csv")
    p.add_argument("--system-tag", default="daily_job_v1")

    p.add_argument("--force-append-nav", action="store_true", default=False)
    p.add_argument("--force-append-signals", action="store_true", default=False)

    p.add_argument("--nav-csv", default="paper_trading/nav.csv")
    p.add_argument("--daily-signals-csv", default="paper_trading/daily_signals.csv")
    p.add_argument("--compat-log-csv", default="paper_trading/log.csv")
    p.add_argument("--charts", action="store_true", default=True)
    p.add_argument("--no-charts", dest="charts", action="store_false")

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
        cmd = [
            py,
            "scripts/run_daily_inference.py",
            "--date",
            date_str,
            "--model",
            str(args.news_model),
            "--lora",
            str(args.news_lora),
            "--load-in-4bit",
            "--batch-size",
            "4",
            "--max-input-chars",
            "6000",
            "--save-every",
            "20",
        ]

        if int(args.news_offset) > 0:
            cmd.extend(["--offset", str(int(args.news_offset))])
        if int(args.news_limit) > 0:
            cmd.extend(["--limit", str(int(args.news_limit))])
        if int(args.news_sample_us) > 0:
            cmd.extend(["--sample-us", str(int(args.news_sample_us))])
        if int(args.news_sample_cn) > 0:
            cmd.extend(["--sample-cn", str(int(args.news_sample_cn))])

        lora_exists = Path(str(args.news_lora)).exists()
        if bool(args.news_use_lora) and lora_exists:
            cmd.append("--use-lora")
        elif bool(args.news_use_lora) and (not lora_exists):
            print(f"[warn] News LoRA not found, running base model only: {args.news_lora}")

        _run(cmd)

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
        cmd = [
            py,
            "scripts/run_paper_trading.py",
            "--date",
            date_str,
            "--decision",
            str(args.out_decision),
            "--paper-dir",
            str(args.paper_dir),
        ]
        if bool(args.paper_allow_same_day):
            cmd.append("--allow-same-day")
        _run_paper_trading(cmd)

    if not bool(args.skip_report):
        report_out = reports_dir / f"{date_str}.md"
        if Path(args.out_decision).exists():
            _write_report_from_decision(decision_path=Path(args.out_decision), out_path=report_out)
            print(f"Saved report: {report_out}")
        else:
            signals_path = daily_dir / f"signals_{date_str}.json"
            if (not bool(args.skip_news_parse)) and signals_path.exists():
                _run([py, "scripts/generate_daily_report.py", "--date", date_str, "--out", str(report_out)])
            else:
                raise SystemExit("No decision file and no signals file; cannot generate report")

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

    decision_path = Path(args.out_decision)
    if bool(args.force_append_signals) and Path(args.daily_signals_csv).exists():
        signals_n = _append_daily_signals(date_str=date_str, decision_path=decision_path, out_path=Path(args.daily_signals_csv))
    else:
        signals_n = 0 if _csv_any_row_with_date(Path(args.daily_signals_csv), date_str) else _append_daily_signals(
            date_str=date_str,
            decision_path=decision_path,
            out_path=Path(args.daily_signals_csv),
        )
    if signals_n > 0:
        print(f"Appended daily signals: {args.daily_signals_csv} (+{signals_n} rows)")

    portfolio_path = Path(args.paper_dir) / "portfolio.json"
    if bool(args.force_append_nav) and Path(args.nav_csv).exists():
        nav_ok = _append_nav(
            date_str=date_str,
            system_tag=str(args.system_tag),
            portfolio_path=portfolio_path,
            orders_csv=orders_csv,
            out_path=Path(args.nav_csv),
        )
    else:
        nav_ok = False if _csv_any_row_with_date(Path(args.nav_csv), date_str) else _append_nav(
            date_str=date_str,
            system_tag=str(args.system_tag),
            portfolio_path=portfolio_path,
            orders_csv=orders_csv,
            out_path=Path(args.nav_csv),
        )
    if nav_ok:
        print(f"Appended NAV: {args.nav_csv}")

    compat_ok = _export_compat_log(
        live_ledger_path=Path(args.live_ledger),
        nav_path=Path(args.nav_csv),
        out_path=Path(args.compat_log_csv),
    )
    if compat_ok:
        print(f"Exported compat log: {args.compat_log_csv}")

    if bool(args.charts) and (not bool(args.skip_report)):
        assets_dir = reports_dir / "assets"
        charts = _generate_charts(date_str=date_str, nav_path=Path(args.nav_csv), assets_dir=assets_dir)
        if charts:
            _append_chart_links_to_report(report_path=report_out, chart_paths=charts)


if __name__ == "__main__":
    main()
