#!/usr/bin/env python

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def _load_env_local(repo_root: Path) -> None:
    fp = repo_root / ".env.local"
    if not fp.exists():
        return

    try:
        raw = fp.read_bytes()
    except Exception:
        return

    text: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        if line.lower().startswith("export "):
            line = line[len("export ") :].strip()
        if line.lower().startswith("set "):
            line = line[len("set ") :].strip()
        if line.startswith("$env:"):
            line = line[len("$env:") :].strip()

        if "=" in line:
            k, v = line.split("=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue

        k = k.strip()
        v = v.strip().rstrip(";").strip()
        if not k:
            continue

        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]

        if k not in os.environ:
            os.environ[k] = v


def _get_teacher_config() -> Tuple[str, str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    _load_env_local(repo_root)

    api_key = (
        os.environ.get("TEACHER_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Missing teacher API key. Set TEACHER_API_KEY (preferred) or DEEPSEEK_API_KEY/OPENAI_API_KEY in .env.local or environment."
        )

    base_url = (
        os.environ.get("TEACHER_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    )

    model = os.environ.get("TEACHER_MODEL") or "deepseek-reasoner"
    return str(api_key), str(base_url), str(model)


def _post_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    base = str(base_url).rstrip("/")
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
    except Exception as e:
        raise RuntimeError(str(e))

    data = json.loads(raw)
    return str(data["choices"][0]["message"]["content"])


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
        s = s.strip()

    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

    return obj if isinstance(obj, dict) else None


def _infer_subject_assets(
    *,
    api_key: str,
    base_url: str,
    model: str,
    ticker_universe: List[str],
    title: str,
    summary: str,
    timeout: float,
    max_retries: int,
) -> List[str]:
    universe = [str(x).strip().upper() for x in ticker_universe if str(x).strip()]

    system = (
        "You are a US equities news analyst. Your task is to map a news event to a list of impacted tickers from the given universe. "
        "Output must be STRICT JSON only."
    )
    user = (
        f"TICKER_UNIVERSE={json.dumps(universe)}\n"
        f"TITLE={title}\n"
        f"SUMMARY={summary}\n\n"
        "Output exactly ONE JSON object: {\"subject_assets\": [\"AAPL\", ...]}. "
        "subject_assets MUST be an array (possibly empty) of uppercase tickers from TICKER_UNIVERSE only. "
        "Do not include any other keys."
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(int(max_retries)):
        try:
            out = _post_chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=120,
                timeout=timeout,
            )
            obj = _parse_json_object(out)
            if not obj:
                raise RuntimeError("parse_failed")
            sa = obj.get("subject_assets")
            if not isinstance(sa, list):
                raise RuntimeError("subject_assets_not_list")
            norm: List[str] = []
            for x in sa:
                t = str(x or "").strip().upper()
                if t and t in universe:
                    norm.append(t)
            norm = sorted(set(norm))
            return norm
        except Exception as e:
            if attempt + 1 >= int(max_retries):
                raise
            time.sleep(2 ** attempt)
            _ = e

    return []


def _match_ticker_symbol_in_text(text: str, ticker: str) -> bool:
    if not text:
        return False
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    pat = re.compile(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])")
    return pat.search(str(text).upper()) is not None


def _match_alias_word(text: str, alias: str) -> bool:
    if not text:
        return False
    a = str(alias or "").strip()
    if not a:
        return False
    pat = re.compile(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", flags=re.IGNORECASE)
    return pat.search(str(text)) is not None


def _universe_hits(title: str, summary: str, tickers: List[str]) -> List[str]:
    hits: List[str] = []

    tset = [str(t).strip().upper() for t in tickers if str(t).strip()]

    alias_map: Dict[str, List[str]] = {
        "AAPL": ["Apple", "iPhone", "iOS", "MacBook"],
        "NVDA": ["Nvidia", "NVIDIA", "GeForce", "CUDA"],
        "TSLA": ["Tesla", "TSLA"],
        "MSFT": ["Microsoft", "Windows", "Azure"],
        "GOOGL": ["Google", "Alphabet"],
    }

    text_title = str(title or "")
    text_summary = str(summary or "")

    for t in tset:
        if _match_ticker_symbol_in_text(text_title, t) or _match_ticker_symbol_in_text(text_summary, t):
            hits.append(t)
            continue

        for alias in alias_map.get(t, []):
            if _match_alias_word(text_title, alias) or _match_alias_word(text_summary, alias):
                hits.append(t)
                break

    return sorted(set(hits))


def _iter_report_days(report: Dict[str, Any], strategy: str) -> Set[str]:
    strategies = report.get("strategies") if isinstance(report.get("strategies"), dict) else {}
    strat = strategies.get(str(strategy)) if isinstance(strategies, dict) else None
    trades = strat.get("trades") if isinstance(strat, dict) and isinstance(strat.get("trades"), list) else []

    out: Set[str] = set()
    for tr in trades:
        if not isinstance(tr, dict):
            continue
        d = str(tr.get("date") or "").strip()
        if len(d) == 10 and d[4] == "-" and d[7] == "-":
            out.add(d)
    return out


def _load_day_signals(daily_dir: Path, day: str) -> List[Dict[str, Any]]:
    fp = daily_dir / f"signals_{day}.json"
    if not fp.exists():
        return []
    data = _read_json(fp)
    if not isinstance(data, list):
        return []
    return [it for it in data if isinstance(it, dict)]


def _save_day_signals_assets(daily_dir: Path, day: str, items: List[Dict[str, Any]]) -> Path:
    outp = daily_dir / f"signals_assets_{day}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return outp


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill subject_assets into daily signals_YYYY-MM-DD.json using teacher API")
    p.add_argument("--report", required=True)
    p.add_argument("--strategy", default="v1_1_news")
    p.add_argument("--daily-dir", default="", help="Override daily dir (default: use report.daily_dir)")
    p.add_argument("--tickers", default="", help="Comma-separated ticker universe (default: use report.tickers)")
    p.add_argument("--limit-days", type=int, default=0)
    p.add_argument("--max-items-per-day", type=int, default=30)
    p.add_argument("--min-abs-impact", type=float, default=1.0)
    p.add_argument("--require-universe-hit", action="store_true")
    p.add_argument("--no-require-universe-hit", dest="require_universe_hit", action="store_false")
    p.set_defaults(require_universe_hit=True)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    report_path = Path(args.report)
    report = _read_json(report_path)
    if not isinstance(report, dict):
        raise SystemExit("Report must be JSON object")

    daily_dir = Path(str(args.daily_dir or report.get("daily_dir") or "data/daily"))
    days = sorted(_iter_report_days(report, str(args.strategy)))
    if int(args.limit_days) > 0:
        days = days[: int(args.limit_days)]

    tickers: List[str]
    if str(args.tickers).strip():
        tickers = [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in (report.get("tickers") or []) if str(t).strip()]

    api_key, base_url, model = _get_teacher_config()

    processed_days = 0
    processed_items = 0
    days_no_candidates = 0

    for day in days:
        outp = daily_dir / f"signals_assets_{day}.json"
        if outp.exists() and (not bool(args.overwrite)):
            continue

        signals = _load_day_signals(daily_dir, day)
        if not signals:
            continue

        candidates: List[Tuple[float, str]] = []
        for it in signals:
            if not isinstance(it, dict):
                continue
            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            title = str(it.get("title") or "")
            summary = str(sig.get("summary") or "")

            hits = _universe_hits(title, summary, tickers)
            if bool(args.require_universe_hit) and not hits:
                continue

            imp = sig.get("impact_equity")
            try:
                imp_f = float(imp)
            except Exception:
                imp_f = 0.0
            if (not hits) and (abs(imp_f) < float(args.min_abs_impact)):
                continue
            it_id = str(it.get("id") or "")
            if not it_id:
                continue
            prio = 1000.0 if hits else abs(imp_f)
            candidates.append((float(prio), it_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        k = max(0, int(args.max_items_per_day))
        id_set = {it_id for _score, it_id in candidates[:k]}

        out_items: List[Dict[str, Any]] = []
        for it in signals:
            cp = dict(it)
            cp.pop("subject_assets", None)
            cp["subject_assets"] = []
            out_items.append(cp)

        if not id_set:
            _save_day_signals_assets(daily_dir, day, out_items)
            processed_days += 1
            days_no_candidates += 1
            continue

        for idx, it in enumerate(out_items):
            it_id = str(it.get("id") or "")
            if it_id not in id_set:
                continue

            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            title = str(it.get("title") or "")
            summary = str(sig.get("summary") or "")

            try:
                sa = _infer_subject_assets(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    ticker_universe=tickers,
                    title=title,
                    summary=summary,
                    timeout=float(args.timeout),
                    max_retries=int(args.max_retries),
                )
                it["subject_assets"] = sa
                if isinstance(it.get("signal"), dict):
                    it["signal"]["subject_assets"] = sa
                processed_items += 1
            except Exception:
                it["subject_assets"] = []
                if isinstance(it.get("signal"), dict):
                    it["signal"]["subject_assets"] = []

            if float(args.delay) > 0:
                time.sleep(float(args.delay))

            _ = idx

        _save_day_signals_assets(daily_dir, day, out_items)
        processed_days += 1

    print(
        json.dumps(
            {
                "report": str(report_path),
                "daily_dir": str(daily_dir),
                "strategy": str(args.strategy),
                "days": len(days),
                "processed_days": processed_days,
                "processed_items": processed_items,
                "model": model,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
