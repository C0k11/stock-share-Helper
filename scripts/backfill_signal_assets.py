#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
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


def _iter_days_in_range(start: str, end: str) -> List[str]:
    s = str(start or "").strip()
    e = str(end or "").strip()
    if not s or not e:
        return []
    try:
        ds = datetime.strptime(s, "%Y-%m-%d").date()
        de = datetime.strptime(e, "%Y-%m-%d").date()
    except Exception:
        return []
    if de < ds:
        ds, de = de, ds
    out: List[str] = []
    cur = ds
    while cur <= de:
        out.append(cur.strftime("%Y-%m-%d"))
        cur = cur + timedelta(days=1)
    return out


def _load_universe_from_stock_features(daily_dir: Path, day: str) -> List[str]:
    fp = daily_dir / f"stock_features_{day}.json"
    if not fp.exists():
        return []
    data = _read_json(fp)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").strip().upper()
        if sym:
            out.append(sym)
    return sorted(list(dict.fromkeys(out)))


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


def _infer_subject_assets_local(
    *,
    model: Any,
    tokenizer: Any,
    ticker_universe: List[str],
    title: str,
    summary: str,
    max_subject_assets: int,
    max_new_tokens: int,
) -> List[str]:
    system = (
        "You are a US equities news analyst. Your task is to map a news event to a list of impacted tickers from the given universe. "
        "Output must be STRICT JSON only."
    )
    user = (
        f"TICKER_UNIVERSE={json.dumps([str(x).strip().upper() for x in ticker_universe if str(x).strip()])}\n"
        f"TITLE={str(title or '').strip()}\n"
        f"SUMMARY={str(summary or '').strip()}\n\n"
        f"Output exactly ONE JSON object: {{\"subject_assets\": [\"AAPL\", ...]}}. "
        f"subject_assets MUST be an array (possibly empty) of uppercase tickers from TICKER_UNIVERSE only. "
        f"Return at most {int(max_subject_assets)} tickers. If uncertain, return an empty array. "
        "Do not include any other keys."
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")
        else:
            # Fallback: concatenate; still enforce STRICT JSON.
            prompt = system + "\n\n" + user
            inputs = tokenizer(prompt, return_tensors="pt")

        inputs = inputs.to(model.device) if hasattr(inputs, "to") else inputs
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    except Exception:
        return []

    obj = _parse_json_object(str(text))
    if not obj:
        return []
    sa = obj.get("subject_assets")
    if not isinstance(sa, list):
        return []
    universe = {str(x).strip().upper() for x in ticker_universe if str(x).strip()}
    out_assets: List[str] = []
    for x in sa:
        t = str(x or "").strip().upper()
        if t and (t in universe):
            out_assets.append(t)
    out_assets = sorted(set(out_assets))
    if int(max_subject_assets) > 0:
        out_assets = out_assets[: int(max_subject_assets)]
    return out_assets


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


def _force_alias_assets(title: str, summary: str, raw_json: Any, tickers: List[str]) -> List[str]:
    tset = {str(t).strip().upper() for t in (tickers or []) if str(t).strip()}

    text_title = str(title or "")
    text_summary = str(summary or "")
    text_raw = str(raw_json or "")

    alias_map: Dict[str, List[str]] = {
        "LMT": [
            "Lockheed",
            "Lockheed Martin",
            "ロッキード",
            "ロッキード・マーティン",
            "F-35",
            "F35",
            "F 35",
            "F３５",
            "HIMARS",
            "HiMARS",
            "ハイマース",
            "Pentagon",
            "ペンタゴン",
            "Defense contract",
            "defense contract",
            "defence contract",
            "missile defense",
            "防衛",
            "国防",
            "軍事",
            "ミサイル",
            "長距離ミサイル",
        ],
        "RTX": [
            "Raytheon",
            "レイセオン",
            "Patriot",
            "パトリオット",
            "Javelin",
            "ジャベリン",
            "missile",
            "ミサイル",
            "air defense",
            "防空",
            "防衛",
        ],
        "NOC": ["Northrop", "Grumman"],
        "GD": ["General Dynamics"],
    }

    out: List[str] = []
    for tk, aliases in alias_map.items():
        if tk not in tset:
            continue
        for a in aliases:
            if _match_alias_word(text_title, a) or _match_alias_word(text_summary, a) or _match_alias_word(text_raw, a):
                out.append(tk)
                break

    return sorted(set(out))


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


def _ensure_item_id(it: Dict[str, Any]) -> str:
    it_id = str(it.get("id") or "").strip()
    if it_id:
        return it_id

    sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
    title = str(it.get("title") or "").strip()
    source = str(it.get("source") or "").strip()
    url = str(sig.get("url") or it.get("url") or "").strip()
    summary = str(sig.get("summary") or "").strip()

    key = "|".join([title, source, url, summary]).encode("utf-8", errors="ignore")
    hid = hashlib.sha1(key).hexdigest()[:16]
    it_id = f"auto_{hid}"
    it["id"] = it_id
    return it_id


def _save_day_signals_assets(daily_dir: Path, day: str, items: List[Dict[str, Any]]) -> Path:
    outp = daily_dir / f"signals_assets_{day}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return outp


def _bump_impact_equity_for_forced_item(it: Dict[str, Any], *, min_abs_impact: float) -> None:
    target = float(min_abs_impact) if float(min_abs_impact) > 0 else 1.0

    sig = it.get("signal") if isinstance(it.get("signal"), dict) else None
    if isinstance(sig, dict):
        try:
            cur = float(sig.get("impact_equity"))
        except Exception:
            cur = 0.0
        if abs(float(cur)) < float(target):
            sig["impact_equity"] = float(target)
        it["signal"] = sig

    raw = it.get("raw_json")
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            try:
                cur2 = float(obj.get("impact_equity"))
            except Exception:
                cur2 = 0.0
            if abs(float(cur2)) < float(target):
                obj["impact_equity"] = float(target)
            try:
                it["raw_json"] = json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill subject_assets into daily signals_YYYY-MM-DD.json")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", default="")
    g.add_argument("--date-range", nargs=2, metavar=("START", "END"), help="YYYY-MM-DD YYYY-MM-DD")

    p.add_argument("--strategy", default="v1_1_news", help="Only used when --report is set")
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

    p.add_argument(
        "--backend",
        default="api",
        choices=["api", "local"],
        help="subject_assets inference backend: api (default) or local (transformers)\n",
    )
    p.add_argument("--local-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    p.set_defaults(load_4bit=True)
    p.add_argument("--max-subject-assets", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=160)
    args = p.parse_args()

    report_path = Path(str(args.report or "").strip()) if str(args.report or "").strip() else None
    report: Optional[Dict[str, Any]] = None
    if report_path is not None:
        report_obj = _read_json(report_path)
        if not isinstance(report_obj, dict):
            raise SystemExit("Report must be JSON object")
        report = report_obj

    daily_dir = Path(str(args.daily_dir or (report.get("daily_dir") if isinstance(report, dict) else "") or "data/daily"))

    if args.date_range:
        days = _iter_days_in_range(str(args.date_range[0]), str(args.date_range[1]))
    else:
        days = sorted(_iter_report_days(report, str(args.strategy))) if isinstance(report, dict) else []

    if int(args.limit_days) > 0:
        days = days[: int(args.limit_days)]

    tickers: List[str]
    if str(args.tickers).strip():
        tickers = [t.strip().upper() for t in str(args.tickers).split(",") if t.strip()]
    elif isinstance(report, dict):
        tickers = [str(t).strip().upper() for t in (report.get("tickers") or []) if str(t).strip()]
    else:
        tickers = _load_universe_from_stock_features(daily_dir, days[0]) if days else []

    if not tickers:
        raise SystemExit("Empty ticker universe. Provide --tickers or pass --report with report.tickers, or ensure stock_features_<day>.json exists.")

    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    local_model = None
    local_tokenizer = None
    if str(args.backend) == "api":
        api_key, base_url, model_name = _get_teacher_config()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_id = str(args.local_model).replace("\\", "/")
        tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["low_cpu_mem_usage"] = True

        if bool(args.load_4bit):
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        local_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        local_model.eval()
        local_tokenizer = tok

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

            forced_hits = _force_alias_assets(title, summary, it.get("raw_json"), tickers)

            hits = sorted(set(_universe_hits(title, summary, tickers) + forced_hits))
            if bool(args.require_universe_hit) and not hits:
                continue

            imp = sig.get("impact_equity")
            try:
                imp_f = float(imp)
            except Exception:
                imp_f = 0.0
            # If we have forced hits (alias match), allow it even with low impact.
            if (not hits) and (abs(imp_f) < float(args.min_abs_impact)):
                continue
            it_id = _ensure_item_id(it)
            prio = 1000.0 if hits else abs(imp_f)
            candidates.append((float(prio), it_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        k = max(0, int(args.max_items_per_day))
        id_set = {it_id for _score, it_id in candidates[:k]}

        out_items: List[Dict[str, Any]] = []
        for it in signals:
            cp = dict(it)
            _ = _ensure_item_id(cp)
            cp.pop("subject_assets", None)
            cp["subject_assets"] = []
            out_items.append(cp)

        if not id_set:
            _save_day_signals_assets(daily_dir, day, out_items)
            processed_days += 1
            days_no_candidates += 1
            continue

        for idx, it in enumerate(out_items):
            it_id = _ensure_item_id(it)
            if it_id not in id_set:
                continue

            sig = it.get("signal") if isinstance(it.get("signal"), dict) else {}
            title = str(it.get("title") or "")
            summary = str(sig.get("summary") or "")

            forced = _force_alias_assets(title, summary, it.get("raw_json"), tickers)
            if forced:
                it["subject_assets"] = forced
                if isinstance(it.get("signal"), dict):
                    it["signal"]["subject_assets"] = forced
                _bump_impact_equity_for_forced_item(it, min_abs_impact=float(args.min_abs_impact))
                processed_items += 1
                continue

            try:
                if str(args.backend) == "api":
                    sa = _infer_subject_assets(
                        api_key=str(api_key),
                        base_url=str(base_url),
                        model=str(model_name),
                        ticker_universe=tickers,
                        title=title,
                        summary=summary,
                        timeout=float(args.timeout),
                        max_retries=int(args.max_retries),
                    )
                else:
                    sa = _infer_subject_assets_local(
                        model=local_model,
                        tokenizer=local_tokenizer,
                        ticker_universe=tickers,
                        title=title,
                        summary=summary,
                        max_subject_assets=int(args.max_subject_assets),
                        max_new_tokens=int(args.max_new_tokens),
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
                "model": (str(model_name) if str(args.backend) == "api" else str(args.local_model)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
