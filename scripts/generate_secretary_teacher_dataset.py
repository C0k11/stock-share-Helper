#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_LOG = PROJECT_ROOT / "data" / "agent_hub" / "chat.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "secretary.yaml"


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(PROJECT_ROOT / ".env.local", override=False)
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        return


def _normalize_openai_compat_base_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("Empty base_url")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    base_url = _normalize_openai_compat_base_url(base_url)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=int(timeout))
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")

    data = resp.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema: {data}") from e


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        obj = None
    return obj if isinstance(obj, dict) else {}


def _iter_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
            if limit and len(out) >= int(limit):
                break
    return out


def _stable_id(parts: List[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _as_compact_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)


def _synthetic_rejected_reply(*, user_message: str, server_ctx: Dict[str, Any]) -> str:
    msg = str(user_message or "").strip()
    lt = server_ctx.get("live_trading") if isinstance(server_ctx.get("live_trading"), dict) else {}
    positions = lt.get("positions") if isinstance(lt.get("positions"), list) else []
    cash = lt.get("cash")
    total_value = lt.get("total_value")
    total_pnl = lt.get("total_pnl")

    lines: List[str] = []
    if "谁" in msg and ("赚钱" in msg or "亏" in msg):
        lines.append("Sensei，我按实时引擎状态给您汇总一下（我先把全量信息都贴出来）。")
    else:
        lines.append("Sensei，我先给您做一个持仓与盈亏汇总。")

    if cash is not None:
        lines.append(f"现金：{cash}")
    if total_value is not None and total_pnl is not None:
        lines.append(f"总资产：{total_value}（总盈亏 {total_pnl}）")
    if positions:
        lines.append("当前持仓：")
        for p in positions[:8]:
            tk = str(p.get("ticker") or "")
            shares = p.get("shares")
            avgp = p.get("avg_price")
            curp = p.get("current_price")
            upnl = p.get("unrealized_pnl")
            lines.append(f"- {tk}: {shares} 股 @ {avgp} -> {curp}, 浮动盈亏 {upnl}")
    else:
        lines.append("当前没有持仓。")

    return "\n".join([s for s in lines if str(s).strip()]).strip()


def _synth_pool(*, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))

    base_server_live = {
        "live_trading": {
            "mode": "live_paper_trading",
            "trading_mode": "online",
            "data_source": "simulated",
            "cash": 500000.0,
            "total_value": 500000.0,
            "total_pnl": 0.0,
            "positions": [],
            "latest_bars": {"AMD": "2026-01-03T17:12:00"},
        }
    }

    pool: List[Dict[str, Any]] = []

    pool.append(
        {
            "user": "我们现在的数据源是真实还是虚拟？最后一根K线是什么时间？",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": base_server_live,
        }
    )
    pool.append(
        {
            "user": "现在开盘了吗？如果休市你要直接说休市，不要猜。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": {
                "live_trading": {
                    "mode": "live_paper_trading",
                    "trading_mode": "online",
                    "data_source": "yfinance",
                    "cash": 498200.0,
                    "total_value": 501120.0,
                    "total_pnl": 1120.0,
                    "positions": [{"ticker": "AMD", "shares": 10, "avg_price": 120.0, "current_price": 121.5, "unrealized_pnl": 15.0}],
                    "latest_bars": {"AMD": "2026-01-03T16:00:00"},
                }
            },
        }
    )

    pool.append(
        {
            "user": "现在谁赚钱最多？同时：把这个结论交给分析员和交易员，给出今天的特别措施，让亏损更少、赚钱更多。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": {
                "live_trading": {
                    "mode": "live_paper_trading",
                    "trading_mode": "online",
                    "data_source": "simulated",
                    "cash": 228448.30,
                    "total_value": 516819.40,
                    "total_pnl": 16819.40,
                    "positions": [
                        {"ticker": "AMD", "shares": 620, "avg_price": 174.90, "current_price": 175.25, "unrealized_pnl": 217.87},
                        {"ticker": "NVDA", "shares": 150, "avg_price": 177.38, "current_price": 181.55, "unrealized_pnl": 625.96},
                        {"ticker": "MSFT", "shares": 390, "avg_price": 165.25, "current_price": 169.88, "unrealized_pnl": 1804.33},
                        {"ticker": "TSLA", "shares": 160, "avg_price": 541.31, "current_price": 538.94, "unrealized_pnl": -379.20},
                    ],
                    "latest_bars": {"AMD": "2026-01-04T11:37:00", "NVDA": "2026-01-04T11:37:00", "MSFT": "2026-01-04T11:37:00", "TSLA": "2026-01-04T11:37:00"},
                }
            },
        }
    )

    pool.append(
        {
            "user": "先告诉我今天亏得最多的是谁，然后给分析师和交易员下任务：针对亏损最大的标的，列一个保守的风控清单。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": {
                "live_trading": {
                    "mode": "live_paper_trading",
                    "trading_mode": "online",
                    "data_source": "simulated",
                    "cash": 460618.50,
                    "total_value": 515439.80,
                    "total_pnl": 15439.80,
                    "positions": [
                        {"ticker": "AMD", "shares": 180, "avg_price": 174.61, "current_price": 178.34, "unrealized_pnl": 670.59},
                        {"ticker": "NVDA", "shares": 130, "avg_price": 176.29, "current_price": 174.77, "unrealized_pnl": -197.39},
                    ],
                    "latest_bars": {"AMD": "2026-01-04T11:37:00", "NVDA": "2026-01-04T11:37:00"},
                }
            },
        }
    )
    pool.append(
        {
            "user": "给我汇总一下现在的现金、持仓和总盈亏。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": {
                "live_trading": {
                    "mode": "live_paper_trading",
                    "trading_mode": "online",
                    "data_source": "yfinance",
                    "cash": 486000.0,
                    "total_value": 503400.0,
                    "total_pnl": 3400.0,
                    "positions": [
                        {"ticker": "AMD", "shares": 30, "avg_price": 118.0, "current_price": 121.0, "unrealized_pnl": 90.0},
                        {"ticker": "NVDA", "shares": 5, "avg_price": 900.0, "current_price": 910.0, "unrealized_pnl": 50.0},
                    ],
                    "latest_bars": {"AMD": "2026-01-03T17:12:00", "NVDA": "2026-01-03T17:12:00"},
                }
            },
        }
    )
    pool.append(
        {
            "user": "把看盘切到 AMD。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": base_server_live,
        }
    )
    pool.append(
        {
            "user": "刷新一下仪表盘，然后告诉我刷新成功没。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": base_server_live,
        }
    )
    pool.append(
        {
            "user": "切到 offline 模式回放一下，然后告诉我你现在是什么模式。",
            "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
            "server": {
                "live_trading": {
                    "mode": "live_paper_trading",
                    "trading_mode": "offline",
                    "data_source": "simulated",
                    "cash": 500000.0,
                    "total_value": 500000.0,
                    "total_pnl": 0.0,
                    "positions": [],
                    "latest_bars": {},
                }
            },
        }
    )

    tickers = ["AMD", "NVDA", "TSLA", "AAPL", "MSFT", "QQQ", "SPY", "META", "AMZN", "GOOGL"]
    for tk in tickers:
        pool.append(
            {
                "user": f"现在 {tk} 怎么样？用两句话说清楚，不要编价格。",
                "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
                "server": {
                    "live_trading": {
                        "mode": "live_paper_trading",
                        "trading_mode": "online",
                        "data_source": "unknown",
                        "cash": 500000.0,
                        "total_value": 500000.0,
                        "total_pnl": 0.0,
                        "positions": [],
                        "latest_bars": {tk: ""},
                    }
                },
            }
        )

    primary_templates = [
        "现在谁赚钱最多？",
        "现在谁亏得最多？",
        "告诉我当前最大仓位是谁？",
        "现在开盘了吗？如果不确定就直说，不要猜。",
        "我们数据源是真实还是模拟？最后一根bar是什么时间？",
        "简短告诉我 {tk} 的数据源和新鲜度。",
        "现在 {tk} 怎么样？用两句话说清楚，不要编价格。",
    ]
    secondary_templates = [
        "同时：把这个结论交给分析员和交易员，给出今天的特别措施，让亏损更少、赚钱更多。",
        "然后给分析师和交易员下任务：针对亏损最大的标的，列一个保守的风控清单。",
        "顺便把看盘切到 {tk}。",
        "再刷新一下仪表盘，然后告诉我刷新成功没。",
        "最后用一句话总结你接下来打算盯什么风险点。",
        "并且把这条指令记录成任务编号，方便我回查。",
        "再给分析员/交易员一个‘事件清单’：财报/宏观数据(FOMC/CPI)/政策/监管/期权波动率，说明为什么要盯这些。",
        "把今天的风险拆成三类：趋势风险、波动风险、流动性风险；每类给一条应对措施。",
        "给一份保守的交易预案：什么条件下加仓/减仓/观望，并解释原因（不要编价格）。",
    ]

    # Create a large, diverse synthetic pool for high target counts (e.g., 3k)
    for _ in range(4200):
        tk = rng.choice(tickers)
        mode = rng.choice(["online", "offline"])
        src = rng.choice(["yfinance", "simulated", "unknown"])

        # Randomly decide whether it's single intent or multi-intent.
        p0 = rng.choice(primary_templates).format(tk=tk)
        msg = p0
        if rng.random() < 0.75:
            s0 = rng.choice(secondary_templates).format(tk=tk)
            # Blend with natural connectors to mimic real user phrasing.
            glue = rng.choice(["同时：", "另外：", "顺便：", "然后：", "并且：", ""])
            msg = (p0 + " " + (glue + s0 if glue else s0)).strip()

        # Sometimes include a direct address and a slightly longer sentence.
        if rng.random() < 0.35:
            msg = rng.choice(["Sensei，", "玛丽，", ""])
            msg = (str(msg) + str(p0)).strip()
            if rng.random() < 0.75:
                msg = (msg + " " + rng.choice(secondary_templates).format(tk=tk)).strip()

        # Build a lightweight live context.
        positions: List[Dict[str, Any]] = []
        if rng.random() < 0.6:
            npos = rng.randrange(0, 5)
            for _k in rng.sample(tickers, k=min(npos, len(tickers))):
                shares = int(rng.randrange(5, 800))
                avgp = float(rng.randrange(50, 600))
                curp = avgp + float(rng.randrange(-40, 40)) / 10.0
                upnl = (curp - avgp) * float(shares)
                positions.append({"ticker": _k, "shares": shares, "avg_price": avgp, "current_price": curp, "unrealized_pnl": upnl})

        latest_bars = {tk: "2026-01-03T17:12:00"}
        if rng.random() < 0.55:
            for _k in rng.sample(tickers, k=min(4, len(tickers))):
                latest_bars[_k] = "2026-01-04T11:37:00"

        pool.append(
            {
                "user": msg,
                "client": {"client": "desktop", "theme": "dark", "user_role": "Sensei"},
                "server": {
                    "live_trading": {
                        "mode": "live_paper_trading",
                        "trading_mode": mode,
                        "data_source": src,
                        "cash": float(rng.randrange(120000, 520000)),
                        "total_value": float(rng.randrange(160000, 680000)),
                        "total_pnl": float(rng.randrange(-25000, 25000)),
                        "positions": positions,
                        "latest_bars": latest_bars,
                    }
                },
            }
        )

    return pool


def _build_teacher_messages(
    *,
    secretary_system_prompt: str,
    user_message: str,
    client_context: Dict[str, Any],
    server_context: Dict[str, Any],
    original_reply: str,
) -> List[Dict[str, str]]:
    system = (
        "You are an expert dialogue editor and a trading assistant coach. "
        "You will be given a character system prompt, a user message, and partial trading context. "
        "Rewrite the assistant reply to be more accurate, more helpful, and strictly grounded in the provided context. "
        "Keep the original persona and constraints. "
        "Return ONLY the final rewritten assistant reply text (no markdown, no explanations)."
    )

    user_payload = {
        "system_prompt": secretary_system_prompt,
        "user_message": user_message,
        "client_context": client_context,
        "server_context": server_context,
        "original_reply": original_reply,
        "hard_rules": [
            "Write as Mari speaking to Sensei: first-person perspective, natural and human tone (no robotic labels).",
            "Keep it short but explanatory: <= 6 sentences, include brief justification when helpful.",
            "Acknowledge the full user intent in one short sentence before answering.",
            "If the user message contains multiple requests, answer the primary question first, then handle the secondary request.",
            "Address the user request directly.",
            "Do not invent market status/data source/portfolio numbers; use only provided server_context.",
            "Never output tool logs, JSON, or ACTION blocks unless the user explicitly asks for an action.",
            "Use the user's required address (e.g., Sensei) if present in system_prompt.",
            "When asked to dispatch tasks to analysts/traders, confirm you will dispatch and provide a task_id placeholder if available; do not claim execution without evidence.",
        ],
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _maybe_load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("meta") and isinstance(obj.get("meta"), dict):
                sid = str(obj["meta"].get("id") or "").strip()
                if sid:
                    done.add(sid)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Secretary teacher distillation dataset from data/agent_hub/chat.jsonl")
    p.add_argument("--in", dest="inp", default=str(DEFAULT_CHAT_LOG), help="Input chat.jsonl path")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to configs/secretary.yaml")

    p.add_argument("--mode", default="chat", choices=["chat", "synth"], help="chat: read chat.jsonl; synth: generate synthetic user prompts")
    p.add_argument("--target", type=int, default=0, help="Target number of NEW samples to generate (0=all available)")

    p.add_argument("--out", default="data/finetune/teacher_secretary/teacher_secretary.jsonl")
    p.add_argument("--out-train", default="data/finetune/teacher_secretary/train_secretary_v1.json")
    p.add_argument("--out-val", default="data/finetune/teacher_secretary/val_secretary_v1.json")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--out-dpo", default="", help="Optional: output DPO preference pairs JSONL")

    p.add_argument("--resume", action="store_true")
    p.add_argument("--max", type=int, default=0, help="Max number of new samples to generate (0=all)")
    p.add_argument("--max-read", type=int, default=0, help="Only read first N chat turns (debug); 0=all")

    p.add_argument("--teacher-base-url", default=os.getenv("TEACHER_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL", "deepseek-chat"))
    p.add_argument("--teacher-api-key-env", default="TEACHER_API_KEY")

    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-output-tokens", type=int, default=350)
    p.add_argument("--sleep", type=float, default=0.2)

    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _try_load_dotenv()

    api_key = os.getenv(str(args.teacher_api_key_env), "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.teacher_api_key_env} (set it in .env.local)")

    in_path = Path(str(args.inp))
    cfg = _read_yaml(Path(str(args.config)))
    secretary_prompt = str(((cfg.get("secretary") or {}) if isinstance(cfg.get("secretary"), dict) else {}).get("system_prompt") or "").strip()

    if not secretary_prompt:
        raise SystemExit(f"secretary.system_prompt missing in config: {args.config}")

    mode = str(args.mode or "chat").strip().lower()
    raw: List[Dict[str, Any]] = []
    if mode == "chat":
        raw = _iter_jsonl(in_path, limit=int(args.max_read) if int(args.max_read) > 0 else 0)
    else:
        raw = []

    out_path = PROJECT_ROOT / str(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_train_path = PROJECT_ROOT / str(args.out_train)
    out_val_path = PROJECT_ROOT / str(args.out_val)
    out_train_path.parent.mkdir(parents=True, exist_ok=True)
    out_val_path.parent.mkdir(parents=True, exist_ok=True)

    out_dpo_path: Optional[Path] = None
    if str(args.out_dpo).strip():
        out_dpo_path = PROJECT_ROOT / str(args.out_dpo)
        out_dpo_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _maybe_load_done_ids(out_path) if bool(args.resume) else set()

    written = 0
    skipped = 0
    progress_every = 20

    target_new = int(args.target) if int(args.target) > 0 else 0

    with open(out_path, "a", encoding="utf-8") as fout, (
        open(out_dpo_path, "a", encoding="utf-8") if out_dpo_path is not None else open(os.devnull, "w")
    ) as fdpo:
        rows_iter: List[Dict[str, Any]]
        if mode == "chat":
            rows_iter = raw
        else:
            rows_iter = _synth_pool(seed=int(args.seed))

        idx = 0
        while True:
            if mode == "chat":
                if idx >= len(rows_iter):
                    break
                row = rows_iter[idx]
                idx += 1
                if str(row.get("type") or "") != "chat.turn":
                    continue
                msg = str(row.get("message") or "").strip()
                rep = str(row.get("reply") or "").strip()
                if not msg:
                    continue
                if not rep:
                    rep = ""
                ctx = row.get("context") if isinstance(row.get("context"), dict) else {}
                server_ctx = row.get("server_context") if isinstance(row.get("server_context"), dict) else {}
                sid = _stable_id([
                    str(row.get("time") or ""),
                    str(ctx.get("client") or ""),
                    msg,
                    rep,
                    _as_compact_json(server_ctx),
                ])
                src_time = row.get("time")
            else:
                it = rows_iter[idx % len(rows_iter)]
                idx += 1
                msg = str(it.get("user") or "").strip()
                ctx = it.get("client") if isinstance(it.get("client"), dict) else {"client": "synth"}
                server_ctx = it.get("server") if isinstance(it.get("server"), dict) else {}
                rep = ""
                sid = _stable_id([
                    "synth",
                    str(idx),
                    msg,
                    _as_compact_json(ctx),
                    _as_compact_json(server_ctx),
                ])
                src_time = ""

            if sid in done_ids:
                skipped += 1
                if mode == "chat":
                    continue
                if target_new and written >= target_new:
                    break
                continue

            teacher_messages = _build_teacher_messages(
                secretary_system_prompt=secretary_prompt,
                user_message=msg,
                client_context=ctx,
                server_context=server_ctx,
                original_reply=rep,
            )

            try:
                better = _call_openai_compatible_chat(
                    base_url=str(args.teacher_base_url),
                    api_key=api_key,
                    model=str(args.teacher_model),
                    messages=teacher_messages,
                    timeout=int(args.timeout),
                    max_tokens=int(args.max_output_tokens),
                    temperature=float(args.temperature),
                )
            except Exception:
                continue

            better = str(better or "").strip()
            if not better:
                continue

            out_obj = {
                "conversations": [
                    {"role": "system", "content": secretary_prompt},
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": better},
                ],
                "meta": {
                    "id": sid,
                    "time": src_time,
                    "client": ctx.get("client"),
                    "teacher_base_url": str(args.teacher_base_url),
                    "teacher_model": str(args.teacher_model),
                    "original_reply": rep,
                },
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()
            done_ids.add(sid)
            written += 1

            if progress_every > 0 and (written % int(progress_every) == 0):
                print(
                    json.dumps(
                        {
                            "written": written,
                            "skipped": skipped,
                            "out_jsonl": str(out_path),
                            "out_train": str(out_train_path),
                            "out_val": str(out_val_path),
                            "out_dpo": str(out_dpo_path) if out_dpo_path is not None else "",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if out_dpo_path is not None:
                rejected_text = rep
                if not rejected_text:
                    rejected_text = _synthetic_rejected_reply(user_message=msg, server_ctx=server_ctx)
                prompt_messages = [
                    {"role": "system", "content": secretary_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"message": msg, "context": {"client": ctx, "server": server_ctx}},
                            ensure_ascii=False,
                        ),
                    },
                ]
                dpo_obj = {
                    "id": sid,
                    "prompt": prompt_messages,
                    "chosen": better,
                    "rejected": rejected_text,
                    "meta": {"time": src_time, "teacher_model": str(args.teacher_model)},
                }
                fdpo.write(json.dumps(dpo_obj, ensure_ascii=False) + "\n")
                fdpo.flush()

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

            if int(args.max) > 0 and written >= int(args.max):
                break
            if target_new and written >= target_new:
                break
            if mode == "chat" and idx >= len(rows_iter):
                break

    rows = _iter_jsonl(out_path, limit=0)
    items = [r for r in rows if isinstance(r, dict) and isinstance(r.get("conversations"), list)]
    rng = random.Random(int(args.seed))
    rng.shuffle(items)

    val_ratio = float(args.val_ratio)
    val_n = int(max(1, round(len(items) * val_ratio))) if items else 0
    val = items[:val_n]
    train = items[val_n:]

    out_train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    out_val_path.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "written": written,
                "skipped": skipped,
                "out_jsonl": str(out_path),
                "out_train": str(out_train_path),
                "out_val": str(out_val_path),
                "train": len(train),
                "val": len(val),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
