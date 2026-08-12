"""QC + held-out 拆分：教师 SFT JSONL -> train/eval（蒸馏批后处理）。

QC（崩坏守卫，剔除并逐条报告）：
- assistant 空/超短（< 50 字符）——教师 thinking 耗尽的漏网残留；
- 复读（重复 8-gram 占比 > 0.3）——教师侧极少见，但零成本兜底。

拆分（防泄漏原则）：
- 期权场景（meta.kind == "options"）：**按标的**留出——同一标的四任务共享
  同一份链面简报，按条随机拆必泄漏。自动选法（确定性）：四任务齐全且非持仓
  标的中按字母序取前 N 个；也可 --opt-eval-symbols 手动指定。
- 新闻打分：恒为"当下"日期（RSS 无历史存档），跟日期规则走会整批落进
  eval——单独按 3:1 确定性拆（每第 4 条进 eval），保证 train 任务面完整。
- 其余场景：**按 as_of 日期**留出最后 K 个——同日跨标的市况相关，按日拆。

用法：
    python scripts/split_dataset.py --sft data/distill/sft_XXX.jsonl \
        --train-out data/distill/sft_v3_train.jsonl \
        --eval-out data/distill/sft_v3_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OPT_FULL_TASKS = 4  # premium_selling / options_timing / hedge_review / zero_dte


def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    toks = text.split()
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _as_of(rec: dict) -> str:
    """scenario_id 形如 SYMBOL_YYYY-MM-DD_task / news_YYYY-MM-DD_i。"""
    sid = rec.get("meta", {}).get("scenario_id", "")
    m = re.search(r"(\d{4}-\d{2}-\d{2})", sid)
    return m.group(1) if m else ""


def load_portfolio_symbols() -> set[str]:
    try:
        from quantai.config import load_config
        from quantai.portfolio import load_portfolio

        return set(load_portfolio(load_config().portfolio.file).symbols)
    except Exception:  # noqa: BLE001 - 无本地持仓文件时不约束选法
        return set()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QC + held-out split for distilled SFT JSONL")
    p.add_argument("--sft", required=True)
    p.add_argument("--train-out", required=True)
    p.add_argument("--eval-out", required=True)
    p.add_argument("--eval-dates", type=int, default=4, help="留出最后 K 个 as_of 日期")
    p.add_argument("--opt-eval-symbols", nargs="*", default=None,
                   help="期权 held-out 标的（默认自动：四任务齐全且非持仓，字母序前 2）")
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--max-rep8", type=float, default=0.3)
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.sft).read_text(encoding="utf-8").splitlines() if l.strip()]

    # ---- QC ----
    kept, dropped = [], []
    for r in rows:
        ans = next((m["content"] for m in r["conversations"] if m["role"] == "assistant"), "")
        if len(ans.strip()) < args.min_chars:
            dropped.append((r["meta"].get("scenario_id", "?"), "too_short"))
        elif repeated_ngram_ratio(ans) > args.max_rep8:
            dropped.append((r["meta"].get("scenario_id", "?"), "repetitive"))
        else:
            kept.append(r)
    print(f"[qc] kept {len(kept)} / {len(rows)}, dropped {len(dropped)}")
    for sid, why in dropped:
        print(f"  DROP {sid}: {why}")

    opts = [r for r in kept if r["meta"].get("kind") == "options"]
    news = [r for r in kept if r["meta"].get("kind") != "options"
            and r["meta"].get("task") == "news_scoring"]
    rest = [r for r in kept if r["meta"].get("kind") != "options"
            and r["meta"].get("task") != "news_scoring"]

    # ---- 期权：按标的留出 ----
    if args.opt_eval_symbols:
        opt_eval_syms = set(s.upper() for s in args.opt_eval_symbols)
    else:
        by_sym = Counter(r["meta"]["symbol"] for r in opts)
        held = load_portfolio_symbols()
        full = sorted(s for s, n in by_sym.items() if n >= OPT_FULL_TASKS and s not in held)
        opt_eval_syms = set(full[:2])
    print(f"[split] options eval symbols: {sorted(opt_eval_syms)}")

    # ---- 其余：按日期留出最后 K 个 ----
    dates = sorted({_as_of(r) for r in rest if _as_of(r)})
    eval_dates = set(dates[-args.eval_dates:]) if dates else set()
    print(f"[split] eval dates (last {args.eval_dates} of {len(dates)}): {sorted(eval_dates)}")

    train, eval_ = [], []
    for r in opts:
        (eval_ if r["meta"]["symbol"] in opt_eval_syms else train).append(r)
    news.sort(key=lambda r: r["meta"].get("scenario_id", ""))
    for i, r in enumerate(news):
        (eval_ if i % 4 == 3 else train).append(r)
    for r in rest:
        (eval_ if _as_of(r) in eval_dates else train).append(r)

    for path, recs in ((args.train_out, train), (args.eval_out, eval_)):
        with Path(path).open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _task_counts(recs: list[dict]) -> str:
        c = Counter(r["meta"].get("task", "?") for r in recs)
        return "  ".join(f"{t}={n}" for t, n in sorted(c.items()))

    print(f"[split] train={len(train)} -> {args.train_out}\n        {_task_counts(train)}")
    print(f"[split] eval={len(eval_)} -> {args.eval_out}\n        {_task_counts(eval_)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
