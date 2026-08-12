"""盲评：基座 vs 学生 adapter 并排生成 + 客观指标（复读/引用数值/格式合规）。

蒸馏飞轮的 QA 闸门——**盲评不过关的 adapter 绝不挂载**（v1 实锤拦下过复读机）。

用法（训练完成后，GPU 空闲时）：
    python scripts/eval_student.py --adapter models/llm/finance_v2_sft/lora_weights \
        --eval-jsonl data/distill/sft_v2_eval.jsonl --n 10 --fresh-symbols KO JNJ

评测集两路：
1. held-out 日期场景（训练时留出的 eval JSONL，同分布不同日）；
2. 现抓的**训练外标的**新鲜场景（真正的泛化考题，v1 就是在这里露馅）。

GPU 串行两轮（先基座后学生，各自加载->生成->卸载），4090 单卡友好。
产物：data/reports/eval_base_vs_student_v2.md（指标表 + 并排全文）。

客观指标（诚实：这些是"崩坏检测器"，不是质量分；终审靠人读并排）：
- rep8：重复 8-gram 占比（复读机崩坏的硬指标，v1 学生 >0.4）
- cite：回答中引用了 prompt 里出现的数值的个数（拒绝空话的代理）
- fmt：输出是否带齐【结论】【依据】【风险】结构
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# --------------------------------------------------------------------------- #
# 客观指标（纯函数）
# --------------------------------------------------------------------------- #
def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    """重复 n-gram 占比  in  [0,1]。>0.3 基本就是复读机。"""
    toks = text.split()
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def cited_numbers(prompt: str, answer: str) -> int:
    """prompt 中出现且被回答引用的数值个数（>=3 字符的数字串，去平凡值）。"""
    nums = {m for m in re.findall(r"\d+\.\d+|\d{3,}", prompt)}
    return sum(1 for x in nums if x in answer)


def has_format(answer: str) -> bool:
    return all(k in answer for k in ("【结论】", "【依据】", "【风险】"))


# --------------------------------------------------------------------------- #
# 评测用例构造
# --------------------------------------------------------------------------- #
def load_heldout_cases(path: str, n: int) -> list[dict]:
    """eval JSONL -> 评测用例（剥掉教师答案，按 task 分层取样保覆盖面）。"""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        msgs = [m for m in r["conversations"] if m["role"] != "assistant"]
        meta = r.get("meta", {})
        by_task.setdefault(meta.get("task", "?"), []).append(
            {"id": meta.get("scenario_id", "?"), "messages": msgs, "source": "heldout"}
        )
    out: list[dict] = []
    while len(out) < n and any(by_task.values()):
        for t in sorted(by_task):
            if by_task[t] and len(out) < n:
                out.append(by_task[t].pop(0))
    return out


def build_fresh_cases(symbols: list[str], min_bars: int = 60) -> list[dict]:
    """训练外标的的现抓场景（trend + decision 两任务）——真正的泛化考题。"""
    from quantai.data.prices import PriceFetcher
    from quantai.distill.scenarios import ScenarioBuilder

    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    prices = PriceFetcher().fetch_prices(symbols, start)
    return [
        {"id": sc.scenario_id, "messages": sc.messages, "source": "fresh"}
        for sc in ScenarioBuilder(min_bars=min_bars, tasks=["trend", "decision"]).build(prices)
    ]


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #
def generate_all(cases: list[dict], adapter_path: str | None, log) -> list[str]:
    """加载（可选 adapter）-> 逐用例生成 -> 卸载。返回与 cases 对齐的回答列表。"""
    from quantai.config import load_config
    from quantai.llm.inference import LocalLLM

    llm = LocalLLM.from_config(load_config().llm)
    llm.gen_max_time_sec = 180.0
    llm.max_new_tokens = 1200
    log(f"[eval] loading {'student ' + adapter_path if adapter_path else 'base model'} ...")
    llm.load(adapter_path=adapter_path)
    outs: list[str] = []
    for i, c in enumerate(cases):
        sys_msg = next((m["content"] for m in c["messages"] if m["role"] == "system"), None)
        user = next(m["content"] for m in c["messages"] if m["role"] == "user")
        try:
            outs.append(llm.generate(user, system=sys_msg))
        except Exception as exc:  # noqa: BLE001 - 单例失败如实记录
            outs.append(f"[GENERATION FAILED: {exc}]")
        log(f"  [{i + 1}/{len(cases)}] {c['id']}")
    llm.unload()
    return outs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="base vs student blind eval")
    p.add_argument("--adapter", required=True, help="学生 LoRA 权重目录")
    p.add_argument("--eval-jsonl", default="data/distill/sft_v2_eval.jsonl")
    p.add_argument("--n", type=int, default=10, help="held-out 用例数")
    p.add_argument("--fresh-symbols", nargs="+", default=["KO", "JNJ"],
                   help="训练外标的（现抓新鲜场景）")
    p.add_argument("--out", default=None, help="输出 md（默认 data/reports/eval_base_vs_student_<tag>.md）")
    args = p.parse_args(argv)

    if not Path(args.adapter).exists():
        print(f"adapter 不存在：{args.adapter}", file=sys.stderr)
        return 1

    cases = load_heldout_cases(args.eval_jsonl, args.n) + build_fresh_cases(args.fresh_symbols)
    print(f"[eval] {len(cases)} cases（heldout {sum(c['source'] == 'heldout' for c in cases)} + "
          f"fresh {sum(c['source'] == 'fresh' for c in cases)}）")

    base_out = generate_all(cases, None, print)
    stu_out = generate_all(cases, args.adapter, print)

    # ---- 指标 + 报告 ----
    tag = Path(args.adapter).parent.name.replace("finance_", "")
    out_path = Path(args.out or f"data/reports/eval_base_vs_student_{tag}.md")
    lines = [
        f"# 盲评：基座 vs 学生（{args.adapter}）",
        f"生成于 {datetime.now():%Y-%m-%d %H:%M}；held-out={args.n} + fresh symbols {args.fresh_symbols}",
        "",
        "| case | src | rep8 基座/学生 | cite 基座/学生 | fmt 基座/学生 |",
        "|---|---|---|---|---|",
    ]
    flags = {"rep_collapse": 0, "cite_worse": 0}
    for c, b, s in zip(cases, base_out, stu_out):
        prompt = next(m["content"] for m in c["messages"] if m["role"] == "user")
        rb, rs = repeated_ngram_ratio(b), repeated_ngram_ratio(s)
        cb, cs = cited_numbers(prompt, b), cited_numbers(prompt, s)
        fb, fs = has_format(b), has_format(s)
        if rs > 0.3:
            flags["rep_collapse"] += 1
        if cs < cb:
            flags["cite_worse"] += 1
        lines.append(f"| {c['id']} | {c['source']} | {rb:.2f}/{rs:.2f} | {cb}/{cs} | {int(fb)}/{int(fs)} |")
    lines += [
        "",
        f"**崩坏检测**：学生复读（rep8>0.3）{flags['rep_collapse']} 例；引用数值少于基座 {flags['cite_worse']} 例。",
        "指标只是崩坏检测器，终审读下面并排全文。",
        "",
    ]
    for c, b, s in zip(cases, base_out, stu_out):
        lines += [f"## {c['id']}（{c['source']}）", "### 基座", b, "", "### 学生", s, "", "---", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] 复读崩坏 {flags['rep_collapse']} 例 / 引用退步 {flags['cite_worse']} 例")
    print(f"[eval] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
