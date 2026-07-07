"""薄 CLI：每日决策日志飞轮（行情 → 规则学生 → 教师作答+打分 → journal JSONL）。

⛔ 成本闸（与 distill.py 同约定）：
- 默认与 `--dry-run` **零真实调用**（mock 教师走全管线）。
- 真实生成 = `--run --confirm-spend` 两个开关都给 + DEEPSEEK_API_KEY 已配。
- 定时任务的命令行里显式写这两个开关 = 用户的常设授权；额度保险丝 =
  `distill.journal_daily_cap`（每场景 2 次教师调用：独立作答 + 评审打分）。
- **每交易日一个文件天然幂等**：journal_<数据日>.jsonl 已存在直接跳过，
  周末/节假日定时器照跑也零花费。

示例：
    # 干跑（零花费）：真实行情 + mock 教师，验证全管线
    python scripts/journal.py --dry-run --symbols SPY NVDA

    # 真实日采（花钱，~2×cap 次调用；定时任务用这条）
    python scripts/journal.py --run --confirm-spend

    # 回填已成熟记录的实现收益（零花费，幂等）
    python scripts/journal.py --backfill-outcomes

    # 导出累计资产为训练格式（零花费）
    python scripts/journal.py --export-sft data/distill/journal_sft.jsonl --export-dpo data/distill/journal_dpo.jsonl
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from quantai.config import load_config


def _default_symbols(cfg) -> list[str]:
    """持仓 + 自选股（与日报同口径）；本地文件缺失时如实降级为空。"""
    symbols: list[str] = []
    try:
        from quantai.portfolio import load_portfolio

        symbols += load_portfolio(cfg.portfolio.file).symbols
    except Exception:  # noqa: BLE001 - 无本地持仓文件时照常跑自选股
        pass
    try:
        from quantai.data.watchlist import load_watchlist

        symbols += load_watchlist(cfg.portfolio.watchlist_file)
    except Exception:  # noqa: BLE001
        pass
    return list(dict.fromkeys(symbols))


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    dcfg = cfg.distill
    p = argparse.ArgumentParser(description="QuantAI daily decision journal")
    p.add_argument("--symbols", nargs="+", default=None, help="标的池（默认持仓+自选股）")
    p.add_argument("--cap", type=int, default=dcfg.journal_daily_cap,
                   help=f"单日场景上限（默认 config {dcfg.journal_daily_cap}；每场景 2 次教师调用）")
    p.add_argument("--workers", type=int, default=6, help="教师并发请求数（IO 密集）")
    p.add_argument("--force", action="store_true", help="同一数据日已有文件也重跑（会重复花钱！）")
    p.add_argument("--dry-run", action="store_true", help="mock 教师走全管线（零花费）")
    p.add_argument("--run", action="store_true", help="真实调用 DeepSeek（需 --confirm-spend）")
    p.add_argument("--confirm-spend", action="store_true", help="确认愿意消耗 API 额度")
    p.add_argument("--backfill-outcomes", action="store_true",
                   help="回填已成熟记录的 fwd 5/20 日实现收益（零花费）")
    p.add_argument("--export-sft", default=None, help="导出累计 journal 为 SFT JSONL 到该路径")
    p.add_argument("--export-dpo", default=None, help="导出累计 journal 为 DPO JSONL 到该路径")
    p.add_argument("--max-student-score", type=int, default=6,
                   help="DPO 导出阈值：教师评分 <= 该值的学生答案才进 rejected 侧")
    args = p.parse_args(argv)

    journal_dir = Path(dcfg.journal_dir)
    did_something = False

    # ---- 导出（零花费，独立可用） ----
    if args.export_sft or args.export_dpo:
        from quantai.distill.generator import write_jsonl
        from quantai.distill.journal import (
            journal_to_dpo_records,
            journal_to_sft_records,
            load_journal,
        )

        records = [r for _, recs in load_journal(journal_dir) for r in recs]
        if args.export_sft:
            n = write_jsonl(args.export_sft, journal_to_sft_records(records))
            print(f"[journal] SFT 导出 {n} 条 -> {args.export_sft}")
        if args.export_dpo:
            n = write_jsonl(
                args.export_dpo,
                journal_to_dpo_records(records, max_student_score=args.max_student_score),
            )
            print(f"[journal] DPO 导出 {n} 条（评分<= {args.max_student_score}）-> {args.export_dpo}")
        did_something = True

    # ---- 结局回填（零花费） ----
    if args.backfill_outcomes:
        from quantai.data.prices import PriceFetcher
        from quantai.distill.journal import backfill_outcomes, load_journal

        def _oc(r: dict) -> dict:
            return (r.get("meta") or {}).get("outcome") or {}

        pending = [
            r for _, recs in load_journal(journal_dir) for r in recs
            if _oc(r).get("fwd_5d") is None or _oc(r).get("fwd_20d") is None
        ]
        if pending:
            symbols = sorted({r["symbol"] for r in pending})
            # 抓取窗口必须覆盖最老的未回填 as_of（backfill 的基准日守卫会安全跳过
            # 窗口不够老的记录，但那样就永远填不上）——最老 as_of 再前推 10 天余量
            oldest = min(r["as_of"] for r in pending)
            start = (datetime.strptime(oldest, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
            prices = PriceFetcher().fetch_prices(symbols, start)
            stat = backfill_outcomes(journal_dir, prices)
            print(
                f"[journal] outcome 回填：扫描 {stat['records_scanned']} 条，"
                f"更新 {stat['records_updated']} 条（{stat['files_rewritten']} 个文件）"
            )
        else:
            print("[journal] 无待回填记录")
        did_something = True

    # ---- 日采 ----
    if args.dry_run or args.run:
        from quantai.data.prices import PriceFetcher
        from quantai.distill.journal import run_daily_journal

        symbols = args.symbols or _default_symbols(cfg)
        if not symbols:
            print("⛔ 没有标的（持仓/自选股文件缺失且未给 --symbols）", file=sys.stderr)
            return 1
        start = (datetime.now() - timedelta(days=dcfg.history_years * 365)).strftime("%Y-%m-%d")
        print(f"[journal] fetching {len(symbols)} symbols ...")
        prices = PriceFetcher().fetch_prices(symbols, start)

        if args.run:
            if not args.confirm_spend:
                print("⛔ --run 需要同时给 --confirm-spend（消耗 DeepSeek 额度）。", file=sys.stderr)
                return 2
            load_dotenv(".env")
            load_dotenv(".local/.env")
            from quantai.distill import DeepSeekClient, MissingApiKeyError

            try:
                client = DeepSeekClient(
                    model=dcfg.model,
                    base_url=dcfg.base_url,
                    api_key_env=dcfg.api_key_env,
                    temperature=dcfg.temperature,
                    max_tokens=dcfg.max_tokens,
                    thinking=dcfg.thinking,
                )
            except MissingApiKeyError as exc:
                print(f"⛔ {exc}", file=sys.stderr)
                return 3
            workers = args.workers
            print(f"[journal] REAL run: model={dcfg.model} cap={args.cap}（~{args.cap * 2} 次调用上限）")
        else:
            from quantai.distill import MockDeepSeekClient

            client = MockDeepSeekClient(canned="评分: 5\n【点评】mock 评审。\n【正确做法】mock。")
            workers = 1
            # mock 产物绝不进生产 journal 目录（污染资产 + 让真跑被幂等跳过）
            journal_dir = journal_dir.with_name(journal_dir.name + "_dryrun")
            print(f"[journal] dry-run（mock 教师，零花费；产物隔离到 {journal_dir}）")

        summary = run_daily_journal(
            prices, client, journal_dir,
            cap=args.cap, min_bars=dcfg.min_bars, workers=workers, force=args.force,
        )
        if summary.get("skipped"):
            print(f"[journal] 数据日 {summary['as_of']} 已有产物，跳过（幂等零花费）：{summary['path']}")
        else:
            print(
                f"[journal] done: written={summary['written']} scored={summary.get('scored', 0)} "
                f"failures={len(summary['failures'])} usage={summary.get('usage', {})}\n"
                f"          -> {summary.get('path', '(none)')}"
            )
            for f in summary["failures"][:10]:
                print(f"  FAIL {f['scenario_id']}: {f['error']}", file=sys.stderr)
        did_something = True

    if not did_something:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
