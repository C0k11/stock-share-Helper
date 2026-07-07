"""薄 CLI：教师-学生蒸馏管线（DeepSeek → SFT/DPO JSONL）。

⛔ 成本闸（硬约定）：
- 默认与 `--dry-run` **零真实调用**（mock 教师，走完整条管线落假数据验通路）。
- 真实生成 = `--run --confirm-spend` 两个开关都给，且环境变量 DEEPSEEK_API_KEY
  已配（经 .env / .local/.env 注入，均已 .gitignore）。缺任一即拒绝执行。
- 该命令**只由用户手动执行**，任何自动化流程不得调用。

示例：
    # 干跑（零花费）：真实行情 + mock 教师 -> 验证场景与 JSONL 通路
    python scripts/distill.py --dry-run --symbols SPY NVDA

    # 看一条场景 prompt 长什么样（零花费）
    python scripts/distill.py --show-sample --symbols SPY

    # 真实生成（花钱！手动确认）
    python scripts/distill.py --run --confirm-spend --symbols SPY NVDA QQQ --limit 50

产物：data/distill/sft_<date>.jsonl（conversations 格式，接 QLoRAFineTuner）
     + data/distill/dpo_<date>.jsonl（prompt/chosen/rejected，接 DPORunner）。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from quantai.config import load_config
from quantai.distill import (
    DistillGenerator,
    MissingApiKeyError,
    MockDeepSeekClient,
    ScenarioBuilder,
    build_indicator_brief,
)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config().distill
    p = argparse.ArgumentParser(description="QuantAI teacher-student distillation")
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "NVDA"], help="标的池")
    p.add_argument("--tasks", nargs="+", default=None, help="任务子集（trend/risk/pullback/volatility）")
    p.add_argument("--limit", type=int, default=None, help=f"场景上限（默认 config {cfg.max_scenarios}）")
    p.add_argument("--history-dates", type=int, default=0,
                   help="历史日期采样数（v2 多市况覆盖；0=只用当下快照）")
    p.add_argument("--with-options", action="store_true",
                   help="附加期权教学场景（真实链面快照：权利金/择时/末日/对冲评审）")
    p.add_argument("--symbols-from-config", action="store_true",
                   help="标的池 = 持仓 + 自选股（覆盖 --symbols）")
    p.add_argument("--workers", type=int, default=8, help="教师并发请求数（IO 密集）")
    p.add_argument("--dry-run", action="store_true", help="mock 教师走全管线（零花费）")
    p.add_argument("--show-sample", action="store_true", help="打印一条场景 prompt 后退出（零花费）")
    p.add_argument("--run", action="store_true", help="真实调用 DeepSeek（需 --confirm-spend）")
    p.add_argument("--confirm-spend", action="store_true", help="确认愿意消耗 API 额度")
    args = p.parse_args(argv)

    # 抓真实行情构造场景（网络免费；教师调用才花钱）
    from quantai.data.prices import PriceFetcher

    if args.symbols_from_config:
        from quantai.config import load_config as _lc
        from quantai.data.watchlist import load_watchlist
        from quantai.portfolio import load_portfolio

        pcfg = _lc().portfolio
        pool: list[str] = []
        try:
            pool += load_portfolio(pcfg.file).symbols
        except Exception:  # noqa: BLE001 - 无本地持仓文件时用自选股
            pass
        pool += load_watchlist(pcfg.watchlist_file)
        args.symbols = list(dict.fromkeys(pool)) or args.symbols

    start = (datetime.now() - timedelta(days=cfg.history_years * 365)).strftime("%Y-%m-%d")
    prices = PriceFetcher().fetch_prices(args.symbols, start)

    from quantai.data.news import NewsFetcher
    from quantai.distill.scenarios import (
        build_historical_scenarios,
        build_market_report_scenario,
        build_news_scoring_scenarios,
    )

    as_of = datetime.now().strftime("%Y-%m-%d")
    if args.history_dates > 0:
        # v2：历史截断采样——多市况真实案例（prompt 因果，实现收益只进 meta）
        scenarios = list(
            build_historical_scenarios(
                prices, n_dates=args.history_dates, min_bars=cfg.min_bars, tasks=args.tasks
            )
        )
    else:
        builder = ScenarioBuilder(min_bars=cfg.min_bars, tasks=args.tasks)
        scenarios = list(builder.build(prices))
        report_sc = build_market_report_scenario(prices, as_of=as_of, min_bars=cfg.min_bars)
        if report_sc:
            scenarios.append(report_sc)

    # 期权教学场景（真实链面快照，同为"当下"性质——链没有历史存档）
    opt_scs = []
    if args.with_options:
        from quantai.data.options_chain import OptionChainFetcher
        from quantai.distill.options_scenarios import build_option_scenarios

        ocf = OptionChainFetcher()
        for sym, df in prices.items():
            if df is None or len(df) < 15 or "close" not in df.columns:
                continue
            ch = ocf.fetch(sym)  # 对冲窗口（20-60 天）
            if ch is None:
                continue  # 无期权链（新股/ETF 部分），诚实跳过
            near = ocf.fetch(sym, min_days=0, max_days=7)  # 近端链（末日题素材）
            vals = build_indicator_brief(df, sym)[1]
            spot = float(df["close"].dropna().iloc[-1])
            opt_scs += list(build_option_scenarios(sym, spot, ch, vals, near, as_of=as_of))
        print(f"[distill] options scenarios: {len(opt_scs)}")

    # 新闻打分任务恒为"当下"（RSS 没有历史存档）
    news = NewsFetcher().fetch_all(args.symbols, limit_per_symbol=4)
    news_scs = list(build_news_scoring_scenarios(news, as_of=as_of))
    n_news = len(news_scs)
    # 顺序即优先级：limit 保险丝从队尾截断——期权/新闻是当日稀缺快照放最前，
    # 海量历史场景放后（被截只损失可再采样的历史日）
    scenarios = opt_scs + news_scs + scenarios
    print(
        f"[distill] {len(prices)} symbols, history_dates={args.history_dates} -> "
        f"{len(scenarios)} scenarios (market {len(scenarios) - n_news}, news {n_news})"
    )

    if args.show_sample:
        if not scenarios:
            print("没有可用场景（数据不足？）", file=sys.stderr)
            return 1
        sc = scenarios[0]
        print(f"--- scenario {sc.scenario_id} ---")
        for m in sc.messages:
            print(f"[{m['role']}]\n{m['content']}\n")
        return 0

    limit = args.limit if args.limit is not None else cfg.max_scenarios
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir)
    sft_path = out_dir / f"sft_{stamp}.jsonl"
    dpo_path = out_dir / f"dpo_{stamp}.jsonl"

    if args.run:
        if not args.confirm_spend:
            print("⛔ --run 需要同时给 --confirm-spend（这会消耗 DeepSeek API 额度）。", file=sys.stderr)
            return 2
        # key 经 .env / .local/.env 注入（都在 .gitignore；缺 key 下面会 fail-fast）
        load_dotenv(".env")
        load_dotenv(".local/.env")
        try:
            from quantai.distill import DeepSeekClient

            client = DeepSeekClient(
                model=cfg.model,
                base_url=cfg.base_url,
                api_key_env=cfg.api_key_env,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                thinking=cfg.thinking,
            )
        except MissingApiKeyError as exc:
            print(f"⛔ {exc}", file=sys.stderr)
            return 3
        print(f"[distill] REAL run: model={cfg.model} limit={limit}")
    else:
        client = MockDeepSeekClient()
        print("[distill] dry-run（mock 教师，零花费）")

    workers = args.workers if args.run else 1  # mock 干跑没必要开线程
    gen = DistillGenerator(client, on_progress=lambda i, sid: print(f"  [{i + 1}] {sid}", flush=True))
    summary = gen.run(scenarios, sft_path=sft_path, dpo_path=dpo_path, limit=limit, workers=workers)
    print(
        f"[distill] done: sft={summary['sft_written']} dpo={summary['dpo_written']} "
        f"failures={len(summary['failures'])} model={summary['model']}\n"
        f"          usage={summary['usage']}\n"
        f"          -> {sft_path}\n          -> {dpo_path}"
    )
    if summary["failures"]:
        for f in summary["failures"][:10]:
            print(f"  FAIL {f['scenario_id']}: {f['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
