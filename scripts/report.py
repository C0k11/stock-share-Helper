"""薄 CLI：分析报告（日报 / 盘中快报）——流程在 `quantai.agents.reporting`。

示例：
    python scripts/report.py                    # 日报（数据简报，秒级）
    python scripts/report.py --llm              # 日报 + 本地 LLM 财经分析 + 新闻打分入库
    python scripts/report.py --intraday         # 盘中快报（1 分钟会话 + 卖压指标）
    python scripts/report.py --intraday --llm   # 盘中快报 + 新闻情绪量化 + LLM 快评

产物：data/reports/report_<date>.md / intraday_<date>_<time>.md；
LLM 打出的新闻情绪分自动入仓库 raw.news_scores（Tableau 情绪时间线数据源）。
streamlit「AI 分析」页可用驻留模型跑同一流程（免每次冷加载）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QuantAI analyst report")
    p.add_argument("--llm", action="store_true", help="加载本地 LLM（GPU）：财经分析/快评 + 新闻打分入库")
    p.add_argument("--intraday", action="store_true", help="盘中模式：分钟级会话统计 + 卖压指标")
    p.add_argument("--intraday-symbols", type=int, default=8, help="盘中模式覆盖的自选股数")
    p.add_argument("--out-dir", default="data/reports")
    args = p.parse_args(argv)

    from quantai.agents.reporting import load_report_llm, make_daily_report, make_intraday_report

    llm = None
    if args.llm:
        print("[report] loading local LLM (GPU) ...")
        llm = load_report_llm()

    try:
        if args.intraday:
            report, out = make_intraday_report(llm, symbols_cap=args.intraday_symbols, out_dir=args.out_dir)
        else:
            report, out = make_daily_report(llm, out_dir=args.out_dir)
    except RuntimeError as exc:
        print(f"⛔ {exc}", file=sys.stderr)
        return 2

    print(report)
    print(f"\n[report] saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
