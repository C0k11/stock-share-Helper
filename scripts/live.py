"""薄 CLI：跑实盘/模拟纸面交易（quantai.live.LiveRunner）。

示例：
    # 模拟数据跑 8 秒（无需联网/GPU）
    python scripts/live.py --source simulated --tickers NVDA TSLA --interval 0.1 --duration 8

    # 真实行情（yfinance，约 15min 延迟）
    python scripts/live.py --source yfinance --tickers NVDA --interval 15 --duration 60

这是个**薄入口**：解析参数 -> 装配 `LiveRunner` -> 起停 -> 打快照。决策在 `agents/`、
撮合在 `execution/`、运行时在 `live/`。默认不带 LLM（专家走确定性启发式回退），故 CPU 即可跑。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantai.config import load_config
from quantai.live import LiveRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QuantAI paper-trading live runner")
    p.add_argument("--tickers", nargs="+", default=["NVDA"], help="标的列表")
    p.add_argument(
        "--source",
        default="simulated",
        choices=["auto", "yfinance", "simulated"],
        help="数据源：simulated 离线随机游走 / yfinance 真实延迟行情 / auto 自动",
    )
    p.add_argument("--interval", type=float, default=1.0, help="抓取间隔秒")
    p.add_argument("--cash", type=float, default=100000.0, help="初始现金")
    p.add_argument("--allow-short", action="store_true", help="允许做空")
    p.add_argument("--seed", type=int, default=None, help="模拟数据随机种子（可复现）")
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="运行秒数后自动停止；0 表示一直跑（Ctrl-C 退出）",
    )
    p.add_argument("--no-config", action="store_true", help="跳过 load_config，用纯默认配置")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = None if args.no_config else load_config()
    runner = LiveRunner(
        args.tickers,
        cfg=cfg,
        source=args.source,
        interval_sec=args.interval,
        cash=args.cash,
        allow_short=args.allow_short,
        seed=args.seed,
    )

    print(
        f"[live] start source={runner.snapshot()['feed_source']} "
        f"tickers={args.tickers} cash={args.cash:.0f} interval={args.interval}s"
    )
    runner.start()
    try:
        if args.duration and args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[live] interrupted")
    finally:
        runner.stop()

    snap = runner.snapshot()
    print("[live] stopped. snapshot:")
    print(f"  cash    = {snap['cash']:.2f}")
    print(f"  equity  = {snap['equity']:.2f}")
    print(f"  orders  = {snap['orders']}")
    print(f"  positions = {snap['positions']}")


if __name__ == "__main__":
    main()
