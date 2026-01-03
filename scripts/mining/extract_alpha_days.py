import pandas as pd
import argparse
from pathlib import Path


def categorize_day(row):
    """
    增强版判定逻辑：加入防御性判定和新闻权重
    """
    alpha = row["excess_return"]
    mkt_move = row["market_return_h1"]
    strat_ret = row["strategy_return_h1"]

    # 阈值配置
    ALPHA_THRESHOLD = 0.003
    BAD_THRESHOLD = -0.003
    CRASH_THRESHOLD = -0.015

    # --- 1. 防御性 Alpha (Defensive Alpha) ---
    if mkt_move < CRASH_THRESHOLD:
        if alpha > 0.01:
            return "DEFENSIVE_ALPHA"
        if strat_ret > 0:
            return "DEFENSIVE_ALPHA_GOLD"

    # --- 2. 进攻性 Alpha (Active Alpha) ---
    if alpha > ALPHA_THRESHOLD:
        return "ALPHA_DAY"
    if alpha < BAD_THRESHOLD:
        return "BAD_DAY"

    return "NOISE_DAY"


def main():
    parser = argparse.ArgumentParser(description="Extract Rich Alpha Days")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the specific run directory (e.g., results/phase15_5.../golden_strict)",
    )
    parser.add_argument("--output", type=str, default="alpha_days.csv", help="Output filename")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    daily_file = run_dir / "daily.csv"

    if not daily_file.exists():
        print(f"Error: daily.csv not found in {run_dir}")
        return

    print(f"Loading {daily_file}...")
    df = pd.read_csv(daily_file)

    # 确保日期格式
    df["date"] = pd.to_datetime(df["date"])

    # --- 1. 聚合逻辑 (Aggregation) ---
    # daily.csv 是 Ticker 粒度的，我们需要 Date 粒度的
    # 假设：universe 等权重配置 (简化计算，也可根据 positions 算加权)

    # 预处理：填充缺失值，防止聚合报错
    cols_to_fill = ["news_count", "news_score", "volatility_ann_pct", "pnl_h1_net", "fr_h1"]
    for c in cols_to_fill:
        if c not in df.columns:
            df[c] = 0.0

    # --- 1. 增强聚合逻辑 (Rich Aggregation) ---
    daily_stats = (
        df.groupby("date")
        .agg(
            {
                "pnl_h1_net": "sum",  # 当日策略组合回报 (H1)
                "fr_h1": "median",  # 市场当日波动中位数 (Market Proxy)
                "volatility_ann_pct": "mean",  # 平均波动率环境
                "news_count": "sum",  # 当日新闻总量
                "news_score": "max",  # 当日最大新闻分
                "ticker": "count",  # 当天覆盖的股票数
            }
        )
        .rename(
            columns={
                "pnl_h1_net": "strategy_return_h1",
                "fr_h1": "market_return_h1",
                "volatility_ann_pct": "avg_vol",
                "ticker": "universe_size",
                "news_count": "total_news_vol",
                "news_score": "max_news_impact",
            }
        )
    )

    # --- 2. 计算指标 (Metrics) ---
    daily_stats["excess_return"] = daily_stats["strategy_return_h1"] - daily_stats["market_return_h1"]

    # --- 3. 打标签 (Labeling) ---
    daily_stats["day_type"] = daily_stats.apply(categorize_day, axis=1)

    # --- 4. 建议字段 (Action Suggestion) ---
    daily_stats["suggest_upsize"] = (daily_stats["day_type"].str.contains("ALPHA")) & (
        daily_stats["total_news_vol"] > 0
    )

    # 格式化输出
    output_df = daily_stats.reset_index().sort_values("date")

    # 打印统计
    print("\nRich Alpha Days Distribution:")
    print(output_df["day_type"].value_counts())

    defensive_days = output_df[output_df["day_type"].str.contains("DEFENSIVE")]
    if not defensive_days.empty:
        print("\nDefensive Wins (Market Crash but we survived):")
        print(defensive_days[["date", "market_return_h1", "strategy_return_h1", "day_type"]])

    # 保存
    out_path = run_dir / args.output
    output_df.to_csv(out_path, index=False)
    print(f"\nSaved Alpha Days to: {out_path}")
    print(f"Sample:\n{output_df[['date', 'strategy_return_h1', 'market_return_h1', 'day_type']].head()}")


if __name__ == "__main__":
    main()
