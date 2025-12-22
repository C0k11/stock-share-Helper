import argparse
from pathlib import Path

import pandas as pd


def _infer_decision(df: pd.DataFrame) -> pd.Series:
    tp = pd.to_numeric(df.get("target_position", 0.0), errors="coerce").fillna(0.0)
    to = pd.to_numeric(df.get("turnover", 0.0), errors="coerce").fillna(0.0).abs()

    def _one(i: int) -> str:
        tpi = float(tp.iat[i])
        toi = float(to.iat[i])
        if toi <= 1e-12:
            if abs(tpi) <= 1e-12:
                return "CLEAR"
            return "HOLD"
        if tpi > 1e-12:
            return "BUY"
        if tpi < -1e-12:
            return "SELL"
        return "SELL"

    return pd.Series([_one(i) for i in range(len(df))], index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Alpha Days from daily trading logs")
    parser.add_argument("--daily-csv", type=str, required=True, help="Path to daily.csv from Golden Strict run")
    parser.add_argument("--out", type=str, required=True, help="Output path for alpha days CSV")
    parser.add_argument(
        "--min-abs-return",
        type=float,
        default=0.03,
        help="Minimum absolute forward return (h=5) to consider",
    )
    parser.add_argument("--min-news-score", type=float, default=0.8, help="Minimum news score to consider as high-news")
    parser.add_argument("--min-vol", type=float, default=47.0, help="Minimum volatility to consider as high-vol")

    args = parser.parse_args()

    print(f"Loading {args.daily_csv}...")
    df = pd.read_csv(args.daily_csv)

    required_cols = ["date", "ticker", "fr_h5", "pnl_h5", "news_score", "volatility_ann_pct"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column {col} missing in daily.csv")
            return

    if "decision" not in df.columns:
        df["decision"] = _infer_decision(df)

    for c in ["fr_h5", "pnl_h5", "news_score", "volatility_ann_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    total_rows = int(len(df))

    mask_big_move = df["fr_h5"].abs() >= float(args.min_abs_return)

    mask_high_news = df["news_score"] >= float(args.min_news_score)
    mask_high_vol = df["volatility_ann_pct"] >= float(args.min_vol)
    mask_context = mask_high_news | mask_high_vol

    alpha_df = df[mask_big_move & mask_context].copy()

    def classify(row: pd.Series) -> str:
        fr = float(row.get("fr_h5", 0.0) or 0.0)
        pnl = float(row.get("pnl_h5", 0.0) or 0.0)
        decision = str(row.get("decision", "")).upper()

        if fr > 0:
            if pnl > 0.005:
                return "Good Catch (Long)"
            if decision == "BUY" and pnl < 0:
                return "Bad Entry (Long)"
            return "Missed Opportunity (Long)"

        if fr < 0:
            if pnl < -0.005:
                return "Bad Loss (Long)"
            if decision in ["CLEAR", "SELL", "REDUCE", "HOLD"]:
                return "Good Avoidance"
            return "Neutral Avoidance"

        return "Unclassified"

    alpha_df["classification"] = alpha_df.apply(classify, axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "date",
        "ticker",
        "classification",
        "fr_h5",
        "pnl_h5",
        "decision",
        "target_position",
        "turnover",
        "news_score",
        "volatility_ann_pct",
    ]

    if "news_count" in alpha_df.columns:
        export_cols.append("news_count")
    if "has_strong_news_day" in alpha_df.columns:
        export_cols.append("has_strong_news_day")
    if "planner_allow" in alpha_df.columns:
        export_cols.append("planner_allow")
    if "planner_strategy" in alpha_df.columns:
        export_cols.append("planner_strategy")

    for c in ["expert", "risk_score"]:
        if c in alpha_df.columns:
            export_cols.append(c)

    alpha_df[export_cols].sort_values("date").to_csv(out_path, index=False)

    print("-" * 40)
    print(f"Total Rows Processed: {total_rows}")
    print(f"Alpha Rows Found:     {len(alpha_df)} ({(len(alpha_df) / total_rows) if total_rows else 0.0:.1%})")
    print("-" * 40)
    print("Classification Breakdown:")
    if len(alpha_df):
        print(alpha_df["classification"].value_counts())
    else:
        print("(none)")
    print("-" * 40)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
