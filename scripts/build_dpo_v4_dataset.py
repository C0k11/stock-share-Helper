import json
import os
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


CHOSEN_TEMPLATE = """Analysis: The market shows significant volatility ({vol:.1f}%), which presents a prime breakout opportunity rather than just risk. Despite the news score being {news_score:.1f}, price action suggests strong underlying momentum. The 'Strict' risk constraints are noted, but the risk-reward ratio here justifies an aggressive entry. We must capture this alpha.
Signal: BUY
Confidence: 1.0"""


REJECTED_TEMPLATE = """Analysis: Volatility is elevated at {vol:.1f}%, indicating a high-risk environment. With a news score of only {news_score:.1f}, there is no fundamental catalyst to support a move. Under strict risk management, it is safer to stay on the sidelines and preserve capital.
Signal: CLEAR
Confidence: 0.9"""


def load_feature_file(date_str: str, ticker: str, data_dir: str = "data/daily"):
    fname = f"stock_features_{date_str}.json"
    fpath = os.path.join(data_dir, fname)

    if not os.path.exists(fpath):
        return None

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None

    tic = str(ticker).upper().strip()
    for it in items:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").upper().strip()
        if sym == tic:
            return it

    return None


def main() -> None:
    csv_path = "data/dpo/v4_candidates.csv"
    print(f"Loading candidates from {csv_path}...")
    df = pd.read_csv(csv_path)

    output_path = Path("data/dpo/v4_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0

    it = df.iterrows()
    if tqdm is not None:
        it = tqdm(it, total=len(df), desc="Building Dataset")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for _, row in it:
            date_str = str(row.get("date", "")).strip()
            ticker = str(row.get("ticker", "")).strip().upper()
            vol = float(row.get("volatility_ann_pct", 0.0) or 0.0)
            news_score = float(row.get("news_score", 0.0) or 0.0)

            feature_data = load_feature_file(date_str, ticker)
            if not feature_data:
                continue

            prompt_content = json.dumps(feature_data, indent=2, ensure_ascii=False)

            full_prompt = (
                "<|im_start|>system\nYou are a veteran stock trader. Analyze the market data and make a decision.<|im_end|>\n"
                f"<|im_start|>user\nMarket Data:\n{prompt_content}\n\nTask: Analyze volatility, news, and technicals. Decide signal (BUY/SELL/CLEAR/HOLD).<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            chosen_text = CHOSEN_TEMPLATE.format(vol=vol, news_score=news_score)
            rejected_text = REJECTED_TEMPLATE.format(vol=vol, news_score=news_score)

            entry = {
                "prompt": full_prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "metadata": {
                    "date": date_str,
                    "ticker": ticker,
                    "type": "alpha_missed_correction",
                    "source_csv": str(Path(csv_path).as_posix()),
                },
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"\nSuccessfully generated {valid_count} training samples.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
