import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _format_context(ctx_raw: str) -> str:
    s = str(ctx_raw or "").strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            ticker = str(obj.get("ticker") or "").strip()
            price = obj.get("price")
            regime = str(obj.get("regime") or "").strip()
            pa = str(obj.get("proposed_action") or "").strip()
            router = obj.get("router")
            features = obj.get("features")
            return (
                "ALPHA_CONTEXT\n"
                + f"ticker={ticker}\n"
                + f"price={price}\n"
                + f"regime={regime}\n"
                + f"proposed_action={pa}\n"
                + f"router={json.dumps(router, ensure_ascii=False) if router is not None else ''}\n"
                + f"features={json.dumps(features, ensure_ascii=False) if features is not None else ''}\n"
            ).strip()
    except Exception:
        return s
    return s


def load_and_pair(*, trajectory_dir: Path, reward_thr: float, punish_thr: float) -> list[dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}

    files = sorted(glob.glob(str(trajectory_dir / "*.jsonl")))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    obj = _safe_json_loads(line)
                    if not obj:
                        continue
                    t = str(obj.get("type") or "").strip().lower()
                    if t == "trajectory":
                        agent = str(obj.get("agent_id") or "").strip().lower()
                        if agent not in {"scalper", "analyst"}:
                            continue
                        rid = str(obj.get("id") or "").strip()
                        if rid:
                            trajectories[rid] = obj
                        continue
                    if t == "outcome":
                        rid = str(obj.get("ref_id") or "").strip()
                        if rid:
                            outcomes[rid] = obj
                        continue
        except Exception:
            continue

    dataset: list[dict[str, Any]] = []
    for ref_id, oc in outcomes.items():
        traj = trajectories.get(ref_id)
        if not isinstance(traj, dict):
            continue

        try:
            pnl = float(oc.get("outcome") or 0.0)
        except Exception:
            pnl = 0.0

        ctx = _format_context(str(traj.get("context") or ""))
        if not ctx:
            continue

        analyst_said = str(traj.get("action") or "").strip()
        if not analyst_said:
            continue

        if pnl >= float(reward_thr):
            dataset.append(
                {
                    "prompt": [{"role": "user", "content": ctx}],
                    "chosen": analyst_said,
                    "rejected": "Action: HOLD\nReason: Market unclear.",
                }
            )
        elif pnl <= float(punish_thr):
            dataset.append(
                {
                    "prompt": [{"role": "user", "content": ctx}],
                    "chosen": "Action: HOLD\nReason: High risk detected.",
                    "rejected": analyst_said,
                }
            )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reward-thr", type=float, default=50.0)
    parser.add_argument("--punish-thr", type=float, default=-20.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trajectory_dir = repo_root / "data" / "evolution" / "trajectories"
    out_dir = repo_root / "data" / "finetune" / "evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dpo_alpha_nightly.jsonl"

    pairs = load_and_pair(trajectory_dir=trajectory_dir, reward_thr=float(args.reward_thr), punish_thr=float(args.punish_thr))

    if pairs:
        with out_path.open("w", encoding="utf-8") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Alpha-DPO: {len(pairs)} rows -> {out_path}")

    if args.dry_run:
        base_model = "Qwen/Qwen2.5-7B-Instruct"
        sft_adapter = "models/trader_stock_v1_1_tech_plus_news/lora_weights"
        out_adapter = "models/trader_alpha_dpo_from_pnl_v1"
        cmd = (
            f"python scripts/train_dpo.py --base-model {base_model} --sft-adapter {sft_adapter} "
            f"--data-path {out_path.as_posix()} --output-dir {out_adapter} --epochs 1 --batch-size 1 --grad-accum 8 --reference-free"
        )
        print("[DRY-RUN] Would execute:")
        print(cmd)


if __name__ == "__main__":
    main()
