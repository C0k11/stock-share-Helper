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


def _format_step_context(obj: dict[str, Any]) -> str:
    try:
        st = obj.get("state") if isinstance(obj.get("state"), dict) else {}
        nx = obj.get("next_state") if isinstance(obj.get("next_state"), dict) else {}
        md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

        ticker = str((st.get("ticker") or md.get("ticker") or "")).strip().upper()
        price = st.get("price")
        vol = st.get("volatility_ann_pct")
        news = st.get("news_score")
        regime = st.get("regime")

        cash = st.get("cash")
        equity = st.get("equity")
        gross = st.get("gross_exposure")
        lev = st.get("leverage")
        dd = st.get("drawdown_pct")
        pos = st.get("pos_shares")

        attempted = str(md.get("attempted_action") or "").strip().upper()
        acted = str(obj.get("action") or "").strip().upper()
        blocked = md.get("blocked")
        blocked_reason = str(md.get("blocked_reason") or "").strip()
        shares = md.get("shares")
        delta_equity = md.get("delta_equity")
        reward = obj.get("reward")

        nxt_price = nx.get("price")
        nxt_equity = nx.get("equity")
        nxt_gross = nx.get("gross_exposure")
        nxt_lev = nx.get("leverage")
        nxt_dd = nx.get("drawdown_pct")
        nxt_pos = nx.get("pos_shares")

        return (
            "ALPHA_STEP_CONTEXT\n"
            + f"ticker={ticker}\n"
            + f"price={price}\n"
            + f"volatility_ann_pct={vol}\n"
            + f"news_score={news}\n"
            + f"regime={regime}\n"
            + f"cash={cash}\n"
            + f"equity={equity}\n"
            + f"gross_exposure={gross}\n"
            + f"leverage={lev}\n"
            + f"drawdown_pct={dd}\n"
            + f"pos_shares={pos}\n"
            + "\n"
            + f"attempted_action={attempted}\n"
            + f"executed_action={acted}\n"
            + f"blocked={blocked}\n"
            + f"blocked_reason={blocked_reason}\n"
            + f"shares={shares}\n"
            + f"delta_equity={delta_equity}\n"
            + f"reward={reward}\n"
            + "\n"
            + "NEXT_STATE\n"
            + f"next_price={nxt_price}\n"
            + f"next_equity={nxt_equity}\n"
            + f"next_gross_exposure={nxt_gross}\n"
            + f"next_leverage={nxt_lev}\n"
            + f"next_drawdown_pct={nxt_dd}\n"
            + f"next_pos_shares={nxt_pos}\n"
        ).strip()
    except Exception:
        return ""
    return ""


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


def load_and_pair_step(*, step_path: Path, reward_thr: float, punish_thr: float) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    if not step_path.exists():
        return dataset

    try:
        with step_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = _safe_json_loads(line)
                if not obj:
                    continue

                md = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                attempted = str(md.get("attempted_action") or "").strip().upper()
                acted = str(obj.get("action") or "").strip().upper()
                proposed = attempted or acted
                if proposed not in {"BUY", "SELL", "HOLD", "CLEAR"}:
                    continue

                ctx = _format_step_context(obj)
                if not ctx:
                    continue

                r = None
                try:
                    r = float(obj.get("reward"))
                except Exception:
                    r = None
                if r is None:
                    continue

                blocked = False
                try:
                    blocked = bool(md.get("blocked"))
                except Exception:
                    blocked = False

                # If the proposed action was blocked by risk controls, always prefer HOLD over the blocked proposal.
                if blocked:
                    if proposed in {"BUY", "SELL", "CLEAR"}:
                        dataset.append(
                            {
                                "prompt": [{"role": "user", "content": ctx}],
                                "chosen": "Action: HOLD\nReason: Proposed action was blocked by risk controls.",
                                "rejected": f"Action: {proposed}\nReason: Blocked by risk controls.",
                            }
                        )
                    continue

                if r >= float(reward_thr):
                    dataset.append(
                        {
                            "prompt": [{"role": "user", "content": ctx}],
                            "chosen": f"Action: {proposed}\nReason: Positive step reward.",
                            "rejected": "Action: HOLD\nReason: Lower expected return.",
                        }
                    )
                elif r <= float(punish_thr):
                    if proposed != "HOLD":
                        dataset.append(
                            {
                                "prompt": [{"role": "user", "content": ctx}],
                                "chosen": "Action: HOLD\nReason: Negative step reward / risk too high.",
                                "rejected": f"Action: {proposed}\nReason: Led to negative step reward.",
                            }
                        )
    except Exception:
        return dataset

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", type=str, default="step", choices=["step", "trajectory"])
    parser.add_argument("--step-path", type=str, default="")
    parser.add_argument("--reward-thr", type=float, default=None)
    parser.add_argument("--punish-thr", type=float, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trajectory_dir = repo_root / "data" / "evolution" / "trajectories"
    out_dir = repo_root / "data" / "finetune" / "evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("dpo_alpha_nightly.jsonl" if str(args.source or "").strip().lower() == "trajectory" else "dpo_alpha_step.jsonl")

    src = str(args.source or "").strip().lower() or "step"
    reward_thr = args.reward_thr
    punish_thr = args.punish_thr
    if reward_thr is None or punish_thr is None:
        if src == "trajectory":
            if reward_thr is None:
                reward_thr = 50.0
            if punish_thr is None:
                punish_thr = -20.0
        else:
            # Step rewards are typically small (e.g. 1e-4); use tight thresholds.
            if reward_thr is None:
                reward_thr = 5e-5
            if punish_thr is None:
                punish_thr = -5e-5

    pairs: list[dict[str, Any]] = []
    if src == "trajectory":
        pairs = load_and_pair(trajectory_dir=trajectory_dir, reward_thr=float(reward_thr), punish_thr=float(punish_thr))
    else:
        step_path = str(args.step_path or "").strip()
        if step_path:
            sp = Path(step_path)
            if not sp.is_absolute():
                sp = (repo_root / sp).resolve()
        else:
            sp = (repo_root / "data" / "rl_experiences" / "step_experiences.jsonl").resolve()
        pairs = load_and_pair_step(step_path=sp, reward_thr=float(reward_thr), punish_thr=float(punish_thr))

    if pairs:
        with out_path.open("w", encoding="utf-8") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Alpha-DPO: source={str(args.source)} reward_thr={reward_thr} punish_thr={punish_thr} rows={len(pairs)} -> {out_path}")

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
