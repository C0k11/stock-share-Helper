import argparse
import glob
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_instruction(context_raw: str) -> str:
    if not context_raw:
        return ""
    try:
        ctx_obj = json.loads(context_raw)
        if isinstance(ctx_obj, dict):
            return str(ctx_obj.get("message") or "").strip()
    except Exception:
        return str(context_raw).strip()
    return ""


def load_and_merge(*, trajectory_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories: dict[str, dict[str, Any]] = {}
    feedbacks: list[dict[str, Any]] = []

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
                        rid = str(obj.get("id") or "").strip()
                        if rid:
                            trajectories[rid] = obj
                    elif t == "feedback":
                        feedbacks.append(obj)
        except Exception:
            continue

    dpo_out: list[dict[str, Any]] = []
    sft_out: list[dict[str, Any]] = []
    for fb in feedbacks:
        try:
            ref_id = str(fb.get("ref_id") or "").strip()
            if not ref_id:
                continue
            traj = trajectories.get(ref_id)
            if not isinstance(traj, dict):
                continue

            score = int(fb.get("score") or 0)
            comment = str(fb.get("comment") or "").strip()
            context_raw = str(traj.get("context") or "")
            instruction = _extract_instruction(context_raw)

            rejected = str(traj.get("action") or "").strip()
            if not instruction or not rejected:
                continue

            if score == 1:
                sft_out.append(
                    {
                        "conversations": [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": rejected},
                        ]
                    }
                )
                continue

            if score == -1 and len(comment) >= 3:
                dpo_out.append(
                    {
                        "prompt": [{"role": "user", "content": instruction}],
                        "chosen": comment,
                        "rejected": rejected,
                    }
                )
        except Exception:
            continue

    return dpo_out, sft_out


def _read_secretary_llm_cfg(repo_root: Path) -> tuple[str, str]:
    cfg_path = repo_root / "configs" / "secretary.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    llm_cfg = cfg.get("llm") if isinstance(cfg, dict) and isinstance(cfg.get("llm"), dict) else {}
    local_model = str((llm_cfg or {}).get("local_model") or "").strip() or "Qwen/Qwen3-8B"
    local_adapter = str((llm_cfg or {}).get("local_adapter") or "").strip()
    return local_model, local_adapter


def _bump_adapter_name(adapter_path: str) -> str:
    s = str(adapter_path or "").strip().rstrip("/\\")
    if not s:
        return "models/llm_secretary_ouroboros_v1"
    m = re.search(r"_v(\d+)$", s)
    if m:
        try:
            n = int(m.group(1)) + 1
            return re.sub(r"_v\d+$", f"_v{n}", s)
        except Exception:
            return s + "_v2"
    return s + "_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trajectory_dir = repo_root / "data" / "evolution" / "trajectories"
    out_dir = repo_root / "data" / "finetune" / "evolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dpo = out_dir / "dpo_nightly.jsonl"
    out_sft = out_dir / "sft_nightly.json"

    dpo_pairs, sft_rows = load_and_merge(trajectory_dir=trajectory_dir)

    if dpo_pairs:
        with out_dpo.open("w", encoding="utf-8") as f:
            for row in dpo_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if sft_rows:
        with out_sft.open("w", encoding="utf-8") as f:
            json.dump(sft_rows, f, ensure_ascii=False, indent=2)

    print(f"SFT: {len(sft_rows)} rows -> {out_sft}")
    print(f"DPO: {len(dpo_pairs)} rows -> {out_dpo}")

    local_model, current_adapter = _read_secretary_llm_cfg(repo_root)
    next_adapter = _bump_adapter_name(current_adapter)
    day = datetime.now().strftime("%Y%m%d")
    sft_outdir = str(next_adapter) + f"_sft_{day}"
    dpo_outdir = str(next_adapter) + f"_dpo_{day}"

    cmd_sft = (
        f"python scripts/finetune_llm.py --data {out_sft.as_posix()} "
        f"--model {local_model} --init-adapter {current_adapter} "
        f"--outdir {sft_outdir} --epochs 1 --batch-size 1 --grad-acc 8 --max-seq-len 1024 --qlora"
    )

    cmd_dpo = (
        f"python scripts/train_dpo.py --base-model {local_model} "
        f"--sft-adapter {sft_outdir}/lora_weights --data-path {out_dpo.as_posix()} "
        f"--output-dir {dpo_outdir} --epochs 1 --batch-size 1 --grad-accum 8 --reference-free"
    )

    if args.dry_run:
        print("\n[DRY-RUN] Would execute:")
        print(cmd_sft)
        print(cmd_dpo)

        try:
            print("\n[DRY-RUN] Alpha Loop:")
            subprocess.run(
                [
                    "python",
                    "scripts/generate_alpha_dataset.py",
                    "--dry-run",
                ],
                cwd=str(repo_root),
                check=False,
            )
        except Exception:
            pass
    return


if __name__ == "__main__":
    main()
