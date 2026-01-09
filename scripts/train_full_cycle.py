
import os
import sys
import subprocess
from pathlib import Path
import argparse
import datetime

import yaml

def find_latest_results_dir(base_dir):
    p = Path(base_dir)
    if not p.exists():
        return None
    # Filter for directories that look like run results (have metrics.json or similar, or just valid dirs)
    # Exclude basic folders if necessary.
    candidates = [d for d in p.iterdir() if d.is_dir() and (d / "golden_strict").exists()]
    if not candidates:
        return None
    # Sort by modification time
    candidates.sort(key=lambda x: x.stat().st_mtime)
    return candidates[-1]

def run_step(cmd, description):
    print(f"\n>>> [STEP] {description}...")
    print(f"    Command: {' '.join(cmd)}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"!!! [ERROR] {description} Failed with code {ret}")
        return False
    print(f">>> [SUCCESS] {description}")
    return True


def _write_secretary_adapter_to_config(*, root: Path, adapter_path: str) -> bool:
    try:
        cfg_path = root / "configs" / "secretary.yaml"
        if not cfg_path.exists():
            return False
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            return False
        trading_cfg = cfg.get("trading") if isinstance(cfg.get("trading"), dict) else {}
        if not isinstance(trading_cfg, dict):
            trading_cfg = {}
        trading_cfg["moe_secretary"] = str(adapter_path)
        cfg["trading"] = trading_cfg
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return True
    except Exception:
        return False

def main():
    root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-secretary", action="store_true", default=False)
    parser.add_argument("--bootstrap-base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--bootstrap-load-4bit", action="store_true", default=True)
    parser.add_argument("--bootstrap-no-4bit", dest="bootstrap_load_4bit", action="store_false")
    parser.add_argument("--bootstrap-epochs", type=int, default=1)
    parser.add_argument("--secretary-triple-train", action="store_true", default=False)
    parser.add_argument("--secretary-base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--secretary-epochs", type=int, default=1)
    parser.add_argument("--secretary-qlora", action="store_true", default=True)
    parser.add_argument("--secretary-no-qlora", dest="secretary_qlora", action="store_false")
    args = parser.parse_args()
    
    print(f"=== Starting Full Training Cycle (Tabular Planner + LLM Agent) ===")
    print(f"Root: {root}")
    
    # 1. Tabular Planner Data & SFT
    results_dir = find_latest_results_dir(root / "results")
    if results_dir:
        print(f"Detected latest results: {results_dir.name}")
        # Build Dataset
        cmd_build = [
            python_exe, 
            str(root / "scripts/training/build_planner_dataset.py"),
            "--run-dir", str(results_dir),
            "--system", "golden_strict",
            "--out", str(root / "data/training/planner_dataset_v1.csv")
        ]
        if run_step(cmd_build, "Build Tabular Planner Dataset"):
            # Train SFT
            cmd_train_tabular = [
                python_exe,
                str(root / "scripts/training/train_planner_sft.py"),
                "--data", str(root / "data/training/planner_dataset_v1.csv"),
                "--out", str(root / "models/planner_sft_v1.pt"),
                "--epochs", "200",
                "--patience", "20"
            ]
            run_step(cmd_train_tabular, "Train Tabular Planner SFT")
    else:
        print("!!! [WARN] No valid results directory found for Tabular Planner training. Skipping.")

    # 2. LLM Agent Evolution (SFT + DPO)
    # This uses data/evolution/trajectories/
    cmd_evolution = [
        python_exe,
        str(root / "scripts/nightly_evolution.py")
    ]
    run_step(cmd_evolution, "Nightly Evolution (LLM SFT + DPO)")

    # 3. Secretary triple training (base -> dispatch SFT -> dispatch DPO)
    if bool(getattr(args, "secretary_triple_train", False)):
        base_data = root / "data" / "finetune" / "teacher_secretary" / "train_secretary_v1.json"
        dispatch_sft_data = root / "data" / "finetune" / "teacher_secretary" / "train_secretary_dispatch_v3.json"
        dispatch_sft_eval = root / "data" / "finetune" / "teacher_secretary" / "val_secretary_dispatch_v3.json"
        dpo_data = root / "data" / "dpo" / "secretary_dispatch_pairs_v3.jsonl"

        if (not base_data.exists()) or (not dispatch_sft_data.exists()) or (not dpo_data.exists()):
            print(
                f"!!! [WARN] secretary triple datasets missing: base={base_data.exists()} dispatch_sft={dispatch_sft_data.exists()} dpo={dpo_data.exists()}"
            )
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_base = root / "models" / f"llm_secretary_triple_base_{stamp}"
            out_sft = root / "models" / f"llm_secretary_triple_sft_{stamp}"
            out_dpo = root / "models" / f"llm_secretary_triple_dpo_{stamp}"

            base_model = str(getattr(args, "secretary_base_model", "Qwen/Qwen2.5-7B-Instruct"))
            epochs = int(getattr(args, "secretary_epochs", 1))
            use_qlora = bool(getattr(args, "secretary_qlora", True))

            cmd_base = [
                python_exe,
                str(root / "scripts" / "finetune_llm.py"),
                "--data",
                str(base_data).replace("\\", "/"),
                "--model",
                base_model,
                "--outdir",
                str(out_base).replace("\\", "/"),
                "--epochs",
                str(epochs),
                "--batch-size",
                "1",
                "--grad-acc",
                "8",
                "--max-seq-len",
                "1024",
            ]
            if use_qlora:
                cmd_base.append("--qlora")

            if run_step(cmd_base, "Secretary Triple: Base (Teacher/Distill) SFT"):
                cmd_sft = [
                    python_exe,
                    str(root / "scripts" / "finetune_llm.py"),
                    "--data",
                    str(dispatch_sft_data).replace("\\", "/"),
                    "--eval-data",
                    str(dispatch_sft_eval).replace("\\", "/"),
                    "--model",
                    base_model,
                    "--init-adapter",
                    str(out_base / "lora_weights").replace("\\", "/"),
                    "--outdir",
                    str(out_sft).replace("\\", "/"),
                    "--epochs",
                    str(epochs),
                    "--batch-size",
                    "1",
                    "--grad-acc",
                    "8",
                    "--max-seq-len",
                    "1024",
                ]
                if use_qlora:
                    cmd_sft.append("--qlora")

                if run_step(cmd_sft, "Secretary Triple: Dispatch SFT"):
                    cmd_dpo = [
                        python_exe,
                        str(root / "scripts" / "train_dpo.py"),
                        "--base-model",
                        base_model,
                        "--sft-adapter",
                        str(out_sft / "lora_weights").replace("\\", "/"),
                        "--data-path",
                        str(dpo_data).replace("\\", "/"),
                        "--output-dir",
                        str(out_dpo).replace("\\", "/"),
                        "--epochs",
                        str(epochs),
                        "--batch-size",
                        "1",
                        "--grad-accum",
                        "8",
                        "--lr",
                        "5e-6",
                        "--beta",
                        "0.1",
                        "--reference-free",
                    ]
                    if run_step(cmd_dpo, "Secretary Triple: Dispatch DPO"):
                        ok = _write_secretary_adapter_to_config(root=root, adapter_path=str(out_dpo))
                        print(f">>> [CONFIG] trading.moe_secretary updated={ok} -> {out_dpo}")

    # 3. Secretary mask bootstrap (use legacy high-quality SFT/DPO data)
    if bool(getattr(args, "bootstrap_secretary", False)):
        sft_data = root / "data" / "finetune" / "teacher_secretary" / "teacher_secretary_dispatch_v3.jsonl"
        dpo_data = root / "data" / "dpo" / "secretary_dispatch_pairs_v3.jsonl"
        if (not sft_data.exists()) or (not dpo_data.exists()):
            print(f"!!! [WARN] bootstrap datasets missing: sft={sft_data.exists()} dpo={dpo_data.exists()}")
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_sft = root / "models" / f"llm_secretary_bootstrap_sft_{stamp}"
            out_dpo = root / "models" / f"llm_secretary_bootstrap_dpo_{stamp}"

            cmd_sft = [
                python_exe,
                str(root / "scripts" / "finetune_llm.py"),
                "--data", str(sft_data).replace("\\", "/"),
                "--model", str(getattr(args, "bootstrap_base_model", "Qwen/Qwen2.5-7B-Instruct")),
                "--outdir", str(out_sft).replace("\\", "/"),
                "--epochs", str(int(getattr(args, "bootstrap_epochs", 1))),
                "--batch-size", "1",
                "--grad-acc", "8",
                "--max-seq-len", "1024",
            ]
            if bool(getattr(args, "bootstrap_load_4bit", True)):
                cmd_sft.append("--qlora")

            if run_step(cmd_sft, "Bootstrap Secretary SFT"):
                sft_adapter = out_sft / "lora_weights"
                cmd_dpo = [
                    python_exe,
                    str(root / "scripts" / "train_dpo.py"),
                    "--base-model", str(getattr(args, "bootstrap_base_model", "Qwen/Qwen2.5-7B-Instruct")),
                    "--sft-adapter", str(sft_adapter).replace("\\", "/"),
                    "--data-path", str(dpo_data).replace("\\", "/"),
                    "--output-dir", str(out_dpo).replace("\\", "/"),
                    "--epochs", str(int(getattr(args, "bootstrap_epochs", 1))),
                    "--batch-size", "1",
                    "--grad-accum", "8",
                    "--reference-free",
                ]
                if run_step(cmd_dpo, "Bootstrap Secretary DPO"):
                    ok = _write_secretary_adapter_to_config(root=root, adapter_path=str(out_dpo))
                    print(f">>> [CONFIG] trading.moe_secretary updated={ok} -> {out_dpo}")
    
    print("\n=== Full Training Cycle Complete ===")

if __name__ == "__main__":
    main()
