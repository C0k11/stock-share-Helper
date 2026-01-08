import argparse
import glob
import json
import re
import os
import sys
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
            msg = str(ctx_obj.get("message") or "").strip()
            # Include shared URLs in the instruction if present
            urls = ctx_obj.get("shared_urls")
            if isinstance(urls, list) and urls:
                url_text = "\n".join([f"Shared URL: {u}" for u in urls if isinstance(u, str)])
                if url_text:
                    msg += f"\n\n{url_text}"
            return msg
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
    
    # Try reading from trading config first (A2 architecture)
    trading_cfg = cfg.get("trading") if isinstance(cfg, dict) and isinstance(cfg.get("trading"), dict) else {}
    base_model = str((trading_cfg or {}).get("base_model") or "").strip()
    secretary_adapter = str((trading_cfg or {}).get("moe_secretary") or "").strip()
    
    if base_model and secretary_adapter:
        return base_model, secretary_adapter

    # Fallback to legacy llm config
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
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
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

    try:
        meta = {
            "time": datetime.now().isoformat(),
            "dry_run": bool(args.dry_run),
            "counts": {"sft_rows": int(len(sft_rows)), "dpo_rows": int(len(dpo_pairs))},
            "inputs": {"trajectory_dir": str(trajectory_dir)},
            "outputs": {
                "sft_nightly": str(out_sft),
                "dpo_nightly": str(out_dpo),
                "next_sft_adapter": str(sft_outdir),
                "next_dpo_adapter": str(dpo_outdir),
            },
            "commands": {"sft": str(cmd_sft), "dpo": str(cmd_dpo)},
        }
        (out_dir / "last_adapter.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

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

    # ===== Real run =====
    py = repo_root / "venv311" / "Scripts" / "python.exe"
    python_exe = str(py) if py.exists() else sys.executable
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    def _run(cmd: list[str], label: str) -> int:
        print(f"\n[RUN] {label}: {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, cwd=str(repo_root), check=False, env=env)
            return int(p.returncode)
        except KeyboardInterrupt:
            print("[RUN] Interrupted")
            return 130
        except Exception as e:
            print(f"[RUN] Failed: {e}")
            return 1

    # Generate alpha dataset (no training here; keeps it safe by default).
    try:
        _ = subprocess.run(
            [str(python_exe), "scripts/generate_alpha_dataset.py"],
            cwd=str(repo_root),
            check=False,
            env=env,
        )
    except Exception:
        pass

    if (not args.skip_sft) and sft_rows:
        sft_cmd = [
            str(python_exe),
            "scripts/finetune_llm.py",
            "--data",
            str(out_sft.as_posix()),
            "--model",
            str(local_model),
            "--outdir",
            str(sft_outdir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--grad-acc",
            "8",
            "--max-seq-len",
            "1024",
            "--qlora",
        ]
        if str(current_adapter or "").strip():
            sft_cmd.extend(["--init-adapter", str(current_adapter)])
        rc = _run(sft_cmd, "SFT")
        if rc != 0:
            raise SystemExit(rc)
    else:
        print("\n[SKIP] SFT: no rows or --skip-sft")

    if (not args.skip_dpo) and dpo_pairs:
        if not (Path(sft_outdir) / "lora_weights").exists():
            print(f"\n[WARN] SFT adapter missing: {sft_outdir}/lora_weights")
            print("[WARN] If you skipped SFT, pass --skip-dpo too, or point DPO to an existing SFT adapter.")
            raise SystemExit(2)
        dpo_cmd = [
            str(python_exe),
            "scripts/train_dpo.py",
            "--base-model",
            str(local_model),
            "--sft-adapter",
            str((Path(sft_outdir) / "lora_weights").as_posix()),
            "--data-path",
            str(out_dpo.as_posix()),
            "--output-dir",
            str(dpo_outdir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--grad-accum",
            "8",
            "--reference-free",
        ]
        rc = _run(dpo_cmd, "DPO")
        if rc != 0:
            raise SystemExit(rc)

        try:
            active = {
                "time": datetime.now().isoformat(),
                "base_model": str(local_model),
                "active_secretary_adapter": str(dpo_outdir),
                "source": "nightly_evolution",
            }
            (out_dir / "active_secretary_adapter.json").write_text(
                json.dumps(active, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    else:
        print("\n[SKIP] DPO: no pairs or --skip-dpo")

    print("\n[OK] Nightly evolution training finished.")
    print(f"[NEXT] Set configs/secretary.yaml trading.moe_secretary to: {dpo_outdir}")
    return


if __name__ == "__main__":
    main()
