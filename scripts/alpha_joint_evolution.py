import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if (line or "").strip():
                n += 1
    return n


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_repo_path(repo_root: Path, p: str) -> Path:
    pp = Path(str(p or "").strip())
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-model", required=True)
    ap.add_argument("--step-path", default="data/rl_experiences/joint_step_experiences.jsonl")
    ap.add_argument("--reward-thr", type=float, default=5e-5)
    ap.add_argument("--punish-thr", type=float, default=-5e-5)

    ap.add_argument("--scalper-adapter", default="")
    ap.add_argument("--analyst-adapter", default="")
    ap.add_argument("--news-adapter", default="")
    ap.add_argument("--system2-adapter", default="")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--reference-free", action="store_true")

    ap.add_argument("--out-root", default="")
    ap.add_argument("--skip-generate", action="store_true")

    ap.add_argument("--train-chartist", action="store_true")
    ap.add_argument("--chartist-model", default="")
    ap.add_argument("--chartist-prompt-yaml", default="configs/prompts/chartist_prompt.yaml")
    ap.add_argument("--chartist-outdir", default="")
    ap.add_argument("--chartist-min-confidence", type=float, default=0.0)
    ap.add_argument("--chartist-limit", type=int, default=0)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    evo_out_dir = repo_root / "data" / "finetune" / "evolution"
    evo_out_dir.mkdir(parents=True, exist_ok=True)

    vlm_out_dir = repo_root / "data" / "finetune" / "vlm"
    vlm_out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "scalper": evo_out_dir / "dpo_alpha_joint_scalper.jsonl",
        "analyst": evo_out_dir / "dpo_alpha_joint_analyst.jsonl",
        "news": evo_out_dir / "dpo_alpha_joint_news.jsonl",
        "system2": evo_out_dir / "dpo_alpha_joint_system2.jsonl",
    }

    if not bool(args.skip_generate):
        gen_cmd = [
            str(python_exe),
            str(repo_root / "scripts" / "generate_alpha_joint_datasets.py"),
            "--step-path",
            str(args.step_path),
            "--out-dir",
            str(evo_out_dir),
            "--reward-thr",
            str(float(args.reward_thr)),
            "--punish-thr",
            str(float(args.punish_thr)),
        ]
        subprocess.run(gen_cmd, cwd=str(repo_root), check=False)

    counts = {k: _count_jsonl(p) for k, p in datasets.items()}

    out_root_s = str(args.out_root or "").strip()
    if not out_root_s:
        out_root_s = str(repo_root / "models" / f"alpha_joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root = Path(out_root_s)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    adapters_in = {
        "scalper": str(args.scalper_adapter or "").strip(),
        "analyst": str(args.analyst_adapter or "").strip(),
        "news": str(args.news_adapter or "").strip(),
        "system2": str(args.system2_adapter or "").strip(),
    }

    adapters_out: Dict[str, str] = {}

    for expert, ds_path in datasets.items():
        if int(counts.get(expert, 0) or 0) <= 0:
            continue
        base_adapter = str(adapters_in.get(expert) or "").strip()
        if not base_adapter:
            continue

        out_dir = out_root / f"{expert}_dpo"
        cmd = [
            str(python_exe),
            str(repo_root / "scripts" / "train_dpo.py"),
            "--base-model",
            str(args.base_model),
            "--sft-adapter",
            str(base_adapter),
            "--data-path",
            str(ds_path.as_posix()),
            "--output-dir",
            str(out_dir),
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--grad-accum",
            str(int(args.grad_accum)),
        ]
        if bool(args.reference_free):
            cmd.append("--reference-free")

        r = subprocess.run(cmd, cwd=str(repo_root), check=False)
        if int(getattr(r, "returncode", 1) or 1) != 0:
            raise SystemExit(int(getattr(r, "returncode", 1) or 1))

        adapters_out[expert] = str(out_dir)

    chartist_out: str = ""
    chartist_rows = 0
    if bool(args.train_chartist):
        chartist_model = str(args.chartist_model or "").strip()
        if chartist_model:
            chart_sft = vlm_out_dir / "chartist_sft_from_joint_steps.jsonl"
            conv_cmd = [
                str(python_exe),
                str(repo_root / "scripts" / "convert_joint_steps_to_chartist_sft.py"),
                "--step-path",
                str(args.step_path),
                "--prompt-yaml",
                str(args.chartist_prompt_yaml),
                "--out",
                str(chart_sft.as_posix()),
                "--min-confidence",
                str(float(args.chartist_min_confidence)),
            ]
            if int(args.chartist_limit) > 0:
                conv_cmd += ["--limit", str(int(args.chartist_limit))]
            subprocess.run(conv_cmd, cwd=str(repo_root), check=False)

            chartist_rows = _count_jsonl(chart_sft)
            if int(chartist_rows) > 0:
                outdir_s = str(args.chartist_outdir or "").strip()
                if not outdir_s:
                    outdir = out_root / "chartist_vlm_lora"
                else:
                    outdir = _resolve_repo_path(repo_root, outdir_s)
                outdir.mkdir(parents=True, exist_ok=True)

                fin_cmd = [
                    str(python_exe),
                    str(repo_root / "scripts" / "finetune_vlm.py"),
                    "--data",
                    str(chart_sft.as_posix()),
                    "--model",
                    str(chartist_model),
                    "--outdir",
                    str(outdir.as_posix()),
                    "--epochs",
                    str(int(args.epochs)),
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--grad-accum",
                    str(int(args.grad_accum)),
                ]
                r2 = subprocess.run(fin_cmd, cwd=str(repo_root), check=False)
                if int(getattr(r2, "returncode", 1) or 1) != 0:
                    raise SystemExit(int(getattr(r2, "returncode", 1) or 1))

                chartist_out = str(outdir.as_posix())
                try:
                    p2 = repo_root / "data" / "finetune" / "evolution" / "active_chartist_adapter.json"
                    _write_json(
                        p2,
                        {
                            "time": datetime.now().isoformat(),
                            "source": "alpha_joint_evolution",
                            "active_chartist_adapter": str(chartist_out),
                            "chartist_model": str(chartist_model),
                            "rows": int(chartist_rows),
                        },
                    )
                except Exception:
                    pass

    payload = {
        "time": datetime.now().isoformat(),
        "source": "alpha_joint_evolution",
        "base_model": str(args.base_model),
        "active_moe_scalper": str(adapters_out.get("scalper") or adapters_in.get("scalper") or "").strip(),
        "active_moe_analyst": str(adapters_out.get("analyst") or adapters_in.get("analyst") or "").strip(),
        "active_moe_news": str(adapters_out.get("news") or adapters_in.get("news") or "").strip(),
        "active_moe_system2": str(adapters_out.get("system2") or adapters_in.get("system2") or "").strip(),
        "active_chartist_adapter": str(chartist_out).strip(),
        "chartist_rows": int(chartist_rows),
        "counts": dict(counts),
        "outputs": dict(adapters_out),
    }

    p = repo_root / "data" / "finetune" / "evolution" / "active_trading_models.json"
    _write_json(p, payload)

    print("[OK] alpha_joint_evolution finished")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
