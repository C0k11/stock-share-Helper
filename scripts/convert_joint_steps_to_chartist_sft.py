import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            obj = _safe_json_loads(s)
            if obj is not None:
                yield obj


def _read_yaml_prompts(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        import yaml

        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {}
        sp = str(obj.get("system_prompt") or "").strip()
        up = str(obj.get("user_prompt") or "").strip()
        if not sp or not up:
            return {}
        return {"system_prompt": sp, "user_prompt": up}
    except Exception:
        return {}


def _resolve_repo_path(repo_root: Path, p: str) -> Path:
    pp = Path(str(p or "").strip())
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-path", default="data/rl_experiences/joint_step_experiences.jsonl")
    ap.add_argument("--prompt-yaml", default="configs/prompts/chartist_prompt.yaml")
    ap.add_argument("--out", default="data/finetune/vlm/chartist_sft_from_joint_steps.jsonl")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    step_path = _resolve_repo_path(repo_root, str(args.step_path))
    prompt_yaml = _resolve_repo_path(repo_root, str(args.prompt_yaml))
    out_path = _resolve_repo_path(repo_root, str(args.out))

    prompts = _read_yaml_prompts(prompt_yaml)
    sp = str(prompts.get("system_prompt") or "").strip()
    up_tmpl = str(prompts.get("user_prompt") or "").strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for rec in _iter_jsonl(step_path):
        md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        tr = md.get("decision_trace") if isinstance(md.get("decision_trace"), dict) else {}
        chart = tr.get("chartist") if isinstance(tr.get("chartist"), dict) else {}

        b64 = str(chart.get("image_base64") or "").strip()
        if not b64:
            continue

        sig = str(chart.get("signal") or "").strip().upper()
        conf = None
        try:
            conf = float(chart.get("confidence"))
        except Exception:
            conf = None

        if conf is not None and float(conf) < float(args.min_confidence):
            continue

        ticker = str(chart.get("ticker") or tr.get("ticker") or "").strip().upper()
        asof = str(tr.get("time") or md.get("time") or "").strip()
        if not asof:
            asof = datetime.now().isoformat()

        reasoning = str(chart.get("reasoning") or "").strip()
        if not reasoning:
            reasoning = str(chart.get("analysis") or chart.get("reason") or "").strip()

        if up_tmpl:
            user_prompt = up_tmpl.format(ticker=ticker or "", asof=asof)
        else:
            user_prompt = f"Analyze this chart for ticker={ticker} asof={asof}. Return only the JSON object."

        if not sp:
            sp2 = "You are a veteran Technical Analyst (CMT). Analyze the provided candlestick chart image."
        else:
            sp2 = sp

        ans = json.dumps(
            {
                "signal": sig if sig else "NEUTRAL",
                "confidence": float(conf) if conf is not None else 0.0,
                "reasoning": reasoning,
            },
            ensure_ascii=False,
        )

        rows.append(
            {
                "system_prompt": sp2,
                "user_prompt": str(user_prompt),
                "assistant": ans,
                "image_base64": b64,
            }
        )

        if int(args.limit) > 0 and len(rows) >= int(args.limit):
            break

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out_path.as_posix()), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
