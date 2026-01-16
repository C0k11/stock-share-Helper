import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _safe_json_loads(line: str) -> Dict[str, Any] | None:
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


def _clip_json(obj: Any, *, max_chars: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) <= int(max_chars):
        return s
    return s[: max(0, int(max_chars) - 3)] + "..."


def _format_joint_context(rec: Dict[str, Any]) -> str:
    st = rec.get("state") if isinstance(rec.get("state"), dict) else {}
    nx = rec.get("next_state") if isinstance(rec.get("next_state"), dict) else {}
    md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}

    ticker = str(st.get("ticker") or md.get("ticker") or "").strip().upper()
    action = str(rec.get("action") or "").strip().upper()

    trace = md.get("decision_trace") if isinstance(md.get("decision_trace"), dict) else {}
    next_trace = md.get("next_trace") if isinstance(md.get("next_trace"), dict) else {}

    lines = [
        "ALPHA_JOINT_STEP_CONTEXT",
        f"ticker={ticker}",
        f"action={action}",
        f"reward={rec.get('reward')}",
        "STATE=" + _clip_json(st, max_chars=1800),
        "TRACE=" + _clip_json(trace, max_chars=2600),
        "NEXT_STATE=" + _clip_json(nx, max_chars=1800),
        "NEXT_TRACE=" + _clip_json(next_trace, max_chars=1200),
    ]
    return "\n".join(lines).strip()


def _mk_pair(*, ctx: str, chosen: str, rejected: str) -> Dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": ctx}],
        "chosen": str(chosen),
        "rejected": str(rejected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-path", default="data/rl_experiences/joint_step_experiences.jsonl")
    ap.add_argument("--out-dir", default="data/finetune/evolution")
    ap.add_argument("--reward-thr", type=float, default=5e-5)
    ap.add_argument("--punish-thr", type=float, default=-5e-5)
    ap.add_argument("--max-context-chars", type=int, default=6000)

    args = ap.parse_args()

    step_path = Path(str(args.step_path))
    if not step_path.is_absolute():
        step_path = (Path(__file__).resolve().parents[1] / step_path).resolve()

    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parents[1] / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outs: Dict[str, Path] = {
        "scalper": out_dir / "dpo_alpha_joint_scalper.jsonl",
        "analyst": out_dir / "dpo_alpha_joint_analyst.jsonl",
        "news": out_dir / "dpo_alpha_joint_news.jsonl",
        "system2": out_dir / "dpo_alpha_joint_system2.jsonl",
    }

    rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in outs.keys()}

    for rec in _iter_jsonl(step_path):
        md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        trace = md.get("decision_trace") if isinstance(md.get("decision_trace"), dict) else {}

        action = str(rec.get("action") or "HOLD").strip().upper() or "HOLD"
        attempted = ""
        try:
            attempted = str((trace.get("proposed_action") if isinstance(trace, dict) else "") or "").strip().upper()
        except Exception:
            attempted = ""

        r = None
        try:
            r = float(rec.get("reward"))
        except Exception:
            r = None
        if r is None:
            continue

        ctx = _format_joint_context(rec)
        if int(args.max_context_chars) > 0 and len(ctx) > int(args.max_context_chars):
            ctx = ctx[: int(args.max_context_chars)]

        selected = ""
        try:
            selected = str(trace.get("selected_expert") or trace.get("expert") or "").strip().lower()
        except Exception:
            selected = ""
        try:
            router = trace.get("router") if isinstance(trace.get("router"), dict) else {}
            if not selected:
                selected = str(router.get("expert") or "").strip().lower()
        except Exception:
            pass

        risk = trace.get("risk") if isinstance(trace.get("risk"), dict) else {}
        blocked = False
        try:
            blocked = bool(risk.get("blocked"))
        except Exception:
            blocked = False

        if blocked and attempted in {"BUY", "SELL", "CLEAR"}:
            for k in ("scalper", "analyst", "news"):
                if selected != k:
                    continue
                rows[k].append(
                    _mk_pair(
                        ctx=ctx,
                        chosen="Action: HOLD\nReason: Proposed action was blocked by risk controls.",
                        rejected=f"Action: {attempted}\nReason: Blocked by risk controls.",
                    )
                )
            rows["system2"].append(
                _mk_pair(
                    ctx=ctx,
                    chosen=json.dumps({"final_decision": "HOLD", "rationale": "Blocked by risk controls."}, ensure_ascii=False),
                    rejected=json.dumps({"final_decision": attempted, "rationale": "Blocked by risk controls."}, ensure_ascii=False),
                )
            )
            continue

        if r >= float(args.reward_thr):
            for k in ("scalper", "analyst", "news"):
                if selected != k:
                    continue
                rows[k].append(
                    _mk_pair(
                        ctx=ctx,
                        chosen=f"Action: {action}\nReason: Positive step reward.",
                        rejected="Action: HOLD\nReason: Lower expected return.",
                    )
                )
            rows["system2"].append(
                _mk_pair(
                    ctx=ctx,
                    chosen=json.dumps({"final_decision": action, "rationale": "Positive step reward."}, ensure_ascii=False),
                    rejected=json.dumps({"final_decision": "HOLD", "rationale": "Lower expected return."}, ensure_ascii=False),
                )
            )
            continue

        if r <= float(args.punish_thr):
            if action != "HOLD":
                for k in ("scalper", "analyst", "news"):
                    if selected != k:
                        continue
                    rows[k].append(
                        _mk_pair(
                            ctx=ctx,
                            chosen="Action: HOLD\nReason: Negative step reward / risk too high.",
                            rejected=f"Action: {action}\nReason: Led to negative step reward.",
                        )
                    )
            rows["system2"].append(
                _mk_pair(
                    ctx=ctx,
                    chosen=json.dumps({"final_decision": "HOLD", "rationale": "Negative step reward / risk too high."}, ensure_ascii=False),
                    rejected=json.dumps({"final_decision": action, "rationale": "Led to negative step reward."}, ensure_ascii=False),
                )
            )
            continue

    for k, path in outs.items():
        rr = rows.get(k) or []
        if rr:
            with path.open("w", encoding="utf-8") as f:
                for row in rr:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for k, path in outs.items():
        n = len(rows.get(k) or [])
        print(f"Alpha-Joint-DPO: {k} rows={n} -> {path}")


if __name__ == "__main__":
    main()
