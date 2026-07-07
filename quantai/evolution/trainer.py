"""离线训练编排 —— 数据飞轮的"训练 + 热切换"端（取代 nightly_evolution 的编排核心）。

把采集到的偏好对喂**离线 DPO**（`quantai.llm.DPORunner`），训练出新 adapter，再写"active 指针"
（哪个 LoRA 当前生效），供 `llm` 推理层热切换。**诚实边界**：这里只编排离线训练；没有任何在线
实时梯度（见 `collector.online_gradient_step` 的占位说明，B-1）。

可测性：
- `to_dpo_records`（偏好对 -> DPO 三列 prompt/chosen/rejected）是**纯逻辑**，全单测。
- active 指针读写是**纯文件 IO**，可单测。
- `train_offline` 的重活（trl DPO）**懒导入**，并支持注入 `runner_factory` 以便不依赖 GPU 单测编排。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from quantai.config.schema import AppConfig, EvolutionConfig, LLMConfig
from quantai.evolution.dataset_builder import PreferenceBuilder

RunnerFactory = Callable[..., Any]


def _format_prompt(context: Dict[str, Any]) -> str:
    """把决策上下文拼成 DPO 的 prompt 文本。"""
    ctx = context or {}
    ticker = ctx.get("ticker", "UNKNOWN")
    price = ctx.get("price")
    regime = ctx.get("regime", "")
    parts = [f"Ticker: {ticker}"]
    if price is not None:
        parts.append(f"Price: {price}")
    if regime:
        parts.append(f"Regime: {regime}")
    return "Given the market context, decide BUY/SELL/HOLD and justify.\n" + " | ".join(parts)


def _format_side(side: Dict[str, Any]) -> str:
    """把 chosen/rejected 一侧拼成回答文本。"""
    s = side or {}
    decision = str(s.get("decision") or "").strip()
    reasoning = str(s.get("reasoning") or "").strip()
    if decision and reasoning:
        return f"{decision}: {reasoning}"
    return decision or reasoning


class EvolutionTrainer:
    """编排：偏好对 -> DPO 训练 -> 写 active adapter 指针。"""

    def __init__(
        self,
        *,
        preferences: PreferenceBuilder,
        adapters_dir: str = "data/evolution/adapters",
        active_pointer: str = "data/evolution/active_adapters.json",
        llm_cfg: Optional[LLMConfig] = None,
    ) -> None:
        self.preferences = preferences
        self.adapters_dir = Path(adapters_dir)
        self.active_pointer = Path(active_pointer)
        self.llm_cfg = llm_cfg

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "EvolutionTrainer":
        return cls(
            preferences=PreferenceBuilder.from_config(cfg.evolution),
            adapters_dir=cfg.evolution.adapters_dir,
            active_pointer=cfg.evolution.active_pointer,
            llm_cfg=cfg.llm,
        )

    # ----------------------------------------------------------------- #
    # 数据集
    # ----------------------------------------------------------------- #
    def build_dataset(self, out_path: Optional[str] = None) -> Path:
        """生成并落盘偏好对 JSONL，返回路径。"""
        return self.preferences.export_pairs(out_path)

    @staticmethod
    def to_dpo_records(pairs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """偏好对 -> DPO 三列 [{prompt, chosen, rejected}]（丢弃任一侧为空的）。"""
        records: List[Dict[str, str]] = []
        for p in pairs or []:
            prompt = _format_prompt(p.get("context") or {})
            chosen = _format_side(p.get("chosen") or {})
            rejected = _format_side(p.get("rejected") or {})
            if chosen and rejected and chosen != rejected:
                records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return records

    # ----------------------------------------------------------------- #
    # active adapter 指针（热切换用）
    # ----------------------------------------------------------------- #
    def get_active_adapters(self) -> Dict[str, str]:
        if not self.active_pointer.exists():
            return {}
        try:
            data = json.loads(self.active_pointer.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def set_active_adapter(self, role: str, name: str) -> Dict[str, str]:
        """把某角色（如 'scalper'）的生效 adapter 指向 `name`，写回指针文件。"""
        active = self.get_active_adapters()
        active[str(role)] = str(name)
        self.active_pointer.parent.mkdir(parents=True, exist_ok=True)
        self.active_pointer.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")
        return active

    # ----------------------------------------------------------------- #
    # 离线训练（重活懒导入 / 可注入）
    # ----------------------------------------------------------------- #
    def train_offline(
        self,
        *,
        role: str = "scalper",
        dataset_path: Optional[str] = None,
        runner_factory: Optional[RunnerFactory] = None,
        set_active: bool = True,
        **runner_kwargs: Any,
    ) -> Dict[str, Any]:
        """跑一轮离线 DPO，产出新 adapter；可选把它设为该角色的 active。

        `runner_factory` 可注入（测试用假 runner）；默认懒导入 `quantai.llm.DPORunner`（需 GPU/trl）。
        返回 {records, output_dir, active}。无偏好对则跳过训练。
        """
        ds_path = Path(dataset_path) if dataset_path else self.build_dataset()
        pairs = _read_jsonl(ds_path)
        records = self.to_dpo_records(pairs)

        output_dir = self.adapters_dir / role
        if not records:
            return {"records": 0, "output_dir": str(output_dir), "active": self.get_active_adapters(), "trained": False}

        if runner_factory is None:
            from quantai.llm import DPORunner  # 懒导入：trl/torch 仅在真训练时才需要

            if self.llm_cfg is None:
                raise ValueError("默认真实 DPO 路径需要 llm_cfg（用 from_config 构造，或注入 runner_factory）")
            sft_adapter = str(runner_kwargs.pop("sft_adapter", "") or "")
            if not sft_adapter:
                raise ValueError(
                    "真实 DPO 按设计在 SFT adapter 之上继续对齐：train_offline(sft_adapter=<路径>)"
                )

            class _RealRunner:
                """把 DPORunner 适配到编排契约：train(records 列表) → 落 JSONL → 按路径训练。

                （审查实锤：旧默认路径 `DPORunner(llm_cfg=...)` 直接 TypeError、
                `train(records)` 传列表给要文件路径的接口——真实路径从没通过。）
                """

                def __init__(self, *, llm_cfg: Any, output_dir: str, **_: Any) -> None:
                    self._out = Path(output_dir)
                    self._runner = DPORunner(
                        model_name=llm_cfg.model_name,
                        sft_adapter=sft_adapter,
                        output_dir=output_dir,
                        cfg=llm_cfg.dpo,
                        cache_dir=llm_cfg.cache_dir,
                    )

                def train(self, recs: List[Dict[str, str]]) -> str:
                    from quantai.distill.generator import write_jsonl

                    self._out.mkdir(parents=True, exist_ok=True)
                    data_path = self._out / "dpo_records.jsonl"
                    write_jsonl(data_path, recs)
                    return self._runner.train(str(data_path))

            runner_factory = _RealRunner

        runner = runner_factory(llm_cfg=self.llm_cfg, output_dir=str(output_dir), **runner_kwargs)
        runner.train(records)

        active = self.set_active_adapter(role, role) if set_active else self.get_active_adapters()
        return {"records": len(records), "output_dir": str(output_dir), "active": active, "trained": True}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
