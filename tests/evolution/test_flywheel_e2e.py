"""进化飞轮端到端契约测试（全 mock，零 GPU/零网络）。

链路：paper trading 决策+结果 -> PreferenceBuilder 记录 -> 偏好对数据集 ->
EvolutionTrainer 离线 DPO（注入假 runner）-> active adapter 指针 ->
LocalLLM 按指针热切换 adapter（生成后复位默认）。

各环节的单测在各自文件；本文件只保证**环节之间的数据契约**真的咬合。
"""

from __future__ import annotations

import json

from quantai.evolution import EvolutionTrainer, PreferenceBuilder
from quantai.llm.inference import LocalLLM


# --------------------------------------------------------------------------- #
# 最小假件（与 tests/llm/test_inference.py 的假件同构）
# --------------------------------------------------------------------------- #
class _FakeIds:
    @property
    def shape(self):
        return (1, 3)


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, text, **kwargs):
        return _FakeEncoding(input_ids=_FakeIds())

    def decode(self, ids, skip_special_tokens=True):
        return "BUY"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.adapter_calls = []

    def set_adapter(self, name):
        self.adapter_calls.append(name)

    def generate(self, **kwargs):
        return [[0, 1, 2, 10, 11]]


class _FakeRunner:
    """记录 train() 收到的 DPO 三列记录，并模拟落盘 adapter。"""

    def __init__(self, *, llm_cfg=None, output_dir="", **kw):
        self.output_dir = output_dir
        self.trained_records = None

    def train(self, records):
        self.trained_records = list(records)
        from pathlib import Path

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_flywheel_end_to_end(tmp_path):
    # ---- 1) 交易轨迹：同 ticker 两笔决策，一好一坏（可配对） ----------------
    pb = PreferenceBuilder(save_dir=str(tmp_path / "pref"), min_pnl_diff=50.0)
    ctx = {"ticker": "NVDA", "price": 500.0, "regime": "trend_up"}
    pb.log_decision("t1", ctx, "BUY", "breakout above 20d high", "scalper")
    pb.log_decision("t2", ctx, "SELL", "panic on noise", "scalper")
    pb.log_outcome("t1", pnl=+300.0, hold_bars=5, exit_reason="target")
    pb.log_outcome("t2", pnl=-200.0, hold_bars=2, exit_reason="stop")

    # ---- 2) 数据集：偏好对 JSONL（chosen=盈利方 / rejected=亏损方） ----------
    trainer = EvolutionTrainer(
        preferences=pb,
        adapters_dir=str(tmp_path / "adapters"),
        active_pointer=str(tmp_path / "active.json"),
    )
    ds = trainer.build_dataset(str(tmp_path / "pairs.jsonl"))
    pairs = [json.loads(x) for x in ds.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["decision"] == "BUY"
    assert pairs[0]["rejected"]["decision"] == "SELL"

    # ---- 3) 离线 DPO（假 runner）：训练收到合法三列 + 写出 adapter ----------
    captured = {}

    def factory(**kw):
        r = _FakeRunner(**kw)
        captured["runner"] = r
        return r

    result = trainer.train_offline(role="scalper", dataset_path=str(ds), runner_factory=factory)
    assert result["trained"] is True and result["records"] == 1
    rec = captured["runner"].trained_records[0]
    assert set(rec) == {"prompt", "chosen", "rejected"}
    assert "NVDA" in rec["prompt"] and rec["chosen"].startswith("BUY")

    # ---- 4) active 指针写入（热切换的"哪个 LoRA 生效"契约） -----------------
    assert trainer.get_active_adapters() == {"scalper": "scalper"}
    assert (tmp_path / "adapters" / "scalper" / "adapter_config.json").exists()

    # ---- 5) 热加载回大脑：LLM 按 active 指针切 adapter，生成后复位 ----------
    llm = LocalLLM(default_adapter="analyst")
    model = _FakeModel()
    llm.attach(model, _FakeTokenizer(), adapters={"analyst", *trainer.get_active_adapters()})
    role = next(iter(trainer.get_active_adapters()))
    out = llm.chat([{"role": "user", "content": "decide"}], adapter=role)
    assert out == "BUY"
    # 先切到进化出的 adapter，生成完复位到默认——热切换全程无重载
    assert model.adapter_calls == ["scalper", "analyst"]
