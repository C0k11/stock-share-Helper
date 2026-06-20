"""quantai.agents.overlays.vlm_chartist 测试（B-3）。

CI 友好：渲染用真 mplfinance，但 VLM 推理用**注入的假 model/processor**（不下载/不上 GPU）。
真 Qwen2.5-VL 加载推理走 env-gated 冒烟（QUANTAI_VLM_TEST=1）。
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os

import pytest

from quantai.agents.overlays.chartist import ChartistOverlay
from quantai.agents.overlays.vlm_chartist import QwenVLChartist, render_candles
from quantai.config import AppConfig

_HAS_MPL = (
    importlib.util.find_spec("mplfinance") is not None
    and importlib.util.find_spec("PIL") is not None
)


def _bars(n=30):
    base = dt.datetime(2024, 1, 1)
    out = []
    px = 100.0
    for i in range(n):
        c = px + (0.5 if i % 2 == 0 else -0.3)
        out.append(
            {
                "time": (base + dt.timedelta(days=i)).isoformat(),
                "open": px,
                "high": max(px, c) + 1.0,
                "low": min(px, c) - 1.0,
                "close": c,
                "volume": 1000 + i,
            }
        )
        px = c
    return out


class FakeProcessor:
    """无 apply_chat_template -> 走 else 分支，避免依赖 qwen_vl_utils。"""

    def __init__(self, decoded):
        self.decoded = decoded
        self.calls = []

    def __call__(self, text=None, images=None, **kw):
        self.calls.append({"text": text, "images": images})
        return {"input_ids": [[1, 2, 3]]}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [self.decoded]


class FakeModel:
    device = None

    def __init__(self):
        self.gen_kwargs = None

    def generate(self, **kw):
        self.gen_kwargs = kw
        return [[1, 2, 3, 4]]


# --- parse（纯逻辑） --- #
def test_parse_valid():
    c = QwenVLChartist(vlm_model="")
    out = c.parse('{"signal": "bullish", "confidence": 0.8, "reasoning": "higher highs"}')
    assert out == {"signal": "BULLISH", "confidence": 0.8, "reasoning": "higher highs"}


def test_parse_garbage_none():
    assert QwenVLChartist(vlm_model="").parse("no json at all") is None


def test_parse_missing_confidence_defaults_zero():
    out = QwenVLChartist(vlm_model="").parse('{"signal": "BEARISH"}')
    assert out["signal"] == "BEARISH"
    assert out["confidence"] == 0.0


def test_parse_alt_reason_keys():
    out = QwenVLChartist(vlm_model="").parse('{"signal":"NEUTRAL","analysis":"sideways"}')
    assert out["reasoning"] == "sideways"


# --- build_messages（纯逻辑） --- #
def test_build_messages_structure():
    c = QwenVLChartist(vlm_model="")
    msgs = c.build_messages(image="IMG", ticker="nvda", asof="2024-01-02")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    content = msgs[1]["content"]
    assert content[0] == {"type": "image", "image": "IMG"}
    assert "NVDA" in content[1]["text"]
    assert "2024-01-02" in content[1]["text"]


# --- 加载状态 / attach --- #
def test_not_loaded_by_default():
    assert QwenVLChartist(vlm_model="").is_loaded is False


def test_attach_sets_loaded():
    c = QwenVLChartist(vlm_model="").attach(FakeModel(), FakeProcessor("{}"))
    assert c.is_loaded is True


def test_load_missing_model_name_is_safe():
    c = QwenVLChartist(vlm_model="")
    c.load()  # 不应抛、不应下载
    assert c.is_loaded is False
    assert "missing" in c._error


def test_analyze_not_loaded_returns_none():
    # vlm_model="" -> load() 提前返回 -> 不触发真实下载
    assert QwenVLChartist(vlm_model="").analyze("NVDA") is None


def test_analyze_no_bars_provider_returns_none():
    c = QwenVLChartist(vlm_model="").attach(FakeModel(), FakeProcessor("{}"))
    assert c.analyze("NVDA") is None  # render -> None（无 bars_provider）


# --- 渲染（真 mplfinance） --- #
@pytest.mark.skipif(not _HAS_MPL, reason="mplfinance/PIL 未安装")
def test_render_candles_produces_image():
    img = render_candles(_bars(30), lookback=60)
    assert img is not None
    assert img.size[0] > 0 and img.size[1] > 0


@pytest.mark.skipif(not _HAS_MPL, reason="mplfinance/PIL 未安装")
def test_render_insufficient_bars_none():
    assert render_candles(_bars(3)) is None


@pytest.mark.skipif(not _HAS_MPL, reason="mplfinance/PIL 未安装")
def test_render_bad_rows_none():
    assert render_candles([{"foo": 1}] * 10) is None


# --- 推理（假 model/processor + 真渲染） --- #
@pytest.mark.skipif(not _HAS_MPL, reason="mplfinance/PIL 未安装")
def test_infer_from_image_with_fakes():
    img = render_candles(_bars(30))
    proc = FakeProcessor('{"signal": "BULLISH", "confidence": 0.9, "reasoning": "up"}')
    c = QwenVLChartist(vlm_model="", temperature=0.0).attach(FakeModel(), proc)
    out = c.infer_from_image(img, "NVDA", "2024-01-02")
    assert "BULLISH" in out
    assert proc.calls  # processor 被调用


@pytest.mark.skipif(not _HAS_MPL, reason="mplfinance/PIL 未安装")
def test_analyze_end_to_end_with_fakes():
    proc = FakeProcessor('{"signal": "BEARISH", "confidence": 0.75, "reasoning": "lower lows"}')
    c = QwenVLChartist(vlm_model="", bars_provider=lambda t: _bars(40)).attach(FakeModel(), proc)
    out = c.analyze("NVDA", asof="2024-02-01")
    assert out == {"signal": "BEARISH", "confidence": 0.75, "reasoning": "lower lows"}


@pytest.mark.skipif(not _HAS_MPL, reason="mplfinance/PIL 未安装")
def test_overlay_assess_with_real_analyzer():
    # ChartistOverlay + QwenVLChartist(假后端) 端到端打分
    proc = FakeProcessor('{"signal": "BULLISH", "confidence": 0.95, "reasoning": "x"}')
    analyzer = QwenVLChartist(vlm_model="", bars_provider=lambda t: _bars(40)).attach(
        FakeModel(), proc
    )
    overlay = ChartistOverlay(enabled=True, confidence_threshold=0.7)
    assert overlay.assess("NVDA", "BUY", analyzer=analyzer) == 1
    assert overlay.assess("NVDA", "SELL", analyzer=analyzer) == -1


# --- from_config --- #
def test_from_config():
    cfg = AppConfig()
    c = QwenVLChartist.from_config(cfg.agents.chartist)
    assert c.vlm_model == cfg.agents.chartist.vlm_model
    assert c.lookback == cfg.agents.chartist.lookback
    assert c.load_4bit == cfg.agents.chartist.load_4bit


# --- env-gated 真 VLM 冒烟 --- #
@pytest.mark.skipif(
    os.getenv("QUANTAI_VLM_TEST") != "1",
    reason="设 QUANTAI_VLM_TEST=1 运行真 Qwen2.5-VL 冒烟（会下载/加载 VLM，需 GPU）",
)
def test_vlm_chartist_gpu_smoke():
    cfg = AppConfig().agents.chartist
    c = QwenVLChartist(
        vlm_model=cfg.vlm_model,
        load_4bit=cfg.load_4bit,
        max_new_tokens=64,
        temperature=0.0,
        bars_provider=lambda t: _bars(60),
    )
    out = c.analyze("NVDA", asof="2024-01-02")
    assert out is None or (isinstance(out, dict) and "signal" in out)
