"""quantai.llm.inference 编排测试（假 model/tokenizer，不下真模型）。

覆盖：解码切片、max_time 回退、adapter 热切换+复位、no_think 注入、上下文截断、
think 清洗、from_config 映射、未加载返回空。
"""

from __future__ import annotations

import pytest

from quantai.config import AppConfig
from quantai.llm import inference
from quantai.llm.inference import LocalLLM, _safe_generate, decode_new_tokens


# --------------------------------------------------------------------------- #
# 假对象
# --------------------------------------------------------------------------- #
class _FakeIds:
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def shape(self):
        return (1, self._n)


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self, input_len: int = 3, decode_value: str = "DECODED") -> None:
        self.input_len = input_len
        self.decode_value = decode_value
        self.eos_token_id = 0
        self.seen_messages = None
        self.seen_tk_kwargs = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen_messages = messages
        return "PROMPT"

    def __call__(self, text, **kwargs):
        self.seen_tk_kwargs = kwargs
        return _FakeEncoding(input_ids=_FakeIds(self.input_len))

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_ids = list(ids)
        return self.decode_value


class FakeModel:
    def __init__(self, output_ids=None, device="cpu", raise_max_time=False) -> None:
        self.output_ids = output_ids if output_ids is not None else [[0, 1, 2, 10, 11]]
        self.device = device
        self.adapter_calls = []
        self.generate_calls = []
        self._raise_max_time = raise_max_time

    def set_adapter(self, name):
        self.adapter_calls.append(name)

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self._raise_max_time and "max_time" in kwargs:
            raise TypeError("generate() got an unexpected keyword argument 'max_time'")
        return self.output_ids


# --------------------------------------------------------------------------- #
# 模块级纯函数
# --------------------------------------------------------------------------- #
def test_decode_new_tokens_slices_prompt() -> None:
    tok = FakeTokenizer(decode_value="hi")
    out = decode_new_tokens(tok, [[1, 2, 3, 4, 5]], input_len=2)
    assert tok.decoded_ids == [3, 4, 5]  # 只解码新 token
    assert out == "hi"


def test_safe_generate_happy() -> None:
    model = FakeModel(output_ids="OK")
    assert _safe_generate(model, _FakeEncoding(), {"max_new_tokens": 8}) == "OK"


def test_safe_generate_max_time_fallback() -> None:
    model = FakeModel(raise_max_time=True)
    out = _safe_generate(model, _FakeEncoding(input_ids=_FakeIds(1)), {"max_time": 5.0, "max_new_tokens": 8})
    assert out == model.output_ids
    assert len(model.generate_calls) == 2  # 重试一次
    assert "max_time" not in model.generate_calls[1]  # 重试去掉 max_time


def test_safe_generate_other_typeerror_reraises() -> None:
    class _M:
        def generate(self, **kw):
            raise TypeError("some other arg")

    with pytest.raises(TypeError):
        _safe_generate(_M(), _FakeEncoding(), {})


# --------------------------------------------------------------------------- #
# chat 编排
# --------------------------------------------------------------------------- #
def _attached(model: FakeModel, tok: FakeTokenizer, **kw) -> LocalLLM:
    llm = LocalLLM(default_adapter="scalper", **kw)
    llm.attach(model, tok, adapters={"scalper", "analyst"})
    return llm


def test_chat_hot_swaps_adapter_and_restores_default() -> None:
    model, tok = FakeModel(), FakeTokenizer()
    llm = _attached(model, tok)
    llm.chat([{"role": "user", "content": "go"}], adapter="analyst")
    assert model.adapter_calls == ["analyst", "scalper"]  # 切到 analyst，生成后复位 scalper


def test_chat_no_adapter_still_restores_default() -> None:
    model, tok = FakeModel(), FakeTokenizer()
    llm = _attached(model, tok)
    llm.chat([{"role": "user", "content": "go"}])
    assert model.adapter_calls == ["scalper"]


def test_chat_injects_no_think_suffix() -> None:
    model, tok = FakeModel(), FakeTokenizer()
    llm = _attached(model, tok)
    llm.chat([{"role": "user", "content": "go"}])
    assert tok.seen_messages[-1]["content"].endswith("/no_think")


def test_chat_strips_think_tags_from_output() -> None:
    model = FakeModel()
    tok = FakeTokenizer(decode_value="<think>reasoning</think>BUY")
    llm = _attached(model, tok)
    assert llm.chat([{"role": "user", "content": "go"}]) == "BUY"


def test_chat_decodes_only_new_tokens() -> None:
    # input_len=3, output_ids 有 5 个 -> 解码 [10, 11]
    model = FakeModel(output_ids=[[0, 1, 2, 10, 11]])
    tok = FakeTokenizer(input_len=3, decode_value="x")
    llm = _attached(model, tok)
    llm.chat([{"role": "user", "content": "go"}])
    assert tok.decoded_ids == [10, 11]


def test_chat_applies_context_truncation_when_set() -> None:
    model, tok = FakeModel(), FakeTokenizer()
    llm = _attached(model, tok, max_context=1024)
    llm.chat([{"role": "user", "content": "go"}])
    assert tok.seen_tk_kwargs.get("truncation") is True
    assert tok.seen_tk_kwargs.get("max_length") == 1024


def test_chat_no_truncation_by_default() -> None:
    model, tok = FakeModel(), FakeTokenizer()
    llm = _attached(model, tok)  # max_context=0
    llm.chat([{"role": "user", "content": "go"}])
    assert "truncation" not in tok.seen_tk_kwargs


def test_chat_returns_empty_when_not_loaded(monkeypatch) -> None:
    llm = LocalLLM()
    monkeypatch.setattr(llm, "load", lambda *a, **k: None)  # 阻止真加载
    assert llm.chat([{"role": "user", "content": "x"}]) == ""


# --------------------------------------------------------------------------- #
# 配置 / 生成参数 / 状态
# --------------------------------------------------------------------------- #
def test_from_config_maps_fields() -> None:
    llm = LocalLLM.from_config(AppConfig().llm)
    assert llm.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert llm.quantization == "8bit"
    assert llm.default_adapter == "scalper"
    assert llm.max_new_tokens == 512


def test_generation_kwargs_greedy_when_temp_zero() -> None:
    llm = LocalLLM(temperature=0.0, gen_max_time_sec=7.0)
    kw = llm._generation_kwargs(None, None, FakeTokenizer())
    assert kw["do_sample"] is False
    assert kw["max_time"] == 7.0
    assert kw["pad_token_id"] == 0


def test_status_not_loaded() -> None:
    assert LocalLLM().status()["loaded"] is False


def test_import_does_not_require_torch() -> None:
    # inference 模块本身可导入（torch 仅在方法内懒导入）
    assert hasattr(inference, "LocalLLM")
