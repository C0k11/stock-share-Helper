"""quantai.llm.prompts 纯逻辑测试（无 torch，全分支覆盖）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantai.llm import prompts


# --------------------------------------------------------------------------- #
# 消息构建
# --------------------------------------------------------------------------- #
def test_build_messages_with_system() -> None:
    msgs = prompts.build_messages("hi", system="be brief")
    assert msgs == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_without_system() -> None:
    assert prompts.build_messages("hi") == [{"role": "user", "content": "hi"}]


def test_with_no_think_suffix_appends_and_is_pure() -> None:
    original = [{"role": "user", "content": "hello"}]
    out = prompts.with_no_think_suffix(original)
    assert out[-1]["content"] == "hello /no_think"
    # 不就地修改入参
    assert original[-1]["content"] == "hello"


def test_with_no_think_suffix_noop_when_last_not_user() -> None:
    msgs = [{"role": "assistant", "content": "x"}]
    assert prompts.with_no_think_suffix(msgs) == msgs


def test_format_messages_fallback() -> None:
    text = prompts.format_messages_fallback(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert text == "[system]\nS\n\n[user]\nU\n\n[assistant]\n"


# --------------------------------------------------------------------------- #
# think / fence 清洗
# --------------------------------------------------------------------------- #
def test_strip_think_tags_paired() -> None:
    assert prompts.strip_think_tags("<think>reasoning</think>BUY") == "BUY"


def test_strip_think_tags_unclosed() -> None:
    assert prompts.strip_think_tags("answer<think>dangling") == "answer"


def test_strip_think_tags_only_think_falls_back_to_raw() -> None:
    # 模型只输出了思考、清洗后为空 -> 回退原始文本（不当成空）
    raw = "<think>only thinking</think>"
    assert prompts.strip_think_tags(raw) == raw


def test_strip_think_tags_passthrough() -> None:
    assert prompts.strip_think_tags("plain text") == "plain text"


def test_strip_code_fences() -> None:
    assert prompts.strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'


# --------------------------------------------------------------------------- #
# 响应解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [("buy", "BUY"), ("SELL", "SELL"), ("Hold", "HOLD"), ("maybe", "HOLD"), (None, "HOLD")],
)
def test_normalize_decision(value, expected) -> None:
    assert prompts.normalize_decision(value) == expected


def test_parse_decision_clean_json() -> None:
    out = prompts.parse_decision('{"decision": "BUY", "analysis": "momentum"}')
    assert out == {"decision": "BUY", "analysis": "momentum"}


def test_parse_decision_json_in_fences_with_prose() -> None:
    raw = 'Here is my answer:\n```json\n{"decision":"SELL","analysis":"weak"}\n```\nthanks'
    assert prompts.parse_decision(raw) == {"decision": "SELL", "analysis": "weak"}


def test_parse_decision_single_quotes_via_ast() -> None:
    # 非严格 JSON（单引号）-> json.loads 失败、ast.literal_eval 兜住
    assert prompts.parse_decision("{'decision': 'HOLD', 'analysis': 'flat'}") == {
        "decision": "HOLD",
        "analysis": "flat",
    }


def test_parse_decision_invalid_decision_field_becomes_hold() -> None:
    out = prompts.parse_decision('{"decision": "LONG", "analysis": "x"}')
    assert out["decision"] == "HOLD"
    assert out["analysis"] == "x"


def test_parse_decision_missing_analysis_default() -> None:
    out = prompts.parse_decision('{"decision": "BUY"}')
    assert out == {"decision": "BUY", "analysis": "(no analysis)"}


def test_parse_decision_regex_fallback_decision() -> None:
    out = prompts.parse_decision("Final thoughts. decision: BUY because trend up")
    assert out["decision"] == "BUY"
    assert "trend up" in out["analysis"]


def test_parse_decision_regex_fallback_final() -> None:
    assert prompts.parse_decision("FINAL = sell").get("decision") == "SELL"


def test_parse_decision_garbage_is_parse_failed() -> None:
    assert prompts.parse_decision("no idea what to do") == {
        "decision": "HOLD",
        "analysis": "parse_failed",
    }


def test_parse_decision_empty() -> None:
    assert prompts.parse_decision("")["analysis"] == "parse_failed"


# --------------------------------------------------------------------------- #
# 量化 / adapter 路径
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "use_4bit,use_8bit,expected",
    [(True, True, "4bit"), (True, False, "4bit"), (False, True, "8bit"), (False, False, "fp16")],
)
def test_select_quantization(use_4bit, use_8bit, expected) -> None:
    assert prompts.select_quantization(use_4bit, use_8bit) == expected


def test_resolve_adapter_path_relative_resolves_under_root(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    target = tmp_path / "models" / "adapter"
    target.mkdir()
    out = prompts.resolve_adapter_path("models/adapter", tmp_path)
    assert Path(out) == target.resolve()


def test_resolve_adapter_path_lora_weights_fallback_to_parent(tmp_path: Path) -> None:
    # 指向不存在的 .../lora_weights，但父目录存在 -> 回退父目录
    parent = tmp_path / "trader_v1"
    parent.mkdir()
    out = prompts.resolve_adapter_path(str(parent / "lora_weights"), tmp_path)
    assert Path(out) == parent


def test_resolve_adapter_path_missing_returns_as_is(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    out = prompts.resolve_adapter_path(str(missing), tmp_path)
    assert Path(out) == missing
