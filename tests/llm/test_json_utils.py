"""quantai.llm.json_utils 测试（鲁棒 JSON 抽取/修复，忠实迁移自 llm_tools）。"""

from __future__ import annotations

from quantai.llm.json_utils import extract_json_text, repair_and_parse_json


def test_extract_from_code_fence():
    txt = 'Here: ```json\n{"a": 1}\n``` done'
    assert extract_json_text(txt) == '{"a": 1}'


def test_extract_plain_object_with_surrounding_text():
    assert extract_json_text('blah {"x": 2} tail') == '{"x": 2}'


def test_extract_none_when_no_json():
    assert extract_json_text("no json here") is None
    assert extract_json_text("") is None


def test_extract_completes_missing_brace():
    assert extract_json_text('{"a": 1') == '{"a": 1}'


def test_parse_valid_json():
    assert repair_and_parse_json('{"decision": "BUY"}') == {"decision": "BUY"}


def test_parse_single_quotes():
    assert repair_and_parse_json("{'decision': 'SELL'}") == {"decision": "SELL"}


def test_parse_smart_quotes():
    out = repair_and_parse_json('{\u201cdecision\u201d: \u201cHOLD\u201d}')
    assert out == {"decision": "HOLD"}


def test_parse_trailing_comma():
    assert repair_and_parse_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_parse_python_literals():
    out = repair_and_parse_json('{"accept": True, "x": None, "y": False}')
    assert out == {"accept": True, "x": None, "y": False}


def test_parse_nested_with_braces_in_string():
    out = repair_and_parse_json('{"reason": "use {x} here", "ok": true}')
    assert out == {"reason": "use {x} here", "ok": True}


def test_parse_garbage_returns_none():
    assert repair_and_parse_json("totally not json") is None
    assert repair_and_parse_json("") is None


def test_parse_array():
    assert repair_and_parse_json("[1, 2, 3]") == [1, 2, 3]
