import ast
import json
import re
from typing import Any, Optional, Tuple


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_markdown_code_fence(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    return text


def _find_json_span(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None

    start = -1
    for ch in ("{", "["):
        i = text.find(ch)
        if i >= 0 and (start < 0 or i < start):
            start = i

    if start < 0:
        return None

    depth_curly = 0
    depth_square = 0
    in_str = False
    esc = False
    quote = '"'

    for i in range(start, len(text)):
        c = text[i]

        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                in_str = False
            continue

        if c in ('"', "'"):
            in_str = True
            quote = c
            continue

        if c == "{":
            depth_curly += 1
        elif c == "}":
            depth_curly -= 1
        elif c == "[":
            depth_square += 1
        elif c == "]":
            depth_square -= 1

        if depth_curly <= 0 and depth_square <= 0 and i > start:
            return start, i + 1

    return start, len(text)


def extract_json_text(text: str) -> Optional[str]:
    if not text:
        return None

    s = _strip_markdown_code_fence(text).strip()

    span = _find_json_span(s)
    if not span:
        return None

    start, end = span
    cand = s[start:end].strip()

    if cand.startswith("{"):
        open_cnt = cand.count("{")
        close_cnt = cand.count("}")
        if close_cnt < open_cnt:
            cand = cand + ("}" * (open_cnt - close_cnt))

    if cand.startswith("["):
        open_cnt = cand.count("[")
        close_cnt = cand.count("]")
        if close_cnt < open_cnt:
            cand = cand + ("]" * (open_cnt - close_cnt))

    return cand


def _normalize_common_tokens(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("\u00a0", " ")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s


def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _try_literal_eval(s: str) -> Optional[Any]:
    py = s
    py = re.sub(r"\bnull\b", "None", py)
    py = re.sub(r"\btrue\b", "True", py)
    py = re.sub(r"\bfalse\b", "False", py)

    try:
        return ast.literal_eval(py)
    except Exception:
        return None


def repair_and_parse_json(text: str) -> Any:
    if not text:
        return None

    cand = extract_json_text(text)
    if not cand:
        return None

    obj = _try_json_loads(cand)
    if obj is not None:
        return obj

    cand2 = _normalize_common_tokens(cand)

    obj = _try_json_loads(cand2)
    if obj is not None:
        return obj

    obj = _try_literal_eval(cand2)
    if obj is not None:
        return obj

    return None


if __name__ == "__main__":
    bad_json = """Here is the output: ```json\n {"action": "BUY", "reason": "Good news"} \n``` Hope it helps."""
    print(repair_and_parse_json(bad_json))
