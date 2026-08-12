"""LLM 纯逻辑工具：消息构建 / think 标签清洗 / 响应解析 / 量化选择 / adapter 路径。

本模块**不导入 torch/transformers**，是 llm 层里可被完整单测的部分（CI 全绿）。
所有函数都从旧码忠实迁移（faithful），来源在各函数 docstring 标注：
- `strip_think_tags` / `select_quantization`：`src/llm/local_chat.py`
- `parse_decision` / `strip_code_fences` / `resolve_adapter_path`：`src/trading/strategy.py`
- `format_messages_fallback`：`scripts/train_dpo.py`
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

#: 合法交易决策（大写）。
DECISIONS = ("BUY", "SELL", "HOLD")

Message = Dict[str, str]

_THINK_PAIRED = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*", flags=re.DOTALL)
_DECISION_PATTERNS = (
    re.compile(r"\bdecision\b\s*[:=]\s*(BUY|SELL|HOLD)\b", flags=re.IGNORECASE),
    re.compile(r"\bfinal\s*[:=]\s*(BUY|SELL|HOLD)\b", flags=re.IGNORECASE),
)


# --------------------------------------------------------------------------- #
# 消息构建
# --------------------------------------------------------------------------- #
def build_messages(user: str, system: Optional[str] = None) -> List[Message]:
    """构造 chat messages：可选 system + 必有 user。"""
    messages: List[Message] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(user)})
    return messages


def with_no_think_suffix(messages: List[Message]) -> List[Message]:
    """给最后一条 user 消息追加 " /no_think"（关闭 Qwen3 思考模式）。

    返回新列表，不就地修改入参（旧 `local_chat.py` 是就地改，这里更安全）。
    """
    out = [dict(m) for m in messages]
    if out and out[-1].get("role") == "user":
        out[-1]["content"] = str(out[-1].get("content", "")) + " /no_think"
    return out


def format_messages_fallback(messages: List[Message]) -> str:
    """无 chat_template 时的纯文本回退格式（忠实迁移 `train_dpo.py::_format_prompt_messages`）。

    chat_template 存在时应优先用 tokenizer.apply_chat_template（那条在 inference 层，需要 tokenizer）。
    """
    chunks: List[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        chunks.append(f"[{role}]\n{content}")
    chunks.append("[assistant]\n")
    return "\n\n".join(chunks)


# --------------------------------------------------------------------------- #
# 文本清洗
# --------------------------------------------------------------------------- #
def strip_think_tags(text: str) -> str:
    """移除 Qwen3 `<think>...</think>`（含未闭合标签）。

    忠实迁移 `local_chat.py`：若清洗后为空（模型只输出了思考），则回退原始文本，
    避免上层把它当成"无有效内容"。
    """
    raw = str(text or "")
    cleaned = _THINK_PAIRED.sub("", raw)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    cleaned = cleaned.replace("</think>", "").strip()
    if not cleaned:
        return raw.strip()
    return cleaned


def strip_code_fences(text: str) -> str:
    """去掉 ```json / ``` 围栏（模型常把 JSON 包在 markdown 代码块里）。"""
    try:
        return str(text).replace("```json", "").replace("```", "").strip()
    except Exception:
        return str(text or "").strip()


# --------------------------------------------------------------------------- #
# 响应解析
# --------------------------------------------------------------------------- #
def normalize_decision(value: Any) -> str:
    """把任意值归一化为 BUY/SELL/HOLD；非法 -> HOLD（保守）。"""
    d = str(value or "").strip().upper()
    return d if d in DECISIONS else "HOLD"


def parse_decision(text: str) -> Dict[str, str]:
    """从模型原始输出解析出 `{"decision", "analysis"}`（忠实迁移 strategy.py）。

    顺序：(1) 去围栏 -> (2) 取首个 `{` 到末个 `}` 试 `json.loads`，失败再试 `ast.literal_eval`
    （容忍单引号等非严格 JSON）-> (3) 正则兜底 `decision[:=]...` / `final[:=]...` ->
    (4) 仍失败返回 `{"decision": "HOLD", "analysis": "parse_failed"}`（不乱交易）。
    """
    raw = strip_code_fences(text)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = raw[start : end + 1]
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(blob)
            except Exception:
                continue
            if isinstance(obj, dict):
                analysis = str(obj.get("analysis") or "").strip() or "(no analysis)"
                return {"decision": normalize_decision(obj.get("decision")), "analysis": analysis}

    for pattern in _DECISION_PATTERNS:
        m = pattern.search(raw)
        if m:
            tail = raw.replace("\n", " ").strip()
            return {"decision": str(m.group(1)).upper(), "analysis": tail}

    return {"decision": "HOLD", "analysis": "parse_failed"}


# --------------------------------------------------------------------------- #
# 量化 / adapter 路径
# --------------------------------------------------------------------------- #
def select_quantization(use_4bit: bool, use_8bit: bool) -> str:
    """把旧的两个互斥布尔解析成单一模式 "4bit"/"8bit"/"fp16"（4bit 优先）。

    忠实迁移 `local_chat.py`：旧码同时传 4bit+8bit 时会静默把 8bit 关掉；这里显式化。
    """
    if use_4bit:
        return "4bit"
    if use_8bit:
        return "8bit"
    return "fp16"


def resolve_adapter_path(raw: str, project_root: Union[str, Path]) -> str:
    """解析 LoRA adapter 路径（忠实迁移 strategy.py::_resolve_adapter_path）。

    相对路径按 project_root 展开；若指向不存在的 `.../lora_weights` 而父目录存在，
    回退到父目录（兼容旧配置把 adapter 放在父目录的情况）。
    """
    p = Path(str(raw or "").strip())
    if not p.is_absolute():
        p = (Path(project_root) / p).resolve()
    if p.exists():
        return str(p)
    if p.name.lower() == "lora_weights" and p.parent.exists():
        return str(p.parent)
    return str(p)
