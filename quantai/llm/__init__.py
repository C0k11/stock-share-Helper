"""quantai.llm —— 本地 LLM 推理与对齐训练层（Qwen2.5-7B + QLoRA + DPO）。

四件套：
    prompts   -> 纯逻辑（消息构建 / think 清洗 / 响应解析 / 量化选择 / adapter 路径），无 torch，可全单测
    inference -> 本地推理（4/8bit 量化加载、chat、MoE adapter 热切换），torch 懒导入
    finetune  -> QLoRA 监督微调（FineTuner），torch/peft 懒导入
    dpo       -> DPO 偏好对齐训练（DPORunner），trl 懒导入

设计：inference/finetune/dpo 的 torch/transformers/peft/trl 全部在函数内部**懒导入**，
因此 `import quantai.llm` 与 `import quantai.llm.prompts` 不需要 GPU 依赖即可成功。

用法：
    from quantai.llm import build_messages, parse_decision   # 纯逻辑
    from quantai.llm import LocalLLM                          # 推理（用到时才加载 torch）
"""

from .inference import LocalLLM, decode_new_tokens
from .prompts import (
    DECISIONS,
    build_messages,
    format_messages_fallback,
    normalize_decision,
    parse_decision,
    resolve_adapter_path,
    select_quantization,
    strip_code_fences,
    strip_think_tags,
    with_no_think_suffix,
)

__all__ = [
    # prompts（纯逻辑）
    "DECISIONS",
    "build_messages",
    "with_no_think_suffix",
    "format_messages_fallback",
    "strip_think_tags",
    "strip_code_fences",
    "normalize_decision",
    "parse_decision",
    "select_quantization",
    "resolve_adapter_path",
    # inference（lazy torch）
    "LocalLLM",
    "decode_new_tokens",
]
