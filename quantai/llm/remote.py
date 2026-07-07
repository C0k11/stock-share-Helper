"""外接远程分析员：任意 OpenAI 兼容 chat API 顶替本地 GPU 模型出分析。

场景：本地显卡不够/被占（训练中）时，接自己买的 API 当分析员——DeepSeek、
OpenAI、Anthropic 的 OpenAI 兼容端点、本地 Ollama/vLLM 等任何
`POST {base_url}/chat/completions` 协议的服务都能接。

接口契约：与 `quantai.llm.inference.LocalLLM` 同鸭子型——
`generate(user, system=None) -> str` + `model_name` 属性；`reporting.load_report_llm`
按 `llm.remote.enabled` 决定给谁，下游（日报/作战台守护线程/新闻打分）零改动。

诚实与钱安全：
- 远程调用**花真钱**——默认 `enabled: false`，必须在 config 显式打开；
- key 只从环境变量读（`api_key_env`，缺失 fail-fast），永不入配置文件/代码/日志；
- 底层复用 `DeepSeekClient`（重试退避、4xx 不重试、空答案抛错、usage 对账累计）。
"""

from __future__ import annotations

from typing import Any, Optional

from quantai.distill.client import DeepSeekClient


class RemoteAnalyst:
    """OpenAI 兼容远程 LLM 的报告/作战台适配器（LocalLLM 的即插替身）。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        thinking: bool = False,
        timeout_sec: float = 120.0,
    ) -> None:
        self._client = DeepSeekClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            timeout_sec=timeout_sec,
        )
        self.model_name = f"remote:{model}"
        # LocalLLM 兼容面：调用方会赋值这些属性（远程侧 max_tokens 建构时已定，
        # 这里保留字段容忍赋值，不再生效——诚实说明而不是悄悄吞掉）
        self.gen_max_time_sec: float = timeout_sec
        self.max_new_tokens: int = max_tokens

    @classmethod
    def from_config(cls, rcfg: Any) -> "RemoteAnalyst":
        """从 `LLMRemoteConfig` 构造（调用方负责先检查 rcfg.enabled）。"""
        return cls(
            model=rcfg.model,
            base_url=rcfg.base_url,
            api_key_env=rcfg.api_key_env,
            temperature=rcfg.temperature,
            max_tokens=rcfg.max_tokens,
            thinking=rcfg.thinking,
        )

    @property
    def usage_totals(self) -> dict:
        """token 用量对账（跨调用累计）。"""
        return dict(self._client.usage_totals)

    def generate(self, user: str, system: Optional[str] = None, **_: Any) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        return self._client.chat(messages)
