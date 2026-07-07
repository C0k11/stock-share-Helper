"""DeepSeek 教师模型客户端（OpenAI 兼容 chat/completions，requests 实现）。

密钥安全（硬要求）：
- key **只从环境变量**（默认 `DEEPSEEK_API_KEY`）读取——构造时读取，缺失立刻报
  `MissingApiKeyError`（fail-fast，不发半个请求）；
- 代码/配置/日志里**永不出现** key 本体；`.env` / `.local/.env` 由 CLI 层用
  python-dotenv 注入（两个路径都在 .gitignore）。

成本闸（硬要求）：本类只在 `scripts/distill.py --run --confirm-spend` 的手动路径
被实例化；测试/CI/loop 一律用 `MockDeepSeekClient`（零真实调用）。`usage_totals`
累计 token 用量，跑完打印对账。

模型名 config 驱动（`distill.model`，默认 `deepseek-v4-pro`，教师质量优先；预算紧可切
`deepseek-v4-flash`）。注意：旧别名 `deepseek-chat` / `deepseek-reasoner` 于
2026-07-24 15:59 UTC 退役，之后调用直接失败，故用显式版本名。思考模式
（`thinking: {"type": "enabled"}`）默认打开，蒸馏走深度推理路径。
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests


class MissingApiKeyError(RuntimeError):
    """环境变量里没有 API key：停下来让用户配，绝不带着空 key 继续。"""


class DeepSeekClient:
    """最小可用的 OpenAI 兼容聊天客户端（仅蒸馏管线所需）。"""

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout_sec: float = 120.0,
        max_retries: int = 3,
        thinking: bool = True,
    ) -> None:
        key = os.environ.get(api_key_env, "").strip()
        if not key:
            raise MissingApiKeyError(
                f"环境变量 {api_key_env} 未设置。请把 key 放进 .env 或 .local/.env"
                f"（两者都被 .gitignore 排除），或在 shell 里 export 后重试。"
            )
        self._key = key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        #: 深度思考模式（DeepSeek V4 系）：True 时请求体带 thinking: {"type": "enabled"}。
        self.thinking = thinking
        #: 累计用量（对账用）：{"prompt_tokens": int, "completion_tokens": int, "requests": int}
        self.usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def chat(self, messages: list[dict], temperature: Optional[float] = None) -> str:
        """一次聊天补全，返回 assistant 文本。5xx/429 指数退避重试。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.thinking:
            payload["thinking"] = {"type": "enabled"}
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code in (429, 500, 502, 503):
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                self.usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
                self.usage_totals["requests"] += 1
                content = data["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    # 思考模式踩坑（实测）：max_tokens 被推理链耗尽时 content 为空、
                    # token 照计——必须当失败抛出，绝不让空答案混进训练集。
                    raise RuntimeError(
                        "teacher returned empty content（疑似 thinking 耗尽 max_tokens，"
                        f"finish_reason={data['choices'][0].get('finish_reason')!r}）"
                    )
                return content
            except requests.RequestException as exc:  # 网络层错误也退避重试
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"DeepSeek 请求在 {self.max_retries} 次重试后仍失败: {last_err}")


class MockDeepSeekClient:
    """测试/干跑用假客户端：零网络、确定性输出、记录收到的请求。

    与 `DeepSeekClient` 同鸭子接口（`chat` + `model` + `usage_totals`）。
    """

    def __init__(self, model: str = "mock-deepseek", canned: Optional[str] = None) -> None:
        self.model = model
        self.canned = canned
        self.calls: list[list[dict]] = []
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    def chat(self, messages: list[dict], temperature: Optional[float] = None) -> str:
        self.calls.append(messages)
        self.usage_totals["requests"] += 1
        if self.canned is not None:
            return self.canned
        # 确定性、结构合规的假教师输出（够格式校验用）
        user = messages[-1]["content"] if messages else ""
        head = user.splitlines()[0] if user else "unknown"
        return (
            f"【结论】基于给定数据的模拟分析（{head}），中性偏多，置信度 60%。\n"
            "【依据】RSI 与均线数据见输入简报；此为 mock 输出，仅用于管线测试。\n"
            "【风险】mock 客户端不产生真实分析。"
        )
