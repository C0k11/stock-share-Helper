"""quantai.ui.client.QuantAIClient 测试。

用 httpx.MockTransport 把客户端请求桥接到真实 FastAPI app 的 TestClient——既验证客户端的
请求构造/响应解析，又验证它与 api/ 的契约，全程进程内、无需起服务器。
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from quantai.api import create_app
from quantai.config.schema import AppConfig, EvolutionConfig
from quantai.ui.client import QuantAIClient


class FakeLLM:
    def chat(self, message: str) -> str:
        return f"echo: {message}"


def _make_client(tmp_path, *, llm=None) -> QuantAIClient:
    cfg = AppConfig(
        evolution=EvolutionConfig(
            trajectories_dir=str(tmp_path / "traj"),
            experiences_dir=str(tmp_path / "exp"),
            preferences_dir=str(tmp_path / "pref"),
            adapters_dir=str(tmp_path / "adapters"),
            active_pointer=str(tmp_path / "active.json"),
        )
    )
    tc = TestClient(create_app(cfg, llm=llm))

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        resp = tc.request(
            request.method,
            request.url.path,
            params=dict(request.url.params),
            content=request.content,
            headers=headers,
        )
        return httpx.Response(status_code=resp.status_code, content=resp.content)

    return QuantAIClient(base_url="http://test", transport=httpx.MockTransport(handler))


def test_health(tmp_path):
    c = _make_client(tmp_path)
    assert c.health()["status"] == "ok"


def test_live_lifecycle(tmp_path):
    c = _make_client(tmp_path)
    assert c.live_status() == {"active": False}
    res = c.start_live(["NVDA"], source="simulated", seed=1, interval_sec=0.05)
    assert res["ok"] is True and res["status"]["feed_source"] == "simulated"
    st = c.live_status()
    assert st["active"] is True and st["cash"] == 100000.0
    assert c.stop_live()["stopped"] is True
    assert c.live_status() == {"active": False}


def test_live_trades(tmp_path):
    c = _make_client(tmp_path)
    c.start_live(["NVDA"], source="simulated", seed=1, interval_sec=0.05)
    t = c.live_trades(limit=10)
    assert "trades" in t and "count" in t
    c.stop_live()


def test_submit_feedback(tmp_path):
    c = _make_client(tmp_path)
    assert c.submit_feedback("ref1", 1, "good")["ok"] is True


def test_build_dataset_and_active(tmp_path):
    c = _make_client(tmp_path)
    assert c.build_dataset()["pairs"] == 0
    assert c.active_adapters() == {"active": {}}


def test_chat_without_llm_raises(tmp_path):
    c = _make_client(tmp_path)
    with pytest.raises(httpx.HTTPStatusError):
        c.chat("hi")  # 503


def test_chat_with_llm(tmp_path):
    c = _make_client(tmp_path, llm=FakeLLM())
    res = c.chat("hello")
    assert res["reply"] == "echo: hello"


def test_context_manager(tmp_path):
    with _make_client(tmp_path) as c:
        assert c.health()["status"] == "ok"
