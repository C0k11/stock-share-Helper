"""quantai.api 测试（FastAPI TestClient）。验证真实 endpoint + 假 endpoint 已删。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from quantai.api import create_app
from quantai.config.schema import AppConfig, EvolutionConfig


def _cfg(tmp_path) -> AppConfig:
    ev = EvolutionConfig(
        trajectories_dir=str(tmp_path / "traj"),
        experiences_dir=str(tmp_path / "exp"),
        preferences_dir=str(tmp_path / "pref"),
        adapters_dir=str(tmp_path / "adapters"),
        active_pointer=str(tmp_path / "active.json"),
    )
    return AppConfig(evolution=ev)


def _client(tmp_path, *, llm=None) -> TestClient:
    return TestClient(create_app(_cfg(tmp_path), llm=llm))


class FakeLLM:
    def chat(self, message: str) -> str:
        return f"echo: {message}"


# --------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------- #
def test_root_and_health(tmp_path):
    c = _client(tmp_path)
    assert c.get("/").status_code == 200
    r = c.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


# --------------------------------------------------------------------- #
# C-2 / C-6：假 endpoint 必须已删
# --------------------------------------------------------------------- #
def test_fake_endpoints_removed(tmp_path):
    c = _client(tmp_path)
    for path in (
        "/api/v1/market/regime",
        "/api/v1/recommendations",
        "/api/v1/portfolio/performance",
        "/api/v1/risk/alerts",
        "/api/v1/news/summary",
    ):
        assert c.get(path).status_code == 404, f"{path} 应已删除"


# --------------------------------------------------------------------- #
# live 会话
# --------------------------------------------------------------------- #
def test_live_status_inactive_then_start_stop(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/v1/live/status").json() == {"active": False}

    r = c.post("/api/v1/live/start", json={"tickers": ["NVDA"], "source": "simulated", "seed": 1, "interval_sec": 0.05})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["status"]["feed_source"] == "simulated"

    st = c.get("/api/v1/live/status").json()
    assert st["active"] is True and st["cash"] == 100000.0

    assert c.post("/api/v1/live/stop").json()["stopped"] is True
    assert c.get("/api/v1/live/status").json() == {"active": False}


def test_live_trades_404_without_session(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/v1/live/trades").status_code == 404


def test_live_trades_returns_orders(tmp_path):
    c = _client(tmp_path)
    c.post("/api/v1/live/start", json={"tickers": ["NVDA"], "source": "simulated", "seed": 1, "interval_sec": 0.05})
    r = c.get("/api/v1/live/trades")
    assert r.status_code == 200
    body = r.json()
    assert "trades" in body and "count" in body
    c.post("/api/v1/live/stop")


# --------------------------------------------------------------------- #
# feedback -> 真实落盘
# --------------------------------------------------------------------- #
def test_feedback_writes_trajectory(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/v1/feedback", json={"ref_id": "abc", "score": 1, "comment": "good"})
    assert r.status_code == 200 and r.json()["ok"] is True
    # 真写进 evolution 轨迹目录
    files = list((tmp_path / "traj").glob("*.jsonl"))
    assert files
    rows = [json.loads(x) for x in files[0].read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(r0["type"] == "feedback" and r0["ref_id"] == "abc" for r0 in rows)


# --------------------------------------------------------------------- #
# evolution
# --------------------------------------------------------------------- #
def test_evolution_build_dataset_empty(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/v1/evolution/build-dataset")
    assert r.status_code == 200
    assert r.json()["pairs"] == 0  # 还没攒数据


def test_evolution_active_empty(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/v1/evolution/active").json() == {"active": {}}


# --------------------------------------------------------------------- #
# chat：无 LLM -> 503；有 LLM -> 真回复
# --------------------------------------------------------------------- #
def test_chat_503_without_llm(tmp_path):
    c = _client(tmp_path)
    assert c.post("/api/v1/chat", json={"message": "hi"}).status_code == 503


def test_chat_with_injected_llm(tmp_path):
    c = _client(tmp_path, llm=FakeLLM())
    r = c.post("/api/v1/chat", json={"message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "echo: hello"
    assert body["message_id"]  # 记进了轨迹


def test_chat_empty_message_400(tmp_path):
    c = _client(tmp_path, llm=FakeLLM())
    assert c.post("/api/v1/chat", json={"message": "  "}).status_code == 400
