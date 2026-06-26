"""Secretary 对话路由。

诚实：chat 需要一个真实 LLM（`ApiState.llm`，如 `quantai.llm.LocalLLM` 切到 secretary adapter）。
**没有配 LLM 时返回 503**（不伪造回复）。配了就调 `llm.chat(message)` 出真回复，并把这轮记进
evolution 轨迹，返回 `message_id` 供后续 `/feedback` 评分。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from quantai.api.deps import get_state
from quantai.api.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    text = str(req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")

    state = get_state()
    if state.llm is None:
        raise HTTPException(
            status_code=503,
            detail="secretary LLM not configured (no model loaded). Inject ApiState.llm to enable chat.",
        )

    try:
        reply = str(state.llm.chat(text))
    except Exception as exc:  # 真实推理失败如实报错，不伪造
        raise HTTPException(status_code=500, detail=f"chat inference failed: {exc}")

    message_id = None
    try:
        message_id = state.recorder.record(
            agent_id="secretary",
            context=json.dumps({"message": text, "context": req.context}, ensure_ascii=False),
            action=reply,
            feedback="wait_for_user",
        )
    except Exception:
        message_id = None

    return {"reply": reply, "message_id": message_id}
