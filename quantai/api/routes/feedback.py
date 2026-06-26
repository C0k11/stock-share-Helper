"""人工反馈路由：把 ref_id 的好坏评分写进 evolution 轨迹（喂数据飞轮）。"""
from __future__ import annotations

from fastapi import APIRouter

from quantai.api.deps import get_state
from quantai.api.schemas import FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    get_state().recorder.log_feedback(ref_id=req.ref_id, score=req.score, comment=req.comment)
    return {"ok": True, "ref_id": req.ref_id}
