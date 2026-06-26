"""数据飞轮路由：构建偏好数据集 + 看 active adapter 指针。

诚实：只暴露离线数据飞轮的真实状态。在线实时梯度未实现（B-1），无对应 endpoint。
真训练（DPO）走 `scripts/evolve.py --train`（GPU），不在 HTTP 同步请求里跑。
"""
from __future__ import annotations

from fastapi import APIRouter

from quantai.api.deps import get_state

router = APIRouter(prefix="/evolution", tags=["evolution"])


@router.post("/build-dataset")
def build_dataset() -> dict:
    trainer = get_state().trainer()
    path = trainer.build_dataset()
    pairs = trainer.preferences.generate_preference_pairs()
    records = trainer.to_dpo_records(pairs)
    return {"ok": True, "dataset": str(path), "pairs": len(records)}


@router.get("/active")
def active_adapters() -> dict:
    return {"active": get_state().trainer().get_active_adapters()}
