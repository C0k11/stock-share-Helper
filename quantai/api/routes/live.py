"""实盘会话路由：启停 + 真实状态/成交。

只读真实状态（`LiveRunner.snapshot()` / `broker.orders`）。无会话时 status 返回 active=False，
trades 返回 404--**不返回假数据**（对比旧版的 fake regime/performance/alerts，C-2 已删）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from quantai.api.deps import get_state
from quantai.api.schemas import StartLiveRequest

router = APIRouter(prefix="/live", tags=["live"])


@router.post("/start")
def start_live(req: StartLiveRequest) -> dict:
    runner = get_state().start_runner(
        req.tickers,
        source=req.source,
        cash=req.cash,
        interval_sec=req.interval_sec,
        allow_short=req.allow_short,
        seed=req.seed,
        collect=req.collect,
    )
    return {"ok": True, "status": runner.snapshot()}


@router.post("/stop")
def stop_live() -> dict:
    return {"ok": True, "stopped": get_state().stop_runner()}


@router.get("/status")
def live_status() -> dict:
    runner = get_state().runner
    if runner is None:
        return {"active": False}
    snap = runner.snapshot()
    snap["active"] = True
    return snap


@router.get("/trades")
def live_trades(limit: int = 200) -> dict:
    runner = get_state().runner
    if runner is None:
        raise HTTPException(status_code=404, detail="no live session")
    orders = list(runner.broker.orders)
    if limit > 0:
        orders = orders[-limit:]
    return {"trades": orders, "count": len(orders)}
