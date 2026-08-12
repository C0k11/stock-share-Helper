"""API 依赖与应用状态 —— 取代旧 `src/api/main.py` 的一堆模块级全局。

`ApiState` 持有：唯一的 `LiveRunner`（实盘会话）、`EvolutionRecorder`（反馈/轨迹）、可选的
`LocalLLM`（chat 用，默认无->chat 返回 503，不伪造）。路由通过 `get_state()` 取它，测试可重置。

诚实边界：**只暴露真实状态**（来自 `LiveRunner.snapshot()` / broker / evolution）。旧版 5 个
假数据 endpoint（regime/recommendations/performance/alerts/news_summary，C-2）与 movers 占位
（C-6）**一律不迁**。
"""
from __future__ import annotations

from typing import Any, List, Optional

from quantai.config.schema import AppConfig
from quantai.evolution.recorder import EvolutionRecorder
from quantai.evolution.trainer import EvolutionTrainer
from quantai.live.runner import LiveRunner


class ApiState:
    """进程内 API 状态：一个实盘会话 + 配置 + 可选 LLM。"""

    def __init__(self, cfg: Optional[AppConfig] = None, *, llm: Any = None) -> None:
        self.cfg: AppConfig = cfg or AppConfig()
        self.runner: Optional[LiveRunner] = None
        self.llm = llm
        self._recorder: Optional[EvolutionRecorder] = None

    # --- 实盘会话 --- #
    def start_runner(
        self,
        tickers: List[str],
        *,
        source: str = "simulated",
        cash: float = 100000.0,
        interval_sec: float = 1.0,
        allow_short: bool = False,
        seed: Optional[int] = None,
        collect: bool = False,
    ) -> LiveRunner:
        self.stop_runner()
        self.runner = LiveRunner(
            tickers,
            cfg=self.cfg,
            source=source,
            cash=cash,
            interval_sec=interval_sec,
            allow_short=allow_short,
            seed=seed,
            collect=collect,
            llm=self.llm,
        )
        self.runner.start()
        return self.runner

    def stop_runner(self) -> bool:
        if self.runner is None:
            return False
        try:
            self.runner.stop()
        finally:
            self.runner = None
        return True

    # --- evolution --- #
    @property
    def recorder(self) -> EvolutionRecorder:
        # 优先用实盘会话里那个（决策与反馈写同一处）；否则按配置新建。
        if self.runner is not None and getattr(self.runner, "recorder", None) is not None:
            return self.runner.recorder
        if self._recorder is None:
            self._recorder = EvolutionRecorder.from_config(self.cfg.evolution)
        return self._recorder

    def trainer(self) -> EvolutionTrainer:
        return EvolutionTrainer.from_config(self.cfg)


# 进程级单例（create_app 时重建；测试可替换）。
_state: Optional[ApiState] = None


def set_state(state: ApiState) -> None:
    global _state
    _state = state


def get_state() -> ApiState:
    global _state
    if _state is None:
        _state = ApiState()
    return _state
