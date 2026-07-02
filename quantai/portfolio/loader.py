"""真实持仓录入：`portfolio.local.yaml`（或 CSV）→ 类型化 `Portfolio`。

隐私边界（硬要求）：真实持仓文件是 `portfolio.local.yaml`，被仓库 `.gitignore` 的
`*.local.yaml` 规则排除，**永不入库**；仓库只带 `portfolio.example.yaml` 示例。

格式（YAML）：
```yaml
cash: 5000.0            # 可选，默认 0
positions:
  - symbol: NVDA
    shares: 10          # 股数（>0 多头；<0 空头也接受但需显式）
    cost_basis: 450.50  # 每股成本
    open_date: 2025-11-03
```
CSV 等价列：`symbol,shares,cost_basis,open_date`（CSV 无现金概念，加载后 cash=0；
要录现金请用 YAML 格式）。

同一 symbol 允许多条（分批建仓的 lot），分析层聚合为加权平均成本。
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Position(BaseModel):
    """单笔持仓（lot）。"""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    shares: float
    cost_basis: float = Field(ge=0, description="每股成本（>=0）")
    open_date: date

    @field_validator("symbol")
    @classmethod
    def _norm_symbol(cls, v: str) -> str:
        s = v.strip().upper()
        if not s:
            raise ValueError("symbol 不能为空")
        return s

    @field_validator("shares")
    @classmethod
    def _nonzero_shares(cls, v: float) -> float:
        if v == 0:
            raise ValueError("shares 不能为 0（清仓的持仓请直接删掉这行）")
        return v


class Portfolio(BaseModel):
    """真实组合：现金 + 持仓 lots。"""

    model_config = ConfigDict(extra="forbid")

    cash: float = Field(default=0.0, ge=0)
    positions: list[Position] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        """去重后的持仓标的（保持首次出现顺序）。"""
        seen: dict[str, None] = {}
        for p in self.positions:
            seen.setdefault(p.symbol, None)
        return list(seen)

    def total_cost(self) -> float:
        """全部持仓的成本总额 Σ shares × cost_basis（空头 lot 为负）。"""
        return float(sum(p.shares * p.cost_basis for p in self.positions))


def load_portfolio(path: str | Path) -> Portfolio:
    """从 YAML 或 CSV 加载真实持仓。

    - 文件不存在 → `FileNotFoundError`，报错信息指向 `portfolio.example.yaml`。
    - 未知字段 / 非法值 → pydantic `ValidationError`（诚实报错，不静默丢弃）。
    - 扩展名 `.yaml`/`.yml` 走 YAML，`.csv` 走 CSV，其它报 `ValueError`。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"持仓文件不存在: {p}。请参照 portfolio.example.yaml 创建（真实文件用 "
            f"*.local.yaml 命名，已被 .gitignore 排除、不会入库）。"
        )
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML 顶层应为 mapping，收到 {type(raw).__name__}")
        return Portfolio(**raw)
    if suffix == ".csv":
        with p.open(newline="", encoding="utf-8-sig") as f:
            # 值侧 (v or "")：行尾字段缺失时 DictReader 填 None，裸 .strip() 会
            # AttributeError；归一成空串让 pydantic 报干净的 ValidationError。
            rows = [
                {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
                for row in csv.DictReader(f)
                if any((v or "").strip() for v in row.values() if not isinstance(v, list))
            ]
        return Portfolio(positions=[Position(**row) for row in rows])
    raise ValueError(f"不支持的持仓文件格式 {suffix!r}（支持 .yaml/.yml/.csv）")
