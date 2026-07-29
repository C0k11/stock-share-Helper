"""DuckDB 仓库连接（config 驱动路径）。

为什么 DuckDB：嵌入式（零服务）、列存 + 向量化 SQL（分析型查询快）、与 pandas/parquet
零摩擦（`con.sql(...).df()` / 直接 `read_parquet`）、BI 工具可经 ODBC/JDBC 直连。
本地单机分析仓库的工业默认选择（Postgres 是多人/服务场景的可选升级，见 modules/warehouse.md）。

分层约定（ELT）：
- `raw`     ：pandas ETL 原样落地（只加 loaded_at 审计列），**不做业务变换**。
- `staging` ：dbt 视图，清洗/改名/类型收敛（1:1 对应 raw 表）。
- `marts`   ：dbt 表，星型模型（dim_symbol / dim_date + fact_*），BI 直接查这层。
"""

from __future__ import annotations

from pathlib import Path

import duckdb

#: 三层 schema 名（dbt 侧 dbt_project.yml 与此保持一致）
SCHEMAS: tuple[str, ...] = ("raw", "staging", "marts")


def connect(db_path: str | Path = "data/warehouse/quantai.duckdb") -> duckdb.DuckDBPyConnection:
    """打开（必要时创建）仓库库文件，并确保三层 schema 存在。

    `db_path=":memory:"` 给测试用（全离线、零落盘）。
    """
    p = str(db_path)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(p)
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return con
