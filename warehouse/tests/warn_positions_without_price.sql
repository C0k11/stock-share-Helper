-- 观测性断言（warn 级）：持仓估不出价（ASOF 找不到 as_of 当日或之前的收盘）。
-- 与 pandas 侧 missing_prices 契约对齐：缺价不许静默——BI 的 Σmarket_value 会
-- 无声少算这部分仓位。warn 不 fail：NULL 传播本身是诚实行为，缺的是"被看见"。
{{ config(severity='warn') }}
select as_of, symbol, shares
from {{ ref('fact_positions') }}
where last_close is null
