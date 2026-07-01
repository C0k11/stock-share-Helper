-- staging：真实持仓 lots（原粒度保留——聚合到 symbol 粒度是 fact_positions 的事）。
select
    as_of,
    upper(symbol)  as symbol,
    shares,
    cost_basis,
    open_date
from {{ source('raw', 'positions') }}
where shares <> 0
