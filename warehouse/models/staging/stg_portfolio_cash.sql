-- staging：组合现金快照。
select
    as_of,
    cash
from {{ source('raw', 'portfolio_cash') }}
