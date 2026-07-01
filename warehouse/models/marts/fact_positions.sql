-- fact_positions：持仓事实表（粒度 = as_of × symbol；lots 在 SQL 层聚合）。
-- 估值：ASOF JOIN 取 as_of 当日或之前最近的收盘价（DuckDB 原生 ASOF）。
-- 该表的 unrealized_pnl 与 pandas PortfolioAnalyzer 有一条 pytest 对账测试
-- （SQL 与 pandas 两条独立路径必须算出同一个数字）。
with lots as (
    select * from {{ ref('stg_positions') }}
),

agg as (
    select
        as_of,
        symbol,
        sum(shares)                                        as shares,
        sum(shares * cost_basis) / nullif(sum(shares), 0)  as avg_cost,
        min(open_date)                                     as first_open_date
    from lots
    group by 1, 2
    having sum(shares) <> 0
),

priced as (
    select
        a.*,
        p.close  as last_close,
        p.date   as price_date
    from agg a
    asof left join {{ ref('stg_prices') }} p
        on a.symbol = p.symbol and p.date <= a.as_of
)

select
    as_of,
    symbol,
    shares,
    avg_cost,
    first_open_date,
    last_close,
    price_date,
    shares * avg_cost                                  as cost_value,
    shares * last_close                                as market_value,
    shares * (last_close - avg_cost)                   as unrealized_pnl,
    case
        when avg_cost > 0 and shares <> 0
        then (shares * (last_close - avg_cost)) / abs(shares * avg_cost)
    end                                                as unrealized_pnl_pct
from priced
