-- dim_symbol：标的维度（自然键 = symbol）。
-- 成员 = 出现在任何 fact 源里的标的并集；价格覆盖范围与「当前是否持有」为属性。
with universe as (
    select distinct symbol from {{ ref('stg_prices') }}
    union
    select distinct symbol from {{ ref('stg_positions') }}
    union
    select distinct symbol from {{ ref('stg_trades') }}
    union
    select distinct symbol from {{ ref('stg_signals') }}
),

price_stats as (
    select
        symbol,
        min(date)  as first_price_date,
        max(date)  as last_price_date,
        count(*)   as price_bars
    from {{ ref('stg_prices') }}
    group by 1
),

latest_positions as (
    -- 最新快照日仍有净持仓的标的
    select p.symbol
    from {{ ref('stg_positions') }} p
    where p.as_of = (select max(as_of) from {{ ref('stg_positions') }})
    group by p.symbol
    having sum(p.shares) <> 0
)

select
    u.symbol,
    ps.first_price_date,
    ps.last_price_date,
    coalesce(ps.price_bars, 0)          as price_bars,
    lp.symbol is not null               as is_currently_held
from universe u
left join price_stats ps using (symbol)
left join latest_positions lp using (symbol)
