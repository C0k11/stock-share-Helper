-- fact_event_odds：事件概率事实表（粒度 = as_of × market_id）。
-- prob_change_1d 用窗口函数派生——BI 直接画"事件概率演变线 + 单日突变"。
select
    as_of,
    market_id,
    event_title,
    question,
    category,
    yes_price,
    yes_price - lag(yes_price) over (
        partition by market_id order by as_of
    )                                          as prob_change_1d,
    volume_24h,
    cast(end_date as date)                     as end_date
from {{ ref('stg_event_odds') }}
