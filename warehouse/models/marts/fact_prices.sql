-- fact_prices：日行情事实表（粒度 = symbol × date）。
-- 派生列用窗口函数在 SQL 层算（daily_return / 52 周高点距离），BI 不用再拉明细回 pandas。
select
    p.symbol,
    p.date,
    p.open,
    p.high,
    p.low,
    p.close,
    p.volume,
    p.close / lag(p.close) over w - 1                            as daily_return,
    p.close / max(p.high) over (
        partition by p.symbol
        order by p.date
        rows between 251 preceding and current row
    ) - 1                                                        as pct_from_52w_high
from {{ ref('stg_prices') }} p
window w as (partition by p.symbol order by p.date)
