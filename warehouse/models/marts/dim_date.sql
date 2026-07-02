-- dim_date：日期维度（日历日 spine + NYSE 交易日属性）。
-- spine 覆盖 trading_days 全范围的**日历日**（含周末/假日）——BI 里做时间轴补齐
-- 与「非交易日」对比都需要完整日历；is_trading_day 区分。
with bounds as (
    -- 边界取「交易日历 ∪ 行情日期」的并集范围：日历若被窄区间重载（trading_days
    -- 是全量替换、prices 按 symbol 累积），fact_prices -> dim_date 的关系测试
    -- 不应因此破裂。
    select min(d) as lo, max(d) as hi
    from (
        select date as d from {{ ref('stg_trading_days') }}
        union all
        select date as d from {{ ref('stg_prices') }}
    )
),

spine as (
    select cast(unnest(generate_series(lo, hi, interval '1 day')) as date) as date
    from bounds
)

select
    s.date,
    extract(year from s.date)                          as year,
    extract(quarter from s.date)                       as quarter,
    extract(month from s.date)                         as month,
    strftime(s.date, '%Y-%m')                          as year_month,
    extract(isodow from s.date)                        as day_of_week,  -- 1=Mon..7=Sun
    t.date is not null                                 as is_trading_day,
    coalesce(t.is_early_close, false)                  as is_early_close
from spine s
left join {{ ref('stg_trading_days') }} t using (date)
