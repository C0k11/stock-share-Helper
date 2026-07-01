-- staging：回测净值曲线（日粒度）。
select
    run_id,
    date,
    equity,
    daily_return
from {{ source('raw', 'backtest_equity') }}
