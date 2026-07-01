-- fact_trades：成交事实表（粒度 = run_id × seq）。
select
    run_id,
    seq,
    symbol,
    action,
    price,
    shares,
    price * shares                       as notional,
    case action when 'BUY' then 1 when 'SELL' then -1 end * shares as signed_shares,
    trace_id
from {{ ref('stg_trades') }}
