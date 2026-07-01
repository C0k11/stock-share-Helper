-- staging：纸面交易成交流水（run_id 区分不同 live 会话）。
select
    run_id,
    seq,
    upper(symbol)  as symbol,
    action,
    price,
    shares,
    trace_id
from {{ source('raw', 'trades') }}
where symbol <> ''
