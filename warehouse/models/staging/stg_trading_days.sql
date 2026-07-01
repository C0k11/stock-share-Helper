-- staging：NYSE 交易日（来源 pandas_market_calendars，经 ETL 落 raw）。
select
    date,
    is_early_close
from {{ source('raw', 'trading_days') }}
