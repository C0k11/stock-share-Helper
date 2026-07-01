-- staging：1:1 清洗 raw.prices —— 统一大小写、去掉无效行（close 缺失/非正）。
-- 不做任何聚合/派生（那是 marts 的事），保持每层职责单一、可审计。
select
    upper(symbol)   as symbol,
    date,
    open,
    high,
    low,
    close,
    volume
from {{ source('raw', 'prices') }}
where close is not null
  and close > 0
