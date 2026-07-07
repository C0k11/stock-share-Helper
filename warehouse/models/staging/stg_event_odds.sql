-- staging：Polymarket 事件概率快照（预测市场隐含概率，非官方预测）。
select
    as_of,
    market_id,
    condition_id,
    event_title,
    question,
    yes_price,
    volume_24h,
    end_date,
    category
from {{ source('raw', 'event_odds') }}
where yes_price between 0 and 1
