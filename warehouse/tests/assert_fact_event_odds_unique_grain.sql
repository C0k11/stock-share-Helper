-- 粒度断言：fact_event_odds 主键 (as_of, market_id) 不得重复。
select as_of, market_id, count(*) as n
from {{ ref('fact_event_odds') }}
group by 1, 2
having count(*) > 1
