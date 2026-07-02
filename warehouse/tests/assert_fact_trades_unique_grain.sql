-- 粒度断言：fact_trades 主键 (run_id, seq) 不得重复。
select run_id, seq, count(*) as n
from {{ ref('fact_trades') }}
group by 1, 2
having count(*) > 1
