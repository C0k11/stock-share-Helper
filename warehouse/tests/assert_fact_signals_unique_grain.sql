-- 粒度断言：fact_signals 主键 (symbol, date) 不得重复。
select symbol, date, count(*) as n
from {{ ref('fact_signals') }}
group by 1, 2
having count(*) > 1
