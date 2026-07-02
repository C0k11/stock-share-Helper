-- 粒度断言：fact_positions 主键 (as_of, symbol) 不得重复
--（当前 GROUP BY 构造天然唯一；此断言防未来重构回归）。
select as_of, symbol, count(*) as n
from {{ ref('fact_positions') }}
group by 1, 2
having count(*) > 1
