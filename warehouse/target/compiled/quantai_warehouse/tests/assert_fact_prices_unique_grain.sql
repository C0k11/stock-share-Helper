-- 粒度断言：fact_prices 主键 (symbol, date) 不得重复（内置 unique 只支持单列，组合键用 singular test）。
select symbol, date, count(*) as n
from "test"."marts"."fact_prices"
group by 1, 2
having count(*) > 1