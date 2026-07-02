-- 粒度断言：fact_backtest_equity 主键 (run_id, date) 不得重复
--（重复行会让 running-max 回撤窗口迭代脏数据、BI 求和翻倍）。
select run_id, date, count(*) as n
from {{ ref('fact_backtest_equity') }}
group by 1, 2
having count(*) > 1
