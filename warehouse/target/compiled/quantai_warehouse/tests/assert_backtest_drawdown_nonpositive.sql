-- 口径断言：回撤按定义恒 <= 0。
select *
from "test"."marts"."fact_backtest_equity"
where drawdown > 1e-12