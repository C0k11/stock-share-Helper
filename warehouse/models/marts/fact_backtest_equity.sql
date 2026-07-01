-- fact_backtest_equity：回测净值曲线（粒度 = run_id × date；Tableau 画净值/回撤曲线）。
-- 回撤在 SQL 层用累计最大值窗口算——与 pandas drawdown_curve 同口径。
select
    run_id,
    date,
    equity,
    daily_return,
    equity / max(equity) over (
        partition by run_id order by date
        rows between unbounded preceding and current row
    ) - 1                                            as drawdown
from {{ ref('stg_backtest_equity') }}
