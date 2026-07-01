-- fact_backtest_results：回测结果事实表（粒度 = run_id）。
select
    run_id,
    strategy,
    symbol,
    fill_timing,
    total_return,
    cagr,
    annual_volatility,
    sharpe,
    max_drawdown,
    win_rate,
    total_turnover,
    start_date,
    end_date,
    trading_days
from {{ ref('stg_backtest_runs') }}
