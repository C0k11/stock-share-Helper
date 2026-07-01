-- staging：回测运行（metrics 1 行/次；口径见 quantai/backtest/metrics.py）。
select
    run_id,
    strategy,
    upper(symbol)      as symbol,
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
from {{ source('raw', 'backtest_runs') }}
