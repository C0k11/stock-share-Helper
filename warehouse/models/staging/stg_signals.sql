-- staging：策略信号（SignalGenerator 输出；warmup 期 NaN 行留给 marts 决定取舍）。
select
    upper(symbol)      as symbol,
    date,
    trend_signal,
    momentum_signal,
    ma_cross_signal,
    breakout_signal,
    composite_signal,
    signal_strength
from {{ source('raw', 'signals') }}
