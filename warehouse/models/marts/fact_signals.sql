-- fact_signals：信号事实表（粒度 = symbol × date），join 次日行情验证信号命中率的
-- 分析在 BI/SQL 层做（诚实口径：signal[t] 只能对 t+1 之后的收益说话）。
select
    s.symbol,
    s.date,
    s.trend_signal,
    s.momentum_signal,
    s.ma_cross_signal,
    s.breakout_signal,
    s.composite_signal,
    s.signal_strength
from {{ ref('stg_signals') }} s
where s.composite_signal is not null   -- 去掉指标 warmup 期的空信号行
