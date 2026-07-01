-- 数据质量断言：high >= low，且 close 落在 [low, high]（允许 1e-9 浮点余量）。
select *
from {{ ref('fact_prices') }}
where high is not null and low is not null
  and (
       high < low - 1e-9
    or close > high + 1e-9
    or close < low - 1e-9
  )
