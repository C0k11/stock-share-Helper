
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  -- 一致性断言：unrealized_pnl 必须等于 market_value - cost_value（列间自洽）。
select *
from "test"."marts"."fact_positions"
where last_close is not null
  and abs(unrealized_pnl - (market_value - cost_value)) > 1e-6
  
  
      
    ) dbt_internal_test