-- fact_news：新闻事实表（粒度 = link；date 由发布时间派生，可 join dim_date）。
-- published 缺失（源没给发布时间）时 date 为 NULL——诚实缺失，不用抓取时间冒充。
select
    link,
    symbol,
    title,
    summary,
    cast(published as date)  as date,
    published,
    source
from {{ ref('stg_news') }}
