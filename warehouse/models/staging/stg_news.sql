-- staging：新闻头条（raw.news 已按 link 去重；这里只做清洗与大小写归一）。
select
    upper(symbol)          as symbol,   -- 通用源的条目 symbol 为 NULL
    title,
    summary,
    link,
    published,
    source
from {{ source('raw', 'news') }}
where title <> '' and link <> ''
