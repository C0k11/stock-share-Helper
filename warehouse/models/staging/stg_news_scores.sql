-- staging：新闻情绪分（LLM 标题级量化；raw 已按 link 幂等覆盖）。
select
    link,
    sentiment,
    label,
    model,
    scored_at
from {{ source('raw', 'news_scores') }}
where sentiment between -1 and 1
