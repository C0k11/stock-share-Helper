# QuantAI Portfolio Analytics — Power BI

The BI layer over the QuantAI DuckDB warehouse, built to the same specification
as [`../tableau/DASHBOARD_SPEC.md`](../tableau/DASHBOARD_SPEC.md): five views
over a star schema of two dimensions and eight fact tables.

The reason this exists alongside a Tableau spec is that the two tools disagree
about what a "window" is, and that disagreement is worth showing rather than
hiding. Tableau's table calculations address rows by their position in the
rendered view. DAX has no equivalent — it only sees the data model — so every
moving average and rolling standard deviation has to be rebuilt from the dates
themselves. Translating `WINDOW_STDEV(..., -19, 0)` as `DATESINPERIOD(..., -20,
DAY)` looks right and is wrong: 20 calendar days is about 14 trading days, so
the sample silently shrinks by roughly 30%. See the comment block in
[`measures.dax`](measures.dax).

## Data

`python scripts/warehouse.py --full --portfolio portfolio.local.yaml` regenerates
`tableau/exports/*.csv`. Those exports are gitignored, so **import (not
DirectQuery) is deliberate**: the `.pbix` embeds its data and opens on a machine
that has never run the pipeline.

Only the tables the five views actually use are imported. `fact_news` (5,290
rows) and `fact_event_odds` (3,079 rows) are left out — nothing in the spec
consumes them, and together they are most of the export's bulk.

| Table | Rows | Role |
|---|---|---|
| `dim_date` | 730 | date dimension, `is_trading_day` filters non-trading days |
| `dim_symbol` | 34 | symbol dimension |
| `fact_prices` | 15,917 | OHLCV + precomputed `daily_return`, `pct_from_52w_high` |
| `fact_signals` | 15,917 | four component signals + `composite_signal` |
| `fact_positions` | 18 | one snapshot, `as_of` 2026-07-11 |
| `fact_backtest_equity` | 500 | equity curve + drawdown |
| `fact_backtest_results` | 1 | run-level metrics |

**What the data does not contain** — worth knowing before building a visual
around it:

- `fact_trades` is **empty** (0 rows). Anything in the spec that depends on
  individual fills has no data behind it.
- There is exactly **one backtest run** (`const60_SPY`). The spec's "colour the
  equity curve by `run_id`" renders a single line today. The measures are
  written to handle more runs, but do not build a legend that implies several.
- `fact_prices[daily_return]` and `[pct_from_52w_high]` are **blank on each
  symbol's first row** — there is no prior close to difference against. This is
  correct, and Power Query must not be allowed to turn those blanks into zeros.

## Build order

### 1. Load

Get data → Text/CSV → select all seven files above from `tableau/exports/`.
In the Power Query editor, before loading:

- `dim_date[is_trading_day]` and `[is_early_close]` arrive as the strings
  `true` / `false`. Change type to **True/False** explicitly; leaving them as
  text makes every filter a string comparison.
- `dim_symbol[is_currently_held]` likewise.
- `fact_backtest_results[start_date]` / `[end_date]` carry a UTC offset
  (`2024-07-26 00:00:00-04:00`). Set them to **Date/Time/Timezone**, or to
  **Date** if you only need the day — silently coercing to Date/Time drops the
  offset and shifts the boundary by a day.
- Leave the blanks in `daily_return` and `pct_from_52w_high` alone. Do **not**
  use "Replace errors → 0".

### 2. Model

Relationships, all single-direction, one-to-many from the dimension:

```
dim_date[date]        1 ─→ *  fact_prices[date]
dim_date[date]        1 ─→ *  fact_signals[date]
dim_date[date]        1 ─→ *  fact_backtest_equity[date]
dim_symbol[symbol]    1 ─→ *  fact_prices[symbol]
dim_symbol[symbol]    1 ─→ *  fact_signals[symbol]
dim_symbol[symbol]    1 ─→ *  fact_positions[symbol]
fact_backtest_results[run_id]  1 ─→ *  fact_backtest_equity[run_id]
```

Two decisions worth stating:

- **`fact_positions` connects only to `dim_symbol`, not to `dim_date`.** Its
  `as_of` is a snapshot stamp, not a transaction date. Wiring it to the date
  dimension would let a date slicer blank out the entire portfolio page, which
  is confusing rather than useful.
- **Keep cross-filter direction Single.** Bidirectional filtering between
  `dim_symbol` and two fact tables at the same grain creates ambiguity that
  Power BI resolves in a way that is hard to predict from the visual.

Mark `dim_date` as the date table (`date` column) so time intelligence works.

### 3. Measures

Paste everything in [`measures.dax`](measures.dax). Suggested home tables are
noted per section.

### 4. Pages

Theme: background `#1B1B1F`, accent `#4C8BF5`, up `#26A69A`, down `#EF5350` —
same palette as the QuantAI app UI, so screenshots of the two sit together.

**Portfolio Overview** — KPI cards (`Total Market Value`, `Total Cost`,
`Unrealized PnL` with conditional colour, `Unrealized PnL %`), a bar chart of
`market_value` by symbol sorted descending with bar colour driven by the sign of
`Unrealized PnL`, and a treemap of `market_value` for concentration.

**PnL** — bar of `unrealized_pnl` by symbol with `unrealized_pnl_pct` as the
label and a reference line at zero; a dumbbell (`avg_cost` → `last_close`) per
symbol; tooltips carrying `first_open_date` and `price_date`.

**Signals & Backtest** — line chart of `fact_backtest_equity[equity]` with
`Benchmark Rebased` on the same axis; `drawdown` as an area chart below sharing
the date axis; a table of `fact_backtest_results` (sharpe / cagr /
max_drawdown / win_rate / fill_timing); `Composite Signal` as a coloured band.

**Risk & Exposure** — `Pct From 52w High` by symbol, diverging colour scale;
histogram of `daily_return` with a symbol slicer; `Rolling Volatility 20D` as a
line. Put a text box next to the volatility chart stating the window is 20
trading days annualised by √252 — the number is meaningless without it.

**Technical** — `Close Price` with `MA 20` / `MA 50` / `MA 200` overlaid
(MA 200 is blank for the first ~200 bars of each symbol, by design); a
date × symbol matrix heat-mapped on `Composite Signal` with a diverging scale
centred at zero. Global slicers: symbol, date range, `is_trading_day = True`.

## Verifying the windowed measures

Before trusting any chart built on them, check one symbol against pandas:

```powershell
python -c "import pandas as pd,math; d=pd.read_csv('tableau/exports/fact_prices.csv',parse_dates=['date']); s=d[d.symbol=='SPY'].sort_values('date').tail(20); print(s.daily_return.std(ddof=1)*math.sqrt(252))"
```

On the current export (SPY, last date 2026-07-24) the two readings are:

| Window definition | Sample | Annualised vol |
|---|---|---|
| 20 **trading** days — what the Tableau spec means | 20 | **0.116622** |
| 20 **calendar** days — what `DATESINPERIOD(-20, DAY)` gives | 15 | 0.106998 |

Those 20 trading days span **28 calendar days**, so the calendar-day window
drops a quarter of the sample and reads **8.3% low**. Both numbers are entirely
plausible volatilities for SPY. Nothing about the chart would look wrong. That is
the whole point: this class of error is invisible without a reference value,
which is why one is checked in here.

`Rolling Volatility 20D` must return 0.116622 on SPY's last date. Reference
values for the moving averages on the same series: `MA 20` = 746.153,
`MA 50` = 744.1101, `MA 200` = 695.4285.
