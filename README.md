# QuantAI — Personal Portfolio Analysis & Decision-Support System

> Analyze real holdings, individual tickers, and strategy backtests with a clean,
> tested Python stack: technical-indicator + financial-statistics engine, a local
> data warehouse (DuckDB + dbt star schema) feeding Tableau, an LLM multi-agent
> layer with an offline teacher-student distillation pipeline, and a professional
> dark-theme dashboard. US equities (NYSE calendar), honest by design.

`refactor/v2` — the codebase was rebuilt bottom-up into the typed, tested
`quantai/` package (620+ tests). Legacy code stays in `src/` untouched.

---

## What it does

Three jobs, nothing decorative:

1. **Analyze your real portfolio** — load actual holdings (local YAML/CSV, never
   committed), compute precise unrealized PnL, exposure/concentration, beta,
   holdings-based risk stats, and per-position technical state.
2. **Analyze tickers** — a pure pandas/numpy indicator & stats engine with math
   definitions, boundary honesty, and causality proofs (no lookahead).
3. **Decision support** — rule + LLM multi-agent pipeline (planner → gatekeeper →
   router → experts → VLM chartist → macro governor → debate), paper-trading
   runtime, offline evolution loop (experience → DPO preference pairs → QLoRA).

## Quick start

```powershell
# 0) All tests (fastest proof everything runs)
python -m pytest

# 1) Analyze your real portfolio (copy portfolio.example.yaml -> portfolio.local.yaml first)
python scripts/analyze.py                      # text report; --json for machines

# 2) Data warehouse: ETL -> dbt star schema -> Tableau-ready exports
python scripts/warehouse.py --full --portfolio portfolio.local.yaml

# 3) Dashboard (portfolio page works standalone)
python -m streamlit run quantai/ui/streamlit_app.py

# 4) Paper trading (offline deterministic demo)
python scripts/live.py --source simulated --tickers NVDA TSLA --interval 0.1 --duration 8 --seed 1

# 5) Distillation dry-run (mock teacher, zero API spend)
python scripts/distill.py --dry-run --symbols SPY NVDA
```

## Architecture

```
quantai/
├── config/      typed pydantic config (extra=forbid), single source of defaults
├── data/        yfinance prices, parquet storage, NYSE calendar (pandas_market_calendars)
├── features/    causal model features (truncation-invariance tested)
├── analysis/    indicator + stats engine: MACD/RSI/Bollinger/ATR/stochastic/pullback,
│                vol, VaR/CVaR, drawdown, Sharpe/Sortino, beta — pure functions,
│                cross-validated against `ta`, no-lookahead proofs
├── signals/     rule signal generator
├── risk/        position sizing, vol targeting, drawdown control, risk gate
├── backtest/    vectorized engine — fills at NEXT OPEN (lookahead fixed & documented)
├── execution/   costs, trading rules, PaperBroker (margin, liquidation, fills)
├── agents/      LLM multi-agent brain (deterministic fallbacks, lazy GPU deps)
├── llm/         prompts, local inference (4/8bit, MoE adapters), QLoRA, DPO
├── live/        event-driven runtime: feeds → engine → strategy adapter → broker
├── evolution/   data flywheel: trajectory recorder → preference pairs → offline DPO
├── portfolio/   real-holdings loader + analyzer (PnL, exposure, honest risk labels)
├── warehouse/   pandas EL into DuckDB `raw`, idempotent loaders
├── distill/     DeepSeek teacher → SFT/DPO JSONL for the local student model
└── ui/          plotly chart builders (pure) + Streamlit dashboard

warehouse/       dbt project: raw → staging (views) → marts (star schema)
tableau/         DASHBOARD_SPEC.md — connection guide + 5 dashboard view specs
```

## Data stack (DA/DE-grade, not a sticker)

- **DuckDB** local warehouse, three schemas: `raw` (pandas extract-load, audit
  columns, idempotent re-runs) → `staging` (dbt views, cleaning only) → `marts`
  (dbt tables, Kimball star schema).
- **Star schema**: `dim_symbol`, `dim_date` (calendar spine + NYSE trading-day
  attrs) and facts with declared grains — `fact_prices` (window-function
  daily returns, 52-week-high distance), `fact_positions` (lot aggregation +
  ASOF-join valuation), `fact_trades`, `fact_signals`, `fact_backtest_results`,
  `fact_backtest_equity` (SQL drawdown).
- **dbt tests**: 58 assertions (not-null/enums/relationships/grain uniqueness/
  OHLC sanity/PnL consistency/drawdown ≤ 0) — all passing on real market data.
- **Reconciliation**: pytest runs a real `dbt build` and asserts SQL and pandas
  compute identical numbers (PnL to 1e-6, returns/drawdown to 1e-12).
- **Tableau**: connect via exported CSVs or DuckDB JDBC; build per
  [tableau/DASHBOARD_SPEC.md](tableau/DASHBOARD_SPEC.md).

## Integrity notes (deliberate, tested)

- Backtest fills at **next open**: the legacy same-bar-close fill inflated returns
  ~3.5×; the fix is the default and the old mode exists only for comparison.
  Honest benchmark: the demo strategy **underperforms** buy-and-hold SPY on CAGR
  (9.7% vs 14.2%) with shallower drawdown (-21% vs -34%) — sold as engineering +
  risk control, not alpha.
- Indicators propagate NaN through data holes instead of carrying stale values;
  MACD cross suppresses seed-artifact warmup signals; every boundary convention
  is documented at the function and pinned by tests.
- No fabricated UI panels; missing data renders as an explicit empty state.
- Online gradient updates are **not implemented** — the "evolution" loop is
  honest offline DPO, and the placeholder raises if called.
- Real API spend (DeepSeek distillation) requires an explicit
  `--run --confirm-spend` and an env-var key; CI and tests are mock-only.

## Privacy

Real holdings (`portfolio.local.yaml`, any `*.local.yaml`), `.env` secrets, and
generated data live outside version control via `.gitignore`. The repo ships
only `portfolio.example.yaml`.

## Requirements

Python ≥ 3.10. Core: numpy, pandas, pyarrow, scipy, ta, yfinance,
pandas_market_calendars, pydantic, duckdb. Extras: `[llm]` torch/transformers/
peft/trl, `[serve]` fastapi/uvicorn, `[ui]` streamlit/plotly/mplfinance,
`[warehouse]` dbt-duckdb. See [pyproject.toml](pyproject.toml).
