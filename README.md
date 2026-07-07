# QuantAI — Personal Portfolio Analysis & Decision-Support System

> Analyze real holdings, individual tickers, and strategy backtests with a clean,
> tested Python stack: technical-indicator + financial-statistics engine, a local
> data warehouse (DuckDB + dbt star schema) feeding Tableau, an LLM multi-agent
> layer with an offline teacher-student distillation pipeline, and a professional
> dark-theme dashboard. US equities (NYSE calendar), honest by design.

The codebase was rebuilt bottom-up into the typed, tested `quantai/` package
(630+ tests). Legacy code stays in `src/` untouched.

---

## What it does

Four jobs, nothing decorative:

1. **Analyze your real portfolio** — load actual holdings (local YAML/CSV, never
   committed), compute precise unrealized PnL, exposure/concentration, beta,
   holdings-based risk stats, and per-position technical state.
2. **Track tickers & markets** — a pure pandas/numpy indicator & stats engine with
   math definitions, boundary honesty, and causality proofs (no lookahead); a
   self-managed watchlist feeding prices, signals, and RSS news into the warehouse;
   a broker-style workstation UI (intraday 1-minute bars, VWAP, indicator toggles).
3. **AI analyst** — scheduled daily and intraday reports: session statistics with
   honest OHLCV-level selling-pressure proxies (down-bar volume share, price vs
   session VWAP, volume vs 20-day average), LLM commentary from a locally served
   Qwen model grounded strictly in system data, and batch **news-sentiment
   quantification** persisted to the warehouse for BI timelines.
4. **Decision support & learning loop** — rule + LLM multi-agent pipeline (planner →
   gatekeeper → router → experts → VLM chartist → macro governor → debate),
   paper-trading runtime, and an executed **teacher–student distillation flywheel**:
   DeepSeek-generated scenario batches (indicator analysis, news scoring,
   multi-symbol reports — prompts shared verbatim with production) → local
   QLoRA SFT / DPO on a single RTX 4090.

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

# 4) AI analyst reports (data brief is seconds; --llm adds local-GPU commentary + news scoring)
python scripts/report.py --llm              # daily
python scripts/report.py --intraday --llm   # intraday session stats + selling-pressure proxies

# 5) Paper trading (offline deterministic demo)
python scripts/live.py --source simulated --tickers NVDA TSLA --interval 0.1 --duration 8 --seed 1

# 6) Distillation flywheel (mock dry-run is free; real teacher runs are cost-gated)
python scripts/distill.py --dry-run --symbols SPY NVDA
python scripts/train.py --sft data/distill/<batch>.jsonl --confirm-compute
```

## Architecture

```mermaid
flowchart TB
    subgraph L0["Foundation"]
        CFG["config/ — typed pydantic settings, extra=forbid"]
        DAT["data/ — yfinance · RSS news · parquet · NYSE calendar"]
    end
    subgraph L1["Pure computation — causal, no lookahead"]
        FEA["features/ — causal model features"]
        ANA["analysis/ — indicators + financial stats"]
        SIG["signals/ — rule signal generator"]
        RSK["risk/ — sizing · vol targeting · drawdown gate"]
    end
    subgraph L2["Simulation & execution"]
        BTE["backtest/ — vectorized, fills at NEXT OPEN"]
        EXE["execution/ — costs · rules · PaperBroker"]
    end
    subgraph L3["Decision & runtime"]
        AGT["agents/ — LLM multi-agent brain"]
        LLM["llm/ — prompts · local inference · QLoRA · DPO"]
        LIV["live/ — event-driven paper trading"]
    end
    subgraph L4["Learning loop — all offline"]
        DIS["distill/ — DeepSeek teacher data"]
        EVO["evolution/ — trajectory recorder · DPO flywheel"]
    end
    subgraph L5["Products"]
        POR["portfolio/ — real holdings PnL & risk"]
        WHS["warehouse/ — DuckDB EL + dbt star schema"]
        API["api/ — FastAPI, real endpoints only"]
        UIX["ui/ — plotly builders + Streamlit"]
    end
    DAT --> FEA --> SIG --> RSK --> BTE
    DAT --> ANA --> POR
    RSK --> LIV
    LIV --> EXE
    LIV --> AGT
    AGT --> LLM
    DAT --> DIS
    ANA --> DIS
    DIS --> LLM
    LIV --> EVO
    EVO --> LLM
    POR --> WHS
    BTE --> WHS
    LIV --> WHS
    LIV --> API
    EVO --> API
    API --> UIX
```

### Multi-agent decision chain

```mermaid
flowchart LR
    IN["market snapshot"] --> PLN["Planner"]
    PLN --> GTK["Gatekeeper — deterministic approval"]
    GTK -->|approved| RTR["Router"]
    GTK -->|rejected| HLD["hold, no trade"]
    RTR --> EXP["Experts — trend / value / risk"]
    EXP --> VLM["VLM Chartist — chart image read, env-gated"]
    VLM --> MAC["Macro Governor — real VIX / TNX gate"]
    MAC --> DEB["Debate & consensus"]
    DEB --> DEC["decision → PaperBroker"]
```

Every LLM stage has a deterministic fallback, so the chain runs end-to-end
offline (CPU-only, no keys) and stays testable.

### Teacher-student distillation & evolution flywheel (offline by design)

```mermaid
flowchart LR
    subgraph DISTILL["Distillation — cost-gated, --confirm-spend"]
        MKT["real market data"] --> SCN["ScenarioBuilder — 4 task templates"]
        SCN --> TCH["DeepSeek teacher — key from env only"]
        TCH --> SFT["SFT JSONL"]
        TCH --> DPD["DPO preference pairs"]
    end
    subgraph TRAIN["Student training — offline"]
        SFT --> QLR["QLoRA fine-tune, local student model"]
        DPD --> DPO["DPO training"]
        QLR --> ADP["LoRA adapter"]
        DPO --> ADP
    end
    subgraph FLYWHEEL["Evolution flywheel — offline"]
        TRJ["paper-trading trajectories"] --> REC["evolution/ recorder"]
        REC --> DSB["DPO dataset builder"]
        DSB --> DPO
        ADP --> HOT["hot-reload adapter into agents"]
        HOT -.-> TRJ
    end
    BND["Honest boundary: NO online gradient updates —<br/>online_gradient_step raises NotImplementedError"]
    style BND fill:#7f1d1d,color:#fff,stroke:#ef5350
```

## Data stack (DA/DE-grade, not a sticker)

```mermaid
flowchart LR
    SRC["yfinance / live runtime / backtests"] --> PQ["parquet files — data/"]
    PQ --> RAW["DuckDB raw — pandas EL, idempotent, audit cols"]
    RAW --> STG["staging — dbt views, cleaning only"]
    STG --> MRT["marts — dbt tables, Kimball star schema"]
    MRT --> TST["dbt tests — 68 assertions, all green"]
    MRT --> CSV["CSV exports — tableau/exports/"]
    MRT --> JDBC["DuckDB JDBC"]
    CSV --> TAB["Tableau dashboards — 5 views"]
    JDBC --> TAB
```

- **DuckDB** local warehouse, three schemas: `raw` (pandas extract-load, audit
  columns, idempotent re-runs) → `staging` (dbt views, cleaning only) → `marts`
  (dbt tables, Kimball star schema).
- **Star schema**: `dim_symbol`, `dim_date` (calendar spine + NYSE trading-day
  attrs) and facts with declared grains — `fact_prices` (window-function
  daily returns, 52-week-high distance), `fact_positions` (lot aggregation +
  ASOF-join valuation), `fact_trades`, `fact_signals`, `fact_news` (RSS
  headlines at link grain **with LLM sentiment columns** — unscored stays NULL,
  never a fake neutral 0), `fact_backtest_results`, `fact_backtest_equity`
  (SQL drawdown).
- **dbt tests**: 80 assertions (not-null/enums/relationships/grain uniqueness on
  every fact/OHLC sanity/PnL consistency/drawdown ≤ 0/warn on unpriceable
  positions) — all passing on real market data.
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
