# <img src="docs/img/icon.png" width="32" alt=""> QuantAI — Personal Portfolio Analysis & Decision-Support System

**A personal quant workbench in two surfaces: a real-time trading console with a
local-LLM analyst (Streamlit — the product anyone can clone and run), backed by a
reconciled offline BI artifact (Power BI over a DuckDB/dbt star schema).**
US equities (NYSE calendar), typed and tested (748 tests), honest by design.

| Frontend — live workstation (Streamlit) | Power BI — offline analysis artifact |
|---|---|
| ![Workstation: real-position banner, 1-minute bars, indicator toggles](docs/img/frontend_workstation.png) | ![Power BI market overview page](powerbi/img/page1_market_overview.png) |
| ![Portfolio page: KPI row, holdings, PnL, net-worth curve](docs/img/frontend_portfolio.png) | ![Power BI signals heatmap page](powerbi/img/page2_signals.png) |

*Frontend screenshots use the bundled `portfolio.example.yaml` — no real holdings shown.*

→ [Quick start](#quick-start) · [Two surfaces, one warehouse](#two-surfaces-one-warehouse) · [Architecture](#architecture)

The codebase was rebuilt bottom-up into the typed, tested `quantai/` package;
the pre-rebuild tree is kept out of the repository entirely.

---

## What it does

Four jobs, nothing decorative:

1. **Analyze your real portfolio** — load actual holdings (local YAML/CSV, never
   committed), compute precise unrealized PnL, exposure/concentration, beta,
   holdings-based risk stats, and per-position technical state.
2. **Track tickers & markets** — a pure pandas/numpy indicator & stats engine with
   math definitions, boundary honesty, and causality proofs (no lookahead); a
   self-managed watchlist feeding prices, signals, and RSS news into the warehouse;
   a broker-style workstation UI (bilingual zh/en chrome, auto-refreshing intraday
   1-minute bars, VWAP, indicator toggles) with a real-position banner and an
   always-on **tactics board**: a five-factor rule engine emits per-symbol action
   cards (add/hold/trim/exit + stop references) every 30s, while a resident
   analyst — the local v3 student, or any OpenAI-compatible remote API via
   `llm.remote` — runs a background daemon thread that summarizes the board and
   persists news sentiment every 5 minutes without ever blocking the UI. A
   **hedging desk** prices option protection deterministically (Black-Scholes
   engine + Greeks + implied-vol solver, tested against textbook values):
   protective-put and covered-call cards with premium %, locked max-drawdown,
   annualized income and honest odd-lot disclosure — the LLM only narrates
   numbers the engine computed, never does option math itself.
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
   QLoRA SFT / DPO on a single RTX 4090. The **v3 student is live**: 7,308
   teacher samples spanning 50 historical dates (bull/bear/chop coverage) plus a
   dedicated **options curriculum** — premium selling, buy-vs-sell timing,
   0-7-day ("0DTE") risk discipline and hedge-plan review, every prompt fed
   real option chains and engine-computed Greeks so the student learns to cite
   and judge, never to compute. Trained overnight at seq 2560 (5.5 h,
   eval-loss 0.794; the max-seq ladder and VRAM-baseline gate live in
   `scripts/night_train_v3.py`), passed the blind-eval gate — zero repetition
   collapse across 26 cases, complete three-section structure on held-out option
   symbols never seen in training, and roughly double the base model's
   prompt-number citations on fresh symbols — and now serves as the production
   analyst adapter (v1 was rejected by the same gate and never shipped; v2
   served before v3 replaced it). On top of the batch flywheel, a scheduled
   **daily decision journal** accumulates triple-labeled samples every trading day:
   a transparent rule engine states the day's position call (citing real indicator
   values) → the teacher independently answers the same scenario (clean SFT data)
   and grades the system's call 0-10 (preference signal; low-scoring calls become
   DPO rejected samples) → realized 5/20-day forward returns are backfilled once
   the market plays out (ground truth for outcome-ranked DPO, never in prompts).

## Two surfaces, one warehouse

The two UIs are **not substitutes** — they answer different questions off the
same tested computation layer:

- **Streamlit frontend = the product surface.** Real-time and interactive:
  intraday workstation (1-minute bars, tactics board, hedging desk), resident
  AI analyst reports, paper-trading session console, portfolio & watchlist,
  parameterized YTD replay. **This is what runs when a stranger clones the
  repo** — quickstart below, only the example YAMLs needed.
- **Power BI = the offline analysis artifact.** Six reconciled pages (signals
  heatmap, rolling volatility, backtest deep-dive, return distribution, news &
  event odds, hidden QA) built on the exported star schema with a 67-check
  DAX-vs-pandas reconciliation. **It is a batch snapshot by design** — all
  partitions import local CSVs produced by `python scripts/warehouse.py
  --export`; it does not and cannot sync live quotes (DirectQuery doesn't
  support CSV; scheduled Service refresh needs a paid license). Free Desktop
  also means **no shareable online link**: the deliverable is the `.pbip`
  project in [`powerbi/`](powerbi/README.md) plus committed screenshots; to
  interact you open it locally in Power BI Desktop (Windows).

Data flow: live quotes/news → frontend directly; the same fetchers → DuckDB →
dbt star schema → CSV exports → Power BI. Unique to the frontend: real-time
anything, LLM interaction, order/paper-trading control. Unique to Power BI:
cross-filtered exploration of the full history at once, conditional-format
heatmaps, and a reconciliation page that proves the numbers.

## Quick start

Requires Python 3.11+ on Windows/macOS/Linux (Power BI artifact is
Windows-only to *view interactively*; everything else is cross-platform).

```powershell
git clone https://github.com/C0k11/quantai.git && cd quantai
python -m venv venv && venv\Scripts\activate          # or source venv/bin/activate
pip install -e .[ui,warehouse,dev]                     # add [llm,serve] for local-GPU analyst + API
copy portfolio.example.yaml portfolio.local.yaml       # fill in your holdings (never committed)
copy watchlist.example.yaml watchlist.local.yaml
```

```powershell
# 0) All tests (fastest proof everything runs)
python -m pytest

# 1) Analyze your real portfolio (copy portfolio.example.yaml -> portfolio.local.yaml first)
python scripts/analyze.py                      # text report; --json for machines

# 2) Data warehouse: ETL -> dbt star schema -> BI exports (CSV)
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

# 7) Daily decision journal (rule engine call -> teacher answer + 0-10 grade -> training asset;
#    idempotent per scenario, outcome returns backfilled as they mature)
python scripts/journal.py --dry-run --symbols SPY NVDA
python scripts/journal.py --backfill-outcomes --export-sft data/distill/journal_sft.jsonl
```

## Architecture

```mermaid
flowchart TB
    subgraph L0["Foundation"]
        CFG["config/ — typed pydantic settings, extra=forbid"]
        DAT["data/ — yfinance · RSS news · Polymarket odds · NYSE calendar"]
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
flowchart TB
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

**Stage 1 — Distillation** (cost-gated: real teacher calls require
`--run --confirm-spend`, key from env only):

```mermaid
flowchart TB
    MKT["real market data + news"] --> SCN["ScenarioBuilder — indicator / news-scoring / report tasks<br/>(prompts shared verbatim with production)"]
    SCN --> TCH["DeepSeek teacher"]
    TCH --> SFT["SFT JSONL — conversations"]
    TCH --> DPD["DPO preference pairs — teacher = chosen, hollow baseline = rejected"]
```

**Stage 2 — Student training** (local RTX 4090, `--confirm-compute` gated):

```mermaid
flowchart TB
    SFT["SFT JSONL"] --> QLR["QLoRA fine-tune — 4-bit NF4 Qwen student"]
    DPD["DPO preference pairs"] --> DPO["DPO alignment on top of SFT adapter"]
    QLR --> ADP["LoRA adapter"]
    DPO --> ADP
    ADP --> HOT["hot-reload adapter into agents / reports"]
```

**Stage 3 — Evolution flywheel** (paper-trading experience feeds the next round):

```mermaid
flowchart TB
    TRJ["paper-trading trajectories"] --> REC["evolution recorder — decisions + realized PnL"]
    REC --> DSB["DPO dataset builder — outcome-ranked preference pairs"]
    DSB --> NXT["next DPO round"]
```

> **Honest boundary:** there are NO online gradient updates — all learning is
> offline batch training; `online_gradient_step()` raises `NotImplementedError`
> by design.

## Data stack (DA/DE-grade, not a sticker)

```mermaid
flowchart TB
    SRC["yfinance / RSS news / live runtime / backtests"] --> PQ["parquet files — data/"]
    PQ --> RAW["DuckDB raw — pandas EL, idempotent + transactional, audit cols"]
    RAW --> STG["staging — dbt views, cleaning only"]
    STG --> MRT["marts — dbt tables, Kimball star schema + LLM sentiment"]
    MRT --> TST["dbt tests — 68 tests, all green"]
    MRT --> CSV["CSV exports — data/exports/"]
    MRT --> JDBC["DuckDB — direct SQL/JDBC, any BI tool"]
    CSV --> PBI["Power BI — 5 pages + hidden QA"]
```

- **DuckDB** local warehouse, three schemas: `raw` (pandas extract-load, audit
  columns, idempotent re-runs) → `staging` (dbt views, cleaning only) → `marts`
  (dbt tables, Kimball star schema).
- **Star schema**: `dim_symbol`, `dim_date` (calendar spine + NYSE trading-day
  attrs) and facts with declared grains — `fact_prices` (window-function
  daily returns, 52-week-high distance), `fact_positions` (lot aggregation +
  ASOF-join valuation), `fact_trades`, `fact_signals`, `fact_news` (RSS
  headlines at link grain **with LLM sentiment columns** — unscored stays NULL,
  never a fake neutral 0), `fact_event_odds` (**Polymarket prediction-market
  implied probabilities** via the public Gamma API — Fed decisions, macro events —
  snapshotted daily into a probability time series), `fact_backtest_results`,
  `fact_backtest_equity` (SQL drawdown).
- **dbt tests**: 68 tests (not-null/enums/relationships/grain uniqueness on
  every fact/OHLC sanity/PnL consistency/drawdown ≤ 0/warn on unpriceable
  positions) — all passing on real market data.
- **Reconciliation**: pytest runs a real `dbt build` and asserts SQL and pandas
  compute identical numbers (PnL to 1e-6, returns/drawdown to 1e-12).
- **Dashboard spec**: [powerbi/SPEC.md](powerbi/SPEC.md) defines the five-view
  semantics — star-schema joins, window rules, per-view fields and encodings.
- **Power BI (BI as code)**: [powerbi/](powerbi/README.md) implements that
  dashboard spec as a PBIP project — semantic model in plain-text TMDL
  (10 tables, 11 single-direction relationships, parameterized UTF-8 CSV
  import), report layout generated by `build_report.py`, everything diffable
  in git. The spec's view-window table calculations are translated to DAX with
  three semantics proven, not assumed: windows are **20 trading rows** (not 20
  calendar days — the calendar version silently drops ~30% of the sample and
  reads ~10% low on current data), **sample** standard deviation
  (`STDEVX.S` = pandas `ddof=1`), and **BLANK on short windows**. A hidden QA
  page reconciles the live DAX against pandas-computed expectations from the
  same CSVs: **66/67 checks PASS at 1e-6** (one skipped by design —
  `fact_trades` is empty), split into measure-arithmetic checks and
  model-wiring checks (relationship propagation + blank-member canaries) that
  provably turn red when a join breaks. Free Desktop only: no Publish-to-web,
  so the deliverable is the project + screenshots, reproducible locally.

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
- Student adapters ship only through a blind-eval gate
  (`scripts/eval_student.py`: base vs student side-by-side on held-out dates
  plus never-trained symbols, with repetition/citation/format collapse
  detectors) — the v1 adapter failed it and was never mounted.

## UI languages

The dashboard chrome (navigation, workstation, tactics board, paper-trading,
AI-analyst page) switches between 中文 and English from the sidebar; the
resident analyst answers in the selected language. Report *bodies* (daily /
intraday archives) and the portfolio/watchlist page internals are still
Chinese-first — honest boundary, on the roadmap.

## Privacy

Real holdings (`portfolio.local.yaml`, any `*.local.yaml`), `.env` secrets, and
generated data live outside version control via `.gitignore`. The repo ships
only `portfolio.example.yaml`.

## Requirements

Python ≥ 3.10. Core: numpy, pandas, pyarrow, scipy, ta, yfinance,
pandas_market_calendars, pydantic, duckdb. Extras: `[llm]` torch/transformers/
peft/trl, `[serve]` fastapi/uvicorn, `[ui]` streamlit/plotly/mplfinance,
`[warehouse]` dbt-duckdb. See [pyproject.toml](pyproject.toml).
