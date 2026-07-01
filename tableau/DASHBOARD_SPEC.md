# QuantAI · Tableau 仪表盘搭建规格（连接说明 + 视图规格）

> 数据侧已交付：DuckDB 星型仓库（`data/warehouse/quantai.duckdb` 的 `marts` schema）
> + 可直连的 CSV 导出（`tableau/exports/*.csv`，由 `python scripts/warehouse.py --export` 再生）。
> 本文档是**在 Tableau Desktop 里照做**的搭建规格：连接、数据模型、5 张仪表盘视图。

---

## 1. 连接方式（二选一）

### A. CSV 导出（最简单，推荐起步）
1. 先刷新数据：`python scripts/warehouse.py --full --portfolio portfolio.local.yaml`
2. Tableau Desktop → **Connect → Text file** → 选 `tableau/exports/fact_prices.csv`
3. 把其余 7 张 CSV 拖进画布建立关系（见 §2 数据模型）。
4. 发布/存档用 **Extract**（.hyper）以获得列存性能。

### B. DuckDB 直连（免导出，数据零拷贝）
- Tableau 无原生 DuckDB 连接器，用 **JDBC**：
  1. 下载 DuckDB JDBC driver（duckdb.org → Docs → Clients → Java），放入
     `C:\Program Files\Tableau\Drivers`。
  2. Connect → **Other Databases (JDBC)** → URL 填
     `jdbc:duckdb:D:\Project\Stock\data\warehouse\quantai.duckdb`，Dialect 选 PostgreSQL。
  3. Schema 选 `marts`。
- 注意 DuckDB 单写者锁：Tableau 连着时别同时跑 `--load/--dbt`。

## 2. 数据模型（Tableau Relationships，照星型连）

```
dim_date (date) 1 ──── * fact_prices (date, symbol) * ──── 1 dim_symbol (symbol)
                  └──── * fact_signals (date, symbol) * ───┘
                  └──── * fact_backtest_equity (date; run_id)
fact_positions (as_of, symbol) * ──── 1 dim_symbol
fact_trades (symbol) * ──── 1 dim_symbol
fact_backtest_results (run_id) 1 ──── * fact_backtest_equity (run_id)
```

- 关系全用 Tableau 的 **Relationships**（逻辑层），不要物理 join（保各 fact 粒度）。
- `dim_date.is_trading_day` 用于滤掉非交易日；`fact_prices.daily_return`、
  `pct_from_52w_high`、`fact_backtest_equity.drawdown` 都已在 SQL 层算好，直接用。

## 3. 仪表盘规格（5 张视图）

### 3.1 组合总览（Portfolio Overview）
- **KPI 卡**（fact_positions 最新 as_of）：总市值 Σmarket_value、总成本 Σcost_value、
  未实现盈亏 Σunrealized_pnl（红绿色编码）、盈亏% = Σpnl/|Σcost|。
- **持仓权重条形图**：symbol × market_value，按权重降序；颜色 = unrealized_pnl 正负。
- **集中度**：market_value 树图（treemap）。

### 3.2 盈亏视图（PnL）
- **每标的盈亏瀑布/条形**：symbol × unrealized_pnl，标注 unrealized_pnl_pct。
- **成本 vs 现价**：dumbbell（avg_cost → last_close 双点连线）每 symbol 一行。
- 参考线：0 线；tooltip 带 first_open_date / price_date。

### 3.3 信号与回测 vs 基准（Signals & Backtest）
- **净值曲线**：fact_backtest_equity.equity 折线（run_id 颜色分组），叠加基准
  （fact_prices 里基准 symbol 的 close 归一化到相同起点：
  `close / {FIXED LOOKBACK: min(close)}` 或用表计算 first()）。
- **回撤带**：drawdown 面积图（同轴或下方小图）。
- **信号叠加**：fact_signals.composite_signal 作副轴色带 / 或 signal_strength
  离散色点打在价格线上（join date+symbol）。
- **回测指标表**：fact_backtest_results 全列文本表（sharpe/cagr/mdd/win_rate/fill_timing）。

### 3.4 风险与暴露（Risk & Exposure）
- **52 周高点距离**：fact_prices.pct_from_52w_high 最新值 × symbol 条形（越负越深红）。
- **日收益分布**：fact_prices.daily_return 直方图（symbol 筛选器）。
- **滚动波动**：daily_return 的 N 日移动标准差 ×√252（Tableau 表计算
  WINDOW_STDEV(SUM([daily_return]), -19, 0) * SQRT(252)，写明口径与仓库层一致）。

### 3.5 指标视图（Technical）
- **价格 + 均线**：close 折线 + 20/50/200 日移动平均（表计算 WINDOW_AVG）。
- **信号热力**：date × symbol 网格，颜色 = composite_signal（发散色板，0 居中）。
- 筛选器：symbol（全局）、date range（全局）、is_trading_day=True。

## 4. 主题与命名
- 暗色主题（背景 #1B1B1F / 强调 #4C8BF5 / 涨 #26A69A 跌 #EF5350——与 app UI 一致）。
- 工作簿命名 `QuantAI Portfolio Analytics`；每张 sheet 名用上面小节标题。

## 5. 刷新流程（数据更新后）
```powershell
python scripts/warehouse.py --full --portfolio portfolio.local.yaml   # ETL + dbt + export
# Tableau: Data → Refresh（CSV/Extract）或重连 JDBC
```
