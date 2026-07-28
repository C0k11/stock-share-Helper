# 前端职责收敛盘点（2026-07-28）

> 决策口径：**前端 = 实时 + 可交互 + 对外可见；Power BI = 深度分析的离线 artifact**。
> Power BI 免费版无可分享链接（`.pbix`/PBIP 只在本机跑），且只吃批处理 CSV 导出 ——
> 因此实时链路与唯一对外可点的组合视图必须留在前端。
> "冻结" = 代码保留、路由保留、能跑就不动；不加功能、不修非阻塞小问题、不写新测试。**不等于删除。**

## 0. 先说与前提不符的事实（盘点第一发现）

开工决策表第三行假设前端存在"信号热力、滚动波动、回测深挖、日收益分布、事件概率明细"
等待 Power BI 接管的页面。**实测：前端从未实现过其中四项** —— 信号热力矩阵、滚动波动
曲线、日收益分布直方图、事件概率页在 `quantai/ui/` 里没有任何对应代码；它们只存在于
`tableau/DASHBOARD_SPEC.md` 规格 → 由 Power BI 首次实现（P2/P4/P5）。唯一沾边的是
模拟盘·YTD 回放（对应"回测深挖"，见 §2.5a）。

结论：**没有"删除重复分析页"的刚需**。本次收敛的实际可执行动作缩水为：
①职责分工文档化（README/架构说明）；②一个真正的入口修正候选（§4）；
③把冻结边界写死。清单如下，等确认后才动。

## 1. 分工总表（决策落地）

| 展示面 | 归属 | 状态 |
|---|---|---|
| 行情工作台（分钟级 K 线/指标/作战台/对冲台/AI 驻留分析） | 前端 | **主投入** |
| 实盘会话（纸面交易 API 控制台） | 前端 | **主投入** |
| AI 分析（LLM 日报/盘中快报生成与浏览） | 前端 | **主投入** |
| 组合分析（持仓/盈亏/净值曲线） | 前端 | **保留，冻结** |
| 自选股（快照表 + universe 管理） | 前端 | **保留，冻结**（建议） |
| 模拟盘·YTD 回放（参数化规则回放） | 前端 | **保留，冻结**（建议，理由见 §2.5a） |
| 信号热力 / 滚动波动 / 日收益分布 / 事件概率 / 固定 run 回测深挖 | Power BI P2/P4/P5/P3 | 前端无对应物，无需动作 |

## 2. 页面级清单（`quantai/ui/streamlit_app.py` 五路由）

### 2.1 行情工作台 `workstation` — **主投入**

- **展示**：分钟/日线 K 线 + 指标（MA/BB/VWAP/RSI/MACD）、实时持仓横幅、作战台
  （tactician 规则操作卡 + 告警）、AI 驻留分析（CockpitDaemon 后台推理）、对冲台
  （期权链 + BS 保护性 put / covered call 计划）、Market details、持仓/新闻页签。
- **数据源**：yfinance 直拉（盘中 25s / 日线 900s TTL）、`yf.Ticker.info`（600s）、
  期权链 `OptionChainFetcher`（300s）、`portfolio.local.yaml`、`NewsFetcher`（900s）。
- **实时性**：**实时**（fragment 自刷 15–120s 分层；持仓横幅 30s）。
- **Power BI 重复**：无。PBI 无盘中链路；P1 的 KPI 卡是批处理快照，粒度/时效完全不同。
- **归类**：主投入（产品核心交互面）。

### 2.2 组合分析 `portfolio` — **保留，冻结**（用户已定）

- **展示**：组合 KPI（总值/盈亏/β/波动/回撤）、持仓表（RSI/趋势/回调标记）、PnL 条形、
  净值曲线 vs 同额基准、归一化对比、编辑持仓（写回 yaml）、个股新闻。
- **数据源**：`portfolio.local.yaml`（gitignore，永不入库）+ `PriceFetcher`（900s TTL）
  + `NewsFetcher`（900s）。
- **实时性**：近实时快照（15 分钟缓存，盘中会变）。
- **Power BI 重复**：**P1 Market Overview 的四张 position KPI 卡**（Market value / Cost /
  Unrealized PnL / PnL%）数字同源同义。但 PBI 侧是批处理导出的最新快照（fact_positions
  一个 as_of），不可交互不可分享；前端是**唯一实时、唯一可交互、唯一对外可见**的组合
  视图（硬约束 #1）——保留。
- **归类**：保留但冻结。

### 2.3 自选股 `watchlist` — **保留，冻结**（建议）

- **展示**：自选股快照表（现价/日涨跌%/RSI14/20D 波动%）+ 添加/删除（yfinance 探针验码）。
- **数据源**：watchlist 文件 + `PriceFetcher`（300s TTL）。
- **实时性**：盘中快照（5 分钟）。
- **Power BI 重复**：低。P4 的滚动波动是全历史**曲线**（34 标的 × 500 日），此处只是
  最新标量一列；P1 无 52w 表（已删）。
- **归类**：保留但冻结。理由：它是工作台/对冲台 universe（held + watch）的管理入口，
  删掉伤实时链路可用性；但功能已完备，无再投入方向。**不删。**

### 2.4 AI 分析 `analyst` — **主投入**

- **展示**：LLM 日报/盘中快报生成（驻留模型，免冷加载）+ 历史报告浏览（mtime 倒序）。
- **数据源**：`data/reports/*.md` + `make_daily_report`（本地 Qwen3-8B + v3 学生 adapter）。
- **实时性**：按需生成（盘中快报走 1 分钟会话数据）。
- **Power BI 重复**：无。
- **归类**：主投入（AI 交互面是决策表第一行的核心）。

### 2.5 模拟盘 `sim`（一路由两模式，分开判）

**a) YTD 回放** — **保留，冻结**（建议）

- **展示**：SignalGenerator 组合信号 × 规则策略的年初至今回放：总净值 vs SPY 买入持有、
  最大回撤、逐标的收益/Sharpe/调仓次数、当前持仓态。
- **数据源**：`PriceFetcher`（2024 起热身喂 MA200）+ `run_backtest`（next_open + 0.1% 成本），
  1800s TTL。
- **实时性**：否（日线批计算）。
- **Power BI 重复**：**P3 Backtest vs Benchmark** 语义最近——但 P3 深挖的是仓库里那**一次
  固定 run**（500 交易日，7 指标已对账）；此页是**任意标的组合 + 任意初始资金**的参数化
  交互回放。PBI 免费版无法承接参数化（What-if 注入在 powerbi/README 已知限制中列为未做），
  删除 = 净损失能力。
- **归类**：保留但冻结（对应决策表"回测深挖 → PBI 接管，前端冻结"的唯一前端实物）。

**b) 实盘会话（live）** — **主投入**

- **展示**：纸面交易会话启停（标的/数据源/初始现金/飞轮采集开关）、现金/权益/订单数、
  持仓、最近 50 成交、活跃 adapter；RL 门控无模型时诚实拒单的空态说明。
- **数据源**：`QuantAIClient` → `scripts/serve.py` v2 API（:8000，5s 自刷）。
- **实时性**：**实时**。
- **Power BI 重复**：无。
- **归类**：主投入（实时链路的展示面；后端通路本次铁律不动）。

## 3. 支撑组件（随宿主页面归类，不单独冻结）

| 文件 | 角色 | 状态 |
|---|---|---|
| `quantai/ui/charts.py` | plotly 纯函数图表层（K 线/指标/PnL 条形），离线可测 | 活跃（workstation 依赖） |
| `quantai/ui/cockpit.py` | AI 驻留守护线程 + 操作卡纯逻辑 | 活跃（workstation 依赖） |
| `quantai/ui/client.py` | v2 API 薄客户端（可注入 transport 单测） | 活跃（sim·live 依赖） |
| `quantai/ui/i18n.py` | 中英双语文案表 | 活跃（全站依赖） |
| `tests/ui/test_charts.py` / `test_client.py` / `test_workstation_chart.py` | UI 纯函数测试 | 保留；冻结页不再新增 |

## 4. 候选动作（唯一真正要改的东西，等确认）

| # | 动作 | 理由 | 建议 |
|---|---|---|---|
| C1 | 侧栏「打开 Tableau」按钮改指向 Power BI 工程（或删除该按钮） | `tableau/` 下**从来没有** .twb/.twbx（repo 只有书面 spec），按钮永远 fallback 到打开导出目录；Tableau 展示职责已被 Power BI 取代——这是一个指向已废弃目标的入口 | 改为「打开 Power BI 工程」`os.startfile(powerbi/QuantAI.pbip)`，找不到则打开 `powerbi/` 目录；独立 commit |
| C2 | README 加"两个展示面分工"节 + 架构说明标注实时链路→前端、批处理 CSV→Power BI，并明写 PBI 免费版无在线链接、交付为截图 + 本机 PBIP | 决策落地文档化（任务第三步） | 与 C1 同轮执行、分开 commit |

**候选删除：无。** 五页里没有"完全重复且无实时价值"或"坏掉"的页面。

## 5. 范围外（如实声明）

- **本地未跟踪残骸**（`.gitignore` 已排除、`git ls-files` 零命中，不在 repo 交付面）：
  `launch.py` + `build_exe.py`（旧"AI Trading Terminal"启动器，引用的
  `scripts/launch_desktop.ps1`、`scripts/run_live_paper_trading.py` **已不存在**，指向的
  `src/ui/desktop` 属旧架构残骸——本地已坏死）、`src/` 整棵旧代码树、`windows_app/StockApp`
  (C#/WPF 壳) + `dist/`、`node_modules/` + `package.json`（pixi-live2d-display）、三个
  Live2D 资产目录。本次收敛不碰；若要清理属盘面卫生任务，另行确认。
- **后端实时通路**：`quantai/api`、`scripts/serve.py`、data fetchers、agents、signals、
  backtest —— 铁律不动。
- **金融数学层**：`quantai/analysis`（BS/Greeks/指标）—— 完全不动。
- `启动仪表盘.bat` 为本地入口（中文名，未入库）；repo 对外启动口径 = README 的
  `python -m streamlit run quantai/ui/streamlit_app.py`，不变。
