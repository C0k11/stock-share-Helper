# 前端编排反哺方案（2026-07-28，未确认前不动代码）

> 目标：把 Power BI 六页逼出来的编排纪律搬回前端，**不砍任何功能**。
> 本文全部行号/键名来自当前工作树实测（`quantai/ui/streamlit_app.py`，1,027 行）。

## 1. 现有分支结构实测（先纠正一处口径）

开工 prompt 说"导航靠 3 个 st.radio（L158/L355/L939）"。实测：**这三个里只有
L941 参与导航，真正的页面路由是 L908 的 `st.sidebar.radio`（prompt 里漏了它，
它用的是 `st.sidebar.radio` 写法，`grep "st.radio"` 抓不到）**。全量分支地图：

| 位置 | 控件 | 职责 | 覆盖行数 |
|---|---|---|---|
| L901 | sidebar.radio `ui_lang_label` | 全站语言 zh/en | 全文件（i18n.tr 全站引用） |
| **L908** | sidebar.radio `ui_page` | **主导航**：5 页互斥渲染 | L921-1025 的 if 链分发全部页面 |
| L158 | radio `pf_view` | 组合页内部视图（净值/归一化） | L162-198（37 行，页内开关，非导航） |
| L355 | radio（图型 line/candle） | 工作台指标 popover 内开关 | 传参给 workstation_figure（1 行效果） |
| L941 | radio（sim mode ytd/live） | 模拟盘二级路由 | ytd→L729-876（148 行）；live→L949-1023（75 行） |
| L602 | st.tabs | 工作台底部 持仓/新闻 页签 | L602-627（26 行） |

页面函数行界：`_render_portfolio_page` L33-215（183 行）、`_render_workstation_page`
L243-627（385 行）、`_render_watchlist_page` L630-699（70 行）、`_render_analyst_page`
L702-726（25 行）、`_render_ytd_replay` L729-876（148 行）、`main()`（路由+实盘会话
内联）L879-1027（149 行）。

## 2. session_state 键清单与拆页保持

带 key 的控件（拆页后由 `st.session_state` 全局保持，**行为与现状完全一致**）：

- 全局：`ui_lang_label` / `ui_lang`（语言）、`ui_page`（导航，拆后由 st.navigation 接管）
- 组合页：`pf_view`、`pf_e_cash`、`pf_e_sh_{i}`、`pf_e_cb_{i}`（编辑持仓行）
- 工作台：`ws_ai_on`、`ws_tact_syms`、`ws_main_chart`、`ws_auto_refresh_{lang}`
- 对冲台：`hd_sym`、`hd_shares`、`hd_floor`、`hd_target`

无 key 控件（工作台标的 selectbox L332、时间档 segmented、指标 popover 勾选组、
watchlist 输入、sim tickers/cash）在**现状的 radio 导航下切页本来就会重置**——
radio 分支= 条件渲染 = 组件树卸载。拆成 multipage 后同样重置，**无新增退化**；
若想要"切页记住标的"，给 selectbox 补 key 是一行改动，可作为拆页附带修复。

进程级共享单例不受拆页影响：`CockpitDaemon`（`st.cache_resource` 后台线程，
切页后台照跑）、`_resident_llm`、各 `st.cache_data` 缓存——全部进程级，页面无关。

## 3. 自动刷新 × multipage 评估（结论：无退化）

7 个 `@st.fragment(run_every=…)` 实测：L295 持仓横幅 30s、L357 主图 15/30/60s/off、
L433 作战台 30s、L469 AI 视图 15s、L510 对冲台 120s、L985 实盘会话 5s。

fragment 只在**其所在页面的脚本运行期间**存在：切走页面 = 分支不执行 = 定时器
消失。这在现状（radio 条件渲染）与 multipage（页面文件不运行）下**语义相同**，
不存在"拆页导致自刷失效"的额外风险。后台推理不依赖 fragment（daemon 线程独立），
切回工作台读 `daemon.state` 毫秒级恢复显示。5s 自刷的实盘会话页独立成页后，
反而不再与 YTD 回放共享一个 radio 分支（现状切 mode 会卸载对方状态）。

## 4. 拆分后的页面清单（一页一个问题）

采用 **`st.navigation` + `st.Page`**（streamlit 1.52.2 实测在装，≥1.36 即可；
pyproject 需把 `streamlit>=1.31.0` 提到 `>=1.36`），入口统一 set_page_config +
语言选择 + 主题注入，五个页面模块从现有函数**平移**（不重写逻辑）：

| 新文件 | 页名（主题一句话） | 来源 |
|---|---|---|
| `quantai/ui/app.py` | 入口：导航/语言/全局 CSS | main() 的 L879-919 |
| `quantai/ui/pages/workstation.py` | 现在该不该动仓位？（实时作战） | L218-627 + cockpit/charts 引用不动 |
| `quantai/ui/pages/portfolio.py` | 我的钱怎么样了？（组合快照+净值） | L33-215 |
| `quantai/ui/pages/watchlist.py` | 候选池里谁值得看？ | L630-699 |
| `quantai/ui/pages/analyst.py` | AI 怎么读今天的市场？ | L702-726 |
| `quantai/ui/pages/sim.py` | 这套规则历史上跑得如何？（回放+纸面会话） | L729-876 + L937-1023 |

同步搬进的编排纪律（各页同构）：KPI 行恒在顶、明细在下（现状 workstation/
portfolio 已如此，watchlist/sim 补齐）；`st.columns` 比例提成模块常量；
**口径写进标题**（对开源最值钱，具体改造点）：
- 组合页"年化波动/最大回撤"→ 标题补"(holdings-based synthetic history)"（正文已有脚注，提级到标题）
- 工作台滚动指标 → "RSI(14) / MACD(12,26,9) on visible bars"（现在只有指标名）
- YTD 回放净值图 → "next-open fills, 0.1% turnover cost, composite>0.1 long/flat"（现状在页脚 footnote，提级）
- watchlist "20D 波动%" → "realized vol, 20 trading bars, annualized"
默认值克制检查：已排查——工作台单标的、回放默认仅持仓、组合页仅持仓条数，
**无"默认全量渲染"问题**（PBI Page 1 那类 34 线全上的坑前端没有）。

## 5. 风险与回滚

- 风险①：五页文件间共享工具（`_fmt_money`、`_TIMEFRAMES`、`_hist` 缓存工厂）
  需抽到 `quantai/ui/common.py`——纯搬运，pytest 的 3 个 UI 测试（charts/client
  纯函数）不受影响，但 **UI 本体无自动化测试**，回归只能实测（启动 + 五页
  逐页点开 + 自刷观察 ≥1 轮 + 语言切换），估 1 小时。
- 风险②：`st.navigation` 的 sidebar 页列表与现有自定义 sidebar（语言/Tableau
  按钮/组合页参数）并存的布局微调，可能有 1-2 轮试错。
- 风险③：深链/书签行为变化（radio 无 URL，st.Page 有 URL path——只赚不赔，
  但 README 截图路径要重截）。
- 回滚：整个拆分做成**单个 commit**（纯平移零逻辑变更），revert 即回现状；
  口径进标题、补 key 等增强放第二个 commit，独立可回滚。

## 6. 工作量与建议

工作量：平移 3-4 h + 实测回归 1 h + 截图/文档 0.5 h ≈ **1 个工作日内**。

**建议：拆（完整拆，不是部分拆）。** 理由：①五个 `_render_*` 函数本就边界清晰，
平移成本低、逻辑零改动；②自刷/单例/状态三项经上文逐项核实**无退化**，
"宁可不拆"的触发条件不成立；③收益实打实——1,027 行单文件变 6 个 ≤400 行
模块、页面获得 URL、实盘会话与回放解耦、口径进标题让开源读者不读代码就懂
数字。唯一不动的是三个页内 radio（L158/L355/L941-mode），它们是视图开关不是
导航，本来就该留在页内。
