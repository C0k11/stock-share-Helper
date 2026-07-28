"""report.json 生成器：QuantAI.Report 的 6 页布局用代码生成，可 diff 可迭代。

Power BI 旧版报表格式（report.json，非 PBIR preview）：
- section.visualContainers[].config 是字符串化 JSON；
- 每个视觉 = visualType + projections（角色→字段）+ prototypeQuery（实体/列/度量引用）。

用法：venv311\\Scripts\\python.exe powerbi\\build_report.py
产物：powerbi/QuantAI.Report/report.json（覆盖写，Desktop 重开生效）。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "QuantAI.Report" / "report.json"

M = "_Measures"

# 布局坐标全部写在 1280x720 的设计网格里，emit 时统一乘 S 输出到实际画布。
# 调画布只改这两行：S=1.25 -> 1600x900。
DESIGN_W, DESIGN_H = 1280, 720
S = 1.25
PAGE_W, PAGE_H = round(DESIGN_W * S), round(DESIGN_H * S)
BG = "#1B1B1F"


# --------------------------------------------------------------------------- #
# prototypeQuery 组装
# --------------------------------------------------------------------------- #
def _src(alias: str) -> dict:
    return {"Expression": {"SourceRef": {"Source": alias}}}


def col(entity_alias: str, prop: str, entity: str) -> dict:
    return {"kind": "col", "alias": entity_alias, "prop": prop, "entity": entity}


def mea(prop: str) -> dict:
    return {"kind": "mea", "alias": "m", "prop": prop, "entity": M}


def _query(fields: list[dict], orderby: dict | None = None) -> dict:
    froms: dict[str, str] = {}
    selects = []
    for f in fields:
        froms[f["alias"]] = f["entity"]
        ref = {**_src(f["alias"]), "Property": f["prop"]}
        if f["kind"] == "agg":
            selects.append({
                "Aggregation": {"Expression": {"Column": ref}, "Function": f.get("fn", 1)},
                "Name": f'Avg({f["entity"]}.{f["prop"]})',
            })
        else:
            key = "Column" if f["kind"] == "col" else "Measure"
            selects.append({key: ref, "Name": f'{f["entity"]}.{f["prop"]}'})
    q: dict = {
        "Version": 2,
        "From": [{"Name": a, "Entity": e, "Type": 0} for a, e in froms.items()],
        "Select": selects,
    }
    if orderby is not None:
        key = "Column" if orderby["kind"] == "col" else "Measure"
        q["OrderBy"] = [{"Direction": 1, "Expression": {key: {**_src(orderby["alias"]), "Property": orderby["prop"]}}}]
    return q


def agg(entity_alias: str, prop: str, entity: str, fn: int = 1) -> dict:
    """聚合列（fn: 0=Sum 1=Avg 2=Count 3=Min 4=Max）。"""
    return {"kind": "agg", "alias": entity_alias, "prop": prop, "entity": entity, "fn": fn}


def in_filter(entity: str, prop: str, values: list[str]) -> dict:
    """视觉级 In 过滤（默认标的选择）。"""
    return {
        "expression": {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}},
        "filter": {
            "Version": 2,
            "From": [{"Name": "s", "Entity": entity, "Type": 0}],
            "Where": [{"Condition": {"In": {
                "Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": prop}}],
                "Values": [[{"Literal": {"Value": f"'{v}'"}}] for v in values],
            }}}],
        },
        "type": "Categorical",
        "howCreated": 1,
    }


def _lit_color(c: str) -> dict:
    return {"color": {"Literal": {"Value": f"'{c}'"}}}


def _fill_rule_expr(measure_prop: str, rule: dict) -> dict:
    """官方 FillRule 结构（经 Desktop 往返序列化实测抄录，含 nullColoringStrategy）。"""
    return {"expr": {"FillRule": {
        "Input": {"Measure": {"Expression": {"SourceRef": {"Entity": M}}, "Property": measure_prop}},
        "FillRule": rule,
    }}}


_WILDCARD_SELECTOR = {"data": [{"dataViewWildcard": {"matchingOption": 1}}]}


def diverging_fill(measure_prop: str, negative: str = "#EF5350", center: str = "#1B1B1F",
                   positive: str = "#26A69A", three_stop: bool = True) -> dict:
    """dataPoint 填色的发散色阶。selector 必须带——没有它 Desktop 渲染直接报错（实锤）。"""
    rule: dict = {
        "linearGradient3": {
            "min": _lit_color(negative),
            "mid": _lit_color(center),
            "max": _lit_color(positive),
            "nullColoringStrategy": {"strategy": {"Literal": {"Value": "'asZero'"}}},
        }
    } if three_stop else {
        "linearGradient2": {
            "min": _lit_color(negative),
            "max": _lit_color(center),
            "nullColoringStrategy": {"strategy": {"Literal": {"Value": "'asZero'"}}},
        }
    }
    return {
        "dataPoint": [{
            "properties": {"fill": {"solid": {"color": _fill_rule_expr(measure_prop, rule)}}},
            "selector": _WILDCARD_SELECTOR,
        }]
    }


def diverging_back(measure_prop: str) -> dict:
    """矩阵值单元格背景的发散色阶（0 居中：负红/正绿）。

    与条形图 dataPoint 不同，矩阵 Cell elements 是按字段作用域的：selector 必须带
    "metadata": "<entity>.<measure>"，纯 dataViewWildcard 会被矩阵渲染器静默忽略
    （Desktop GUI 往返序列化实锤——config 其余部分逐字节相同也不渲染）。
    空值策略官方写法是 'noColor'（bar 的 'asZero'/'asNoColor' 家族在这里不适用）。
    """
    rule = {"linearGradient3": {
        "min": _lit_color("#EF5350"),
        "mid": _lit_color("#1B1B1F"),
        "max": _lit_color("#26A69A"),
        "nullColoringStrategy": {"strategy": {"Literal": {"Value": "'noColor'"}}},
    }}
    return {
        "values": [{
            "properties": {"backColor": {"solid": {"color": _fill_rule_expr(measure_prop, rule)}}},
            "selector": {**_WILDCARD_SELECTOR, "metadata": f"{M}.{measure_prop}"},
        }]
    }


def bool_filter(entity: str, prop: str, value: str = "true") -> dict:
    """视觉级布尔过滤（如 dim_date.is_trading_day = true）。"""
    return {
        "expression": {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": prop}},
        "filter": {
            "Version": 2,
            "From": [{"Name": "d", "Entity": entity, "Type": 0}],
            "Where": [{"Condition": {"Comparison": {
                "ComparisonKind": 0,
                "Left": {"Column": {"Expression": {"SourceRef": {"Source": "d"}}, "Property": prop}},
                "Right": {"Literal": {"Value": value}},
            }}}],
        },
        "type": "Categorical",
        "howCreated": 1,
    }


def visual(name: str, vtype: str, x: float, y: float, w: float, h: float,
           roles: dict[str, list[dict]], orderby: dict | None = None,
           title: str | None = None, extra_objects: dict | None = None,
           no_totals: bool = False, filters: list[dict] | None = None,
           title_size: str | None = None) -> dict:
    x, y, w, h = x * S, y * S, w * S, h * S
    all_fields = [f for fs in roles.values() for f in fs]
    projections = {}
    for role, fs in roles.items():
        refs = []
        for f in fs:
            if f["kind"] == "agg":
                refs.append({"queryRef": f'Avg({f["entity"]}.{f["prop"]})'})
            else:
                refs.append({"queryRef": f'{f["entity"]}.{f["prop"]}'})
        projections[role] = refs
    cfg: dict = {
        "name": name,
        "layouts": [{"id": 0, "position": {"x": x, "y": y, "width": w, "height": h, "z": 0}}],
        "singleVisual": {
            "visualType": vtype,
            "projections": projections,
            "prototypeQuery": _query(all_fields, orderby),
            "drillFilterOtherVisuals": True,
        },
    }
    objects: dict = dict(extra_objects or {})
    if no_totals:
        objects["total"] = [{"properties": {"totals": {"expr": {"Literal": {"Value": "false"}}}}}]
        objects["subTotals"] = [{"properties": {
            "rowSubtotals": {"expr": {"Literal": {"Value": "false"}}},
            "columnSubtotals": {"expr": {"Literal": {"Value": "false"}}},
        }}]
    if title:
        title_props = {
            "show": {"expr": {"Literal": {"Value": "true"}}},
            "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
        }
        if title_size:
            title_props["fontSize"] = {"expr": {"Literal": {"Value": title_size}}}
        cfg["singleVisual"]["vcObjects"] = {"title": [{"properties": title_props}]}
    if objects:
        cfg["singleVisual"]["objects"] = objects
    return {
        "config": json.dumps(cfg, ensure_ascii=False),
        "filters": json.dumps(filters or [], ensure_ascii=False),
        "height": h, "width": w, "x": x, "y": y, "z": 0,
    }


def section(name: str, display: str, ordinal: int, visuals: list[dict], hidden: bool = False) -> dict:
    return {
        "config": json.dumps({"visibility": 1} if hidden else {}),
        "displayName": display,
        "displayOption": 1,
        "filters": "[]",
        "height": float(PAGE_H),
        "name": name,
        "ordinal": ordinal,
        "visualContainers": visuals,
        "width": float(PAGE_W),
    }


# --------------------------------------------------------------------------- #
# 字段速记 / 通用样式
# --------------------------------------------------------------------------- #
def d_date() -> dict: return col("d", "date", "dim_date")
def s_sym() -> dict: return col("s", "symbol", "dim_symbol")


# 卡片：压字号防裁切（callout 默认 45pt），关类别标签（标题已说明含义，类别标签纯冗余还吃 20px）。
def _card_style(size: str) -> dict:
    return {
        "labels": [{"properties": {"fontSize": {"expr": {"Literal": {"Value": size}}}}}],
        "categoryLabels": [{"properties": {"show": {"expr": {"Literal": {"Value": "false"}}}}}],
    }


CARD_LABEL = _card_style("20D")     # 单值主卡
CARD_LABEL_SM = _card_style("16D")  # 窄卡 / 六连卡

CARD_TITLE = "12D"  # 卡片标题统一字号

# 34 个标的用列表切片器本来就不合适 → 下拉模式。
SLICER_DROPDOWN = {"data": [{"properties": {"mode": {"expr": {"Literal": {"Value": "'Dropdown'"}}}}}]}


# --------------------------------------------------------------------------- #
# Page 1 · Market Overview
# --------------------------------------------------------------------------- #
def market_page() -> dict:
    # 12 视觉太密 -> 10：去掉底部 mo_table（其列在本页图表里全有），两张图吃满下半页。
    visuals = [
        visual("mo_slicer_date", "slicer", 20, 20, 300, 80,
               {"Values": [d_date()]}),
        visual("mo_slicer_symbol", "slicer", 340, 20, 200, 80,
               {"Values": [s_sym()]}, extra_objects=SLICER_DROPDOWN),
        visual("mo_slicer_trading", "slicer", 560, 20, 180, 80,
               {"Values": [col("d", "is_trading_day", "dim_date")]},
               extra_objects=SLICER_DROPDOWN),
        visual("mo_card_symbols", "card", 760, 20, 240, 80,
               {"Values": [mea("Symbols Tracked")]}, title="Symbols",
               extra_objects=CARD_LABEL_SM, title_size=CARD_TITLE),
        visual("mo_card_tdays", "card", 1020, 20, 240, 80,
               {"Values": [mea("Trading Days")]}, title="Trading days",
               extra_objects=CARD_LABEL_SM, title_size=CARD_TITLE),
        visual("mo_card_mv", "card", 20, 115, 300, 110,
               {"Values": [mea("Market Value")]}, title="Position market value (latest)",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("mo_card_cost", "card", 330, 115, 300, 110,
               {"Values": [mea("Cost Value")]}, title="Position cost (latest)",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("mo_card_pnl", "card", 640, 115, 300, 110,
               {"Values": [mea("Unrealized PnL")]}, title="Unrealized PnL (latest)",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("mo_card_pnlpct", "card", 950, 115, 310, 110,
               {"Values": [mea("Unrealized PnL %")]}, title="Unrealized PnL %",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("mo_close_line", "lineChart", 20, 240, 740, 460,
               {"Category": [d_date()], "Series": [s_sym()], "Y": [mea("Close")]},
               title="Close over time by symbol (default SPY/QQQ/DIA - use slicer for more)",
               filters=[in_filter("dim_symbol", "symbol", ["SPY", "QQQ", "DIA"])]),
        visual("mo_52w_bar", "barChart", 780, 240, 480, 460,
               {"Category": [s_sym()], "Y": [mea("Pct From 52W High (Latest)")]},
               title="% from 52-week high (latest)",
               extra_objects=diverging_fill("Pct From 52W High (Latest)", three_stop=False)),
    ]
    return section("sec_market_overview", "Market Overview", 0, visuals)


# --------------------------------------------------------------------------- #
# Page 2 · Signals
# --------------------------------------------------------------------------- #
def signals_page() -> dict:
    # 三条带布局，消掉原 x=500~1260 的整条空白：卡片左侧竖排 / 强度图横满右侧；
    # 矩阵列粒度 月 -> 季度（25 列变 9 列，一屏放下，热力图看趋势不看每月读数）。
    visuals = [
        visual("sig_card_long", "card", 20, 20, 220, 95,
               {"Values": [mea("Strong Long %")]}, title="strong_long share",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("sig_card_short", "card", 20, 125, 220, 95,
               {"Values": [mea("Strong Short %")]}, title="strong_short share",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("sig_strength_bar", "barChart", 260, 20, 1000, 200,
               {"Category": [s_sym()], "Series": [col("f", "signal_strength", "fact_signals")],
                "Y": [mea("Signal Days")]},
               title="Signal strength mix by symbol"),
        visual("sig_matrix", "pivotTable", 20, 230, 740, 240,
               {"Rows": [s_sym()], "Columns": [col("d", "year_quarter", "dim_date")],
                "Values": [mea("Composite Signal")]},
               title="Avg composite signal - symbol x quarter (red short / green long)",
               no_totals=True, extra_objects=diverging_back("Composite Signal")),
        visual("sig_sub_avg", "clusteredColumnChart", 780, 230, 480, 240,
               {"Y": [agg("f", "trend_signal", "fact_signals"),
                      agg("f", "momentum_signal", "fact_signals"),
                      agg("f", "ma_cross_signal", "fact_signals"),
                      agg("f", "breakout_signal", "fact_signals")]},
               title="Component signal averages"),
        visual("sig_combo", "lineClusteredColumnComboChart", 20, 480, 1240, 220,
               {"Category": [d_date()], "Y": [mea("Composite Signal")], "Y2": [mea("Close")]},
               title="Composite signal (bars) vs close (line) - filter one symbol"),
    ]
    return section("sec_signals", "Signals", 1, visuals)


# --------------------------------------------------------------------------- #
# Page 3 · Backtest vs Benchmark
# --------------------------------------------------------------------------- #
def backtest_page() -> dict:
    kpis = [("Sharpe", "Sharpe"), ("CAGR", "CAGR"), ("Annual Volatility", "Annual vol"),
            ("Max Drawdown", "Max drawdown"), ("Win Rate", "Win rate"), ("Total Return", "Total return")]
    visuals = [
        visual(f"bt_card_{i}", "card", 20 + i * 210, 20, 195, 100,
               {"Values": [mea(m_)]}, title=t, extra_objects=CARD_LABEL_SM,
               title_size=CARD_TITLE)
        for i, (m_, t) in enumerate(kpis)
    ]
    visuals += [
        visual("bt_equity_line", "lineChart", 20, 130, 1240, 270,
               {"Category": [col("e", "date", "fact_backtest_equity")],
                "Y": [mea("Equity Indexed 100"), mea("SPY Indexed 100")]},
               title="Strategy vs SPY buy-and-hold (indexed to 100)"),
        visual("bt_dd_area", "areaChart", 20, 410, 1240, 130,
               {"Category": [col("e", "date", "fact_backtest_equity")], "Y": [mea("Drawdown")]},
               title="Drawdown"),
        # 14 列挤在一行 -> 只留 8 个有信息量的列
        visual("bt_table", "tableEx", 20, 550, 1240, 150,
               {"Values": [
                   col("r", "run_id", "fact_backtest_results"),
                   col("r", "strategy", "fact_backtest_results"),
                   col("r", "fill_timing", "fact_backtest_results"),
                   mea("Sharpe"), mea("CAGR"), mea("Annual Volatility"),
                   mea("Max Drawdown"), mea("Win Rate"),
               ]},
               title="Backtest run detail", no_totals=True),
    ]
    return section("sec_backtest", "Backtest vs Benchmark", 2, visuals)


# --------------------------------------------------------------------------- #
# Page 4 · Risk & Volatility
# --------------------------------------------------------------------------- #
def risk_page() -> dict:
    visuals = [
        visual("rk_card_best", "card", 20, 20, 220, 100,
               {"Values": [mea("Best Day")]}, title="Best day", extra_objects=CARD_LABEL,
               title_size=CARD_TITLE),
        visual("rk_card_worst", "card", 260, 20, 220, 100,
               {"Values": [mea("Worst Day")]}, title="Worst day", extra_objects=CARD_LABEL,
               title_size=CARD_TITLE),
        visual("rk_card_p05", "card", 500, 20, 220, 100,
               {"Values": [mea("Return P05")]}, title="5th percentile daily return",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("rk_vol_line", "lineChart", 20, 130, 1240, 280,
               {"Category": [d_date()], "Series": [s_sym()],
                "Y": [mea("Rolling Vol 20D (Ann)")]},
               title="Rolling 20-trading-day volatility, annualized sqrt(252) - blank until 20 bars"),
        visual("rk_hist", "columnChart", 20, 420, 740, 280,
               {"Category": [col("f", "ret_bin", "fact_prices")], "Y": [mea("Price Rows")]},
               orderby=col("f", "ret_bin", "fact_prices"),
               title="Daily return distribution (bin 0.5%)"),
        visual("rk_52w_bar", "barChart", 780, 420, 480, 280,
               {"Category": [s_sym()], "Y": [mea("Pct From 52W High (Latest)")]},
               title="% from 52-week high (latest)",
               extra_objects=diverging_fill("Pct From 52W High (Latest)", three_stop=False)),
    ]
    return section("sec_risk", "Risk & Volatility", 3, visuals)


# --------------------------------------------------------------------------- #
# Page 5 · News & Event Odds
# --------------------------------------------------------------------------- #
def news_page() -> dict:
    visuals = [
        visual("nw_card_total", "card", 20, 20, 220, 100,
               {"Values": [mea("News Count")]}, title="News rows", extra_objects=CARD_LABEL,
               title_size=CARD_TITLE),
        visual("nw_card_scored", "card", 260, 20, 220, 100,
               {"Values": [mea("Scored News")]}, title="Scored (of 5,581)", extra_objects=CARD_LABEL,
               title_size=CARD_TITLE),
        visual("nw_card_cov", "card", 500, 20, 220, 100,
               {"Values": [mea("Scored Coverage %")]}, title="Sentiment coverage - most news is unscored",
               extra_objects=CARD_LABEL, title_size=CARD_TITLE),
        visual("nw_card_syms", "card", 740, 20, 180, 100,
               {"Values": [mea("News Symbols")]}, title="Symbols with news", extra_objects=CARD_LABEL_SM,
               title_size=CARD_TITLE),
        visual("nw_card_markets", "card", 940, 20, 180, 100,
               {"Values": [mea("Odds Markets")]}, title="Polymarket markets", extra_objects=CARD_LABEL_SM,
               title_size=CARD_TITLE),
        visual("nw_by_date", "columnChart", 20, 130, 740, 260,
               {"Category": [d_date()], "Series": [col("n", "source", "fact_news")],
                "Y": [mea("News Count")]},
               title="News volume by date and source"),
        visual("nw_cat_bar", "barChart", 780, 130, 480, 260,
               {"Category": [col("e", "category", "fact_event_odds")],
                "Y": [mea("Odds Markets")]},
               title="Event markets by category"),
        visual("nw_odds_line", "lineChart", 20, 400, 620, 300,
               {"Category": [d_date()], "Series": [col("e", "category", "fact_event_odds")],
                "Y": [mea("Avg Yes Price")]},
               title="Avg implied probability by category over time"),
        visual("nw_odds_table", "tableEx", 660, 400, 600, 300,
               {"Values": [
                   col("e", "question", "fact_event_odds"),
                   col("e", "category", "fact_event_odds"),
                   mea("Latest Yes Price"),
                   agg("e", "prob_change_1d", "fact_event_odds"),
                   mea("Odds Volume 24h"),
                   col("e", "end_date", "fact_event_odds"),
               ]},
               title="Event odds detail", no_totals=True),
    ]
    return section("sec_news_events", "News & Event Odds", 4, visuals)


# --------------------------------------------------------------------------- #
# Page 6 · QA（验收页）
# --------------------------------------------------------------------------- #
def qa_page() -> dict:
    visuals = [
        visual("qa_pass_card", "card", 40, 40, 280, 120,
               {"Values": [mea("QA Pass Count")]}, title="Checks PASS"),
        visual("qa_total_card", "card", 360, 40, 280, 120,
               {"Values": [mea("QA Total Count")]}, title="Checks total (1 skipped by design)"),
        visual("qa_table", "tableEx", 40, 200, 1200, 480,
               {"Values": [
                   col("q", "check_id", "qa_expected"),
                   col("q", "description", "qa_expected"),
                   col("q", "expected", "qa_expected"),
                   mea("QA Actual (display)"),
                   mea("QA Status"),
               ]},
               orderby=col("q", "check_id", "qa_expected"),
               title="DAX vs pandas reconciliation (rel. tol 1e-6)", no_totals=True),
    ]
    return section("sec_qa", "QA", 5, visuals, hidden=True)


def main() -> None:
    report = {
        "config": json.dumps({
            "version": "5.43",
            "themeCollection": {
                "baseTheme": {"name": "CY24SU10", "version": "5.43", "type": 2},
                "customTheme": {"name": "quantai-dark.json", "version": "5.43", "type": 1},
            },
            "activeSectionIndex": 0,
            "defaultDrillFilterOtherVisuals": True,
            "settings": {"useNewFilterPaneExperience": True},
        }),
        "layoutOptimization": 0,
        "resourcePackages": [
            {"resourcePackage": {"disabled": False, "items": [
                {"name": "quantai-dark.json", "path": "quantai-dark.json", "type": 202}],
                "name": "RegisteredResources", "type": 1}},
            {"resourcePackage": {"disabled": False, "items": [
                {"name": "CY24SU10", "path": "BaseThemes/CY24SU10.json", "type": 202}],
                "name": "SharedResources", "type": 2}},
        ],
        "sections": [
            market_page(),
            signals_page(),
            backtest_page(),
            risk_page(),
            news_page(),
            qa_page(),
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    n_vis = sum(len(s["visualContainers"]) for s in report["sections"])
    print(f"[build_report] {len(report['sections'])} pages, {n_vis} visuals -> {OUT}")


if __name__ == "__main__":
    main()
