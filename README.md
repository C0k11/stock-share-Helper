# QuantAI - Intelligent Quantitative Investment Assistant

> A multimodal AI-powered personal securities investment assistant providing ETF portfolio strategy recommendations, risk alerts, and explainable decisions.

[中文版](#quantai---智能量化投顾助手-1)

---

## 🎯 Project Overview

**Not a black-box promising profits, but an intelligent assistant providing explainable recommendations under risk constraints.**

Core Capabilities:
- **Strategy Recommendations**: Daily target positions, entry/exit conditions, stop-loss/take-profit
- **Risk Alerts**: Volatility spikes, drawdown warnings, correlation surge detection
- **News Understanding**: LLM-powered news structuring, explaining "why reduce position"
- **Historical Analysis**: Analyze historical trends + news to identify key market drivers
- **Personalization**: Adjust positions based on user risk profiles (Conservative/Balanced/Aggressive)

---

## 📋 MVP Definition (v1)

| Dimension | Decision |
|-----------|----------|
| **Market** | US ETFs first, HK ETFs in Phase 2 |
| **Symbol Pool** | TLT, IEF, GLD, SPY, QQQ, SHY |
| **Frequency** | Daily (post-market), intraday as Phase 2 |
| **Output** | Strategy recommendations + simulated portfolio curve |
| **Risk Profiles** | Conservative(5% DD) / Balanced(10%) / Aggressive(20%) |
| **Tech Stack** | Python + Local LLM fine-tuning (RTX 4090) |
| **Data Sources** | Free sources first (yfinance/Stooq/NewsAPI) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Interface Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Web UI    │  │  REST API   │  │   Alerts (Optional)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Decision Engine Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Strategy   │  │    Risk     │  │    LLM Explainer        │  │
│  │ (Trend/Mom) │  │  (DD/Vol)   │  │  (News→Reasoning)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Feature & Signal Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Technical  │  │   Regime    │  │    News Factors         │  │
│  │  (MA/Mom)   │  │  Detection  │  │  (Events/Sentiment)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                          Data Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Price    │  │    Macro    │  │      News Data          │  │
│  │   (OHLCV)   │  │    (VIX)    │  │     (RSS/API)           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Stock/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore
├── config/                   # Configuration files
│   ├── settings.yaml         # Global settings
│   ├── symbols.yaml          # Symbol pool config
│   └── risk_profiles.yaml    # Risk profile config
├── data/                     # Data directory (not in git)
├── src/                      # Source code
│   ├── data/                 # Data layer (fetcher/calendar/storage)
│   ├── features/             # Features (technical/regime/news)
│   ├── strategy/             # Strategy (signals/position/rules)
│   ├── risk/                 # Risk management (drawdown/vol/alerts)
│   ├── backtest/             # Backtesting (engine/costs/metrics)
│   ├── llm/                  # LLM (news_parser/explainer/finetune)
│   └── api/                  # FastAPI interface
├── scripts/                  # Scripts
│   ├── download_data.py      # Download historical data
│   ├── daily_update.py       # Daily update
│   └── run_backtest.py       # Run backtest
├── models/                   # Model files (not in git)
└── tests/                    # Unit tests
```

---

## 🚀 Roadmap

### Phase 1: Data & Backtest Foundation ✅
- [x] Data fetcher module (yfinance)
- [x] Trading calendar & timezone handling
- [x] Technical factors (MA/Momentum/Volatility)
- [x] Backtest engine (costs/slippage/rebalancing)
- [x] Performance metrics

### Phase 2: Strategy MVP ✅
- [x] Regime detection (Risk-On/Off/Transition)
- [x] Volatility-targeted position sizing
- [x] Trend/Momentum signals
- [x] Drawdown protection rules
- [ ] Walk-forward validation

### Phase 3: LLM Fine-tuning 🔄
- [ ] News data collection & labeling
- [ ] Fine-tuning dataset construction
- [ ] LoRA fine-tuning Qwen2.5-7B
- [ ] News structuring inference
- [ ] Decision explanation generation

### Phase 4: Product Integration
- [ ] User risk profile configuration
- [ ] Daily recommendation pipeline
- [ ] Simulated portfolio tracking
- [ ] API endpoints
- [ ] Simple web UI

### Phase 5: Enhancements (Future)
- [ ] Intraday signals & triggers
- [ ] HK ETF support
- [ ] ML prediction models
- [ ] A-share ETF support

---

## 📊 Output Protocol

Daily output for each ETF:

```json
{
  "date": "2024-01-15",
  "symbol": "SPY",
  "recommendation": {
    "action": "reduce",
    "target_position": 0.3,
    "stop_loss": "$465",
    "exit_condition": "Execute if price drops below $470"
  },
  "risk_alerts": [
    {"type": "volatility_up", "severity": "medium", "message": "VIX rose to 22"}
  ],
  "explanation": {
    "technical": "SPY broke below 20-day MA",
    "regime": "Risk state shifted to Transition",
    "news": "Fed minutes suggest rates stay higher for longer"
  },
  "confidence": 0.72
}
```

---

## ⚙️ Risk Profile Configuration

| Profile | Max Drawdown | Target Vol | Max Single Position | Min Cash |
|---------|--------------|------------|---------------------|----------|
| Conservative | 5% | 6% | 25% | 30% |
| Balanced | 10% | 10% | 35% | 15% |
| Aggressive | 20% | 15% | 50% | 5% |

---

## 🔧 Setup

### Hardware Requirements
- **Minimum**: 8GB RAM, no GPU
- **Recommended**: 16GB RAM, 8GB VRAM
- **Current**: RTX 4090 + AMD 7950X3D (LLM fine-tuning ready)

### Installation

```bash
git clone https://github.com/C0k11/stock-share-Helper.git
cd stock-share-Helper

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_backtest.py
```

---

## 📜 Disclaimer

⚠️ **For educational and research purposes only. Not investment advice.**

- Past performance does not guarantee future results
- All recommendations are for reference only
- Users bear all responsibility for investment decisions

---

## 📄 License

MIT License

---
---

# QuantAI - 智能量化投顾助手

> 基于多模态AI的个人证券投资助手，提供ETF组合策略建议、风险预警与可解释决策。

[English Version](#quantai---intelligent-quantitative-investment-assistant)

---

## 🎯 项目定位

**不是"保证赚钱"的黑箱，而是"风险约束下可解释建议"的智能助手。**

核心能力：
- **策略建议**：每日输出目标仓位、入场/离场条件、止损止盈
- **风险预警**：波动上升、回撤预警、相关性飙升检测
- **新闻理解**：LLM结构化新闻事件，解释"为什么建议减仓"
- **历史分析**：分析历史走势+新闻，识别影响股市的关键因素
- **个性化**：根据用户风险偏好（保守/平衡/进取）调整仓位

---

## 📋 MVP定义（第1版）

| 维度 | 决定 |
|------|------|
| **市场** | 美股ETF优先，港股ETF第二阶段 |
| **标的池** | TLT, IEF, GLD, SPY, QQQ, SHY |
| **频率** | 日频（收盘后生成建议），盘中作为二期增强 |
| **输出** | 策略建议/提醒 + 模拟组合收益曲线 |
| **风险档位** | 保守(5%回撤) / 平衡(10%) / 进取(20%) |
| **技术栈** | Python + 本地LLM微调(4090) |
| **数据源** | 免费源先跑通（yfinance/Stooq/NewsAPI） |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Web前端    │  │  API接口    │  │  提醒/告警（可选）       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         决策引擎层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 策略模块    │  │ 风控模块    │  │  LLM解释模块            │  │
│  │ (趋势/动量) │  │ (止损/回撤) │  │  (新闻→建议原因)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         特征与信号层                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 技术因子    │  │ 风险状态    │  │  新闻因子               │  │
│  │ (MA/动量)   │  │ (Regime)    │  │  (事件/情绪)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                         数据层                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 行情数据    │  │ 宏观指标    │  │  新闻数据               │  │
│  │ (OHLCV)     │  │ (VIX等)     │  │  (RSS/API)              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
Stock/
├── README.md                 # 本文件
├── requirements.txt          # Python依赖
├── .gitignore
├── config/                   # 配置文件
│   ├── settings.yaml         # 全局配置
│   ├── symbols.yaml          # 标的池配置
│   └── risk_profiles.yaml    # 风险档位配置
├── data/                     # 数据目录（不提交到git）
├── src/                      # 源代码
│   ├── data/                 # 数据层（fetcher/calendar/storage）
│   ├── features/             # 特征工程（technical/regime/news）
│   ├── strategy/             # 策略模块（signals/position/rules）
│   ├── risk/                 # 风控模块（drawdown/volatility/alerts）
│   ├── backtest/             # 回测模块（engine/costs/metrics）
│   ├── llm/                  # LLM模块（news_parser/explainer/finetune）
│   └── api/                  # API接口
├── scripts/                  # 脚本
│   ├── download_data.py      # 下载历史数据
│   ├── daily_update.py       # 每日更新
│   └── run_backtest.py       # 运行回测
├── models/                   # 模型文件（不提交到git）
└── tests/                    # 单元测试
```

---

## 🚀 迭代路线图

### Phase 1: 数据与回测基础 ✅
- [x] 数据抓取模块（yfinance）
- [x] 交易日历与时区处理
- [x] 技术因子计算（MA/动量/波动率）
- [x] 回测引擎（成本/滑点/再平衡）
- [x] 基础绩效指标

### Phase 2: 策略MVP ✅
- [x] 风险状态检测（Regime: Risk-On/Off/Transition）
- [x] 目标波动仓位计算
- [x] 趋势/动量信号
- [x] 回撤保护规则
- [ ] Walk-forward验证

### Phase 3: LLM微调 🔄
- [ ] 新闻数据收集与标注
- [ ] 微调数据集构建（事件分类/情绪/影响方向）
- [ ] LoRA微调Qwen2.5-7B
- [ ] 新闻结构化推理
- [ ] 决策解释生成

### Phase 4: 产品集成
- [ ] 用户风险档位配置
- [ ] 每日建议生成流程
- [ ] 模拟组合跟踪
- [ ] API接口
- [ ] 简单Web展示

### Phase 5: 增强（后续）
- [ ] 盘中信号与触发器
- [ ] 港股ETF支持
- [ ] ML预测模型（LightGBM）
- [ ] A股ETF支持

---

## 📊 输出协议

每日为每个ETF输出：

```json
{
  "date": "2024-01-15",
  "symbol": "SPY",
  "recommendation": {
    "action": "减仓",
    "target_position": 0.3,
    "stop_loss": "$465",
    "exit_condition": "价格跌破$470立即执行"
  },
  "risk_alerts": [
    {"type": "volatility_up", "severity": "medium", "message": "VIX升至22，波动上升"}
  ],
  "explanation": {
    "technical": "SPY跌破20日均线，短期趋势转弱",
    "regime": "风险状态从Risk-On转为Transition",
    "news": "美联储会议纪要显示可能维持高利率更长时间"
  },
  "confidence": 0.72
}
```

---

## ⚙️ 风险档位配置

| 档位 | 最大回撤容忍 | 目标年化波动 | 单标的上限 | 现金下限 |
|------|-------------|-------------|-----------|---------|
| 保守 | 5% | 6% | 25% | 30% |
| 平衡 | 10% | 10% | 35% | 15% |
| 进取 | 20% | 15% | 50% | 5% |

---

## 🔧 环境配置

### 硬件要求
- **最低**：8GB内存，无GPU
- **推荐**：16GB内存，8GB显存
- **当前配置**：RTX 4090 + AMD 7950X3D（支持LLM微调）

### 安装

```bash
git clone https://github.com/C0k11/stock-share-Helper.git
cd stock-share-Helper

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_backtest.py
```

---

## 📜 免责声明

⚠️ **本项目仅供学习研究，不构成任何投资建议。**

- 历史业绩不代表未来表现
- 所有策略建议仅供参考，投资决策需自行判断
- 使用本系统产生的任何损失由用户自行承担

---

## 📄 License

MIT License
