"""趋势 / 动量指标：MACD（金叉/死叉）、均线体系、ROC/动量、RSI（超卖）、随机指标、回踩检测。

设计约定（全模块一致）：
- **纯函数**：输入 `pd.Series`/`pd.DataFrame`，输出新对象，不修改输入、无 IO、无全局状态。
- **因果性**：所有窗口只用 <= t 的数据（`rolling`/`ewm`/`shift(+n)`），无 lookahead；
  由 `tests/analysis/test_no_lookahead.py` 的截断不变性测试保证。
- **边界诚实**：数据不足（len < window）返回全 NaN 而非报错；输入含 NaN 时受影响的
  位置传播 NaN；数学上未定义处（如分母为 0）按各函数 docstring 标注的口径处理。
- **口径**：日频、NYSE 交易日；年化因子 252（见 `modules/analysis.md`「美股规则」一节）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

_DEFAULT_MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 50, 200)


# --------------------------------------------------------------------------- #
# 均线
# --------------------------------------------------------------------------- #
def sma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均。SMA_t = mean(close[t-w+1 .. t])。前 w-1 个值为 NaN。"""
    return close.rolling(window).mean().rename(f"sma_{window}")


def ema(close: pd.Series, span: int) -> pd.Series:
    """指数移动平均（递推口径，`adjust=False`）。

    EMA_t = α·close_t + (1-α)·EMA_{t-1}，α = 2/(span+1)，EMA_0 = close_0。
    与 TradingView/常见图表软件的 EMA 一致；只依赖 <= t 的数据（因果）。
    """
    return close.ewm(span=span, adjust=False).mean().rename(f"ema_{span}")


def moving_average_system(
    close: pd.Series, windows: Sequence[int] = _DEFAULT_MA_WINDOWS
) -> pd.DataFrame:
    """均线体系：各周期 SMA + 多头排列度。

    - `sma_{w}`：各周期简单均线。
    - `ma_alignment` ∈ [0, 1]：相邻均线对中「短均线 > 长均线」的占比
      （windows 升序两两比较）。1.0 = 完美多头排列，0.0 = 完美空头排列。
      任一参与比较的均线为 NaN 时该行为 NaN（不足以判断，不装懂）。
    """
    ws = sorted(set(int(w) for w in windows))
    if len(ws) == 0:
        raise ValueError("windows 不能为空")
    out = pd.DataFrame(index=close.index)
    for w in ws:
        out[f"sma_{w}"] = close.rolling(w).mean()
    if len(ws) >= 2:
        pairs = list(zip(ws[:-1], ws[1:]))
        acc = pd.Series(0.0, index=close.index)
        valid = pd.Series(True, index=close.index)
        for short_w, long_w in pairs:
            s, l = out[f"sma_{short_w}"], out[f"sma_{long_w}"]
            acc = acc + (s > l).astype(float)
            valid = valid & s.notna() & l.notna()
        out["ma_alignment"] = (acc / len(pairs)).where(valid)
    return out


# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #
def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD（Moving Average Convergence Divergence）。

    - macd_line = EMA(close, fast) − EMA(close, slow)
    - signal_line = EMA(macd_line, signal)
    - histogram = macd_line − signal_line

    经典参数 (12, 26, 9)。EMA 为 `adjust=False` 递推口径（见 :func:`ema`）。
    """
    if not (fast < slow):
        raise ValueError(f"要求 fast < slow，收到 fast={fast}, slow={slow}")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_line - signal_line,
        }
    )


def macd_cross(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD 金叉 / 死叉检测（布尔列）。

    - golden_cross_t：macd 在 t 上穿 signal，即 macd_{t-1} <= signal_{t-1} 且 macd_t > signal_t。
    - death_cross_t ：macd 在 t 下穿 signal，即 macd_{t-1} >= signal_{t-1} 且 macd_t < signal_t。

    任一参与比较的值为 NaN 时判 False（warmup 期不产信号）。第 0 行恒 False（无前值）。
    """
    m = macd(close, fast=fast, slow=slow, signal=signal)
    line, sig = m["macd"], m["macd_signal"]
    prev_line, prev_sig = line.shift(1), sig.shift(1)
    golden = (prev_line <= prev_sig) & (line > sig)
    death = (prev_line >= prev_sig) & (line < sig)
    return pd.DataFrame({"golden_cross": golden, "death_cross": death})


# --------------------------------------------------------------------------- #
# 动量
# --------------------------------------------------------------------------- #
def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """变动率（Rate of Change，百分比）。ROC_t = (close_t / close_{t-w} − 1) × 100。"""
    return (close.pct_change(window, fill_method=None) * 100.0).rename(f"roc_{window}")


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """价差动量。MOM_t = close_t − close_{t-w}（与 ROC 的比值口径相对，保留量纲）。"""
    return (close - close.shift(window)).rename(f"momentum_{window}")


# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标（Wilder 口径）。

    delta_t = close_t − close_{t-1}；gain = max(delta, 0)，loss = max(−delta, 0)。
    Wilder 平滑：avg_t = (avg_{t-1}·(w−1) + x_t)/w，等价 `ewm(alpha=1/w, adjust=False)`。
    RS = avg_gain / avg_loss；RSI = 100 − 100/(1+RS) ∈ [0, 100]。

    边界口径：
    - avg_loss = 0 且 avg_gain > 0（一路涨）→ RSI = 100；对称地一路跌 → 0。
    - avg_gain = avg_loss = 0（价格纹丝不动）→ RS 为 0/0 数学未定义，按**中性 50** 处理
      （多数图表软件口径；比返回 NaN 更利于下游布尔判定，已在测试固定该行为）。
    - 前 window 个值为 NaN（平滑未热身）。
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()

    out = pd.Series(np.nan, index=close.index, name=f"rsi_{window}")
    denom = avg_gain + avg_loss
    valid = denom.notna()
    flat = valid & (denom == 0)
    normal = valid & (denom > 0)
    # RSI = 100·avg_gain/(avg_gain+avg_loss) 与 100−100/(1+RS) 代数等价，且天然处理 avg_loss=0。
    out[normal] = 100.0 * avg_gain[normal] / denom[normal]
    out[flat] = 50.0
    # 热身期：Wilder 平滑前 window 步仍受初值主导，按惯例置 NaN。
    out.iloc[: min(window, len(out))] = np.nan
    return out


def rsi_signals(
    rsi_values: pd.Series, oversold: float = 30.0, overbought: float = 70.0
) -> pd.DataFrame:
    """RSI 超卖/超买布尔信号。oversold_t = RSI_t < oversold；overbought_t = RSI_t > overbought。

    NaN 处判 False（无数据不产信号）。
    """
    if not oversold < overbought:
        raise ValueError(f"要求 oversold < overbought，收到 {oversold} / {overbought}")
    return pd.DataFrame(
        {
            "rsi_oversold": rsi_values < oversold,
            "rsi_overbought": rsi_values > overbought,
        }
    )


# --------------------------------------------------------------------------- #
# 随机指标
# --------------------------------------------------------------------------- #
def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """随机指标（Stochastic Oscillator，慢速口径）。

    raw %K_t = 100 × (close_t − LL_w) / (HH_w − LL_w)，
    其中 HH_w/LL_w 为近 w 日最高价最高点 / 最低价最低点（含当日，因果）。
    %K = SMA(raw %K, smooth_k)（`smooth_k=1` 即快速随机指标）；%D = SMA(%K, d_window)。

    边界口径：HH_w = LL_w（w 日内价格区间为零）→ 0/0 未定义，置 NaN（诚实缺数据，
    不猜方向）；随 SMA 传播。
    """
    hh = high.rolling(k_window).max()
    ll = low.rolling(k_window).min()
    span = hh - ll
    raw_k = 100.0 * (close - ll) / span.where(span > 0)  # span==0 -> NaN
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_window).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


# --------------------------------------------------------------------------- #
# 回踩 / 反抽（pullback）
# --------------------------------------------------------------------------- #
def pullback(
    close: pd.Series,
    high: pd.Series,
    ma_fast: int = 20,
    ma_slow: int = 50,
    lookback: int = 20,
    min_retrace: float = 0.02,
    max_retrace: float = 0.10,
) -> pd.DataFrame:
    """上升趋势中的回踩（pullback）检测。

    定义（三个条件同时成立记为回踩，全部只用 <= t 的数据）：
    1. **趋势向上**：SMA(close, ma_fast)_t > SMA(close, ma_slow)_t 且 close_t > SMA_slow_t
       （快线在慢线上方，价格仍站在慢线上——回踩没破坏趋势）。
    2. **自近期高点回落**：retrace_t = 1 − close_t / max(high[t-lookback+1 .. t])
       ∈ [min_retrace, max_retrace]。回落太浅不算回踩，太深视为趋势破位而非回踩。
    3. **高点非当日**：close_t < 当日 high 的 lookback 期高点（由 2 的 retrace >= min_retrace 隐含）。

    输出列：
    - `retrace_from_high`：条件 2 的回落深度（连续值，便于下游/仪表盘用）。
    - `in_uptrend`：条件 1 布尔。
    - `pullback`：1 ∧ 2 布尔信号。
    NaN（均线/高点未热身）一律判 False。
    """
    if not 0 <= min_retrace < max_retrace:
        raise ValueError(f"要求 0 <= min_retrace < max_retrace，收到 {min_retrace} / {max_retrace}")
    fast_ma = close.rolling(ma_fast).mean()
    slow_ma = close.rolling(ma_slow).mean()
    rolling_high = high.rolling(lookback).max()
    retrace = 1.0 - close / rolling_high
    in_uptrend = (fast_ma > slow_ma) & (close > slow_ma)
    is_pullback = in_uptrend & (retrace >= min_retrace) & (retrace <= max_retrace)
    return pd.DataFrame(
        {
            "retrace_from_high": retrace,
            "in_uptrend": in_uptrend,
            "pullback": is_pullback,
        }
    )
