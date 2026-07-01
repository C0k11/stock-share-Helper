"""波动 / 通道指标：realized & annualized 波动率、Bollinger、ATR、rolling 相关性。

同 `trend.py` 的约定：纯函数、因果窗口、边界诚实（数据不足给 NaN 不报错）。
年化因子 252（NYSE 交易日，见 `modules/analysis.md`）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.analysis.trend import _require_positive

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# 波动率
# --------------------------------------------------------------------------- #
def realized_volatility(
    close: pd.Series, window: int = 20, annualize: bool = True
) -> pd.Series:
    """已实现波动率（rolling 标准差口径）。

    r_t = close_t/close_{t-1} − 1（简单收益）；
    vol_t = std(r[t-w+1 .. t])（样本标准差，ddof=1）；
    年化：vol_t × √252（美股日频惯例）。

    前 window 个位置 NaN（首个收益缺失 + 窗口热身）。window < 2 无意义，报错。
    """
    if window < 2:
        raise ValueError(f"window 需 >= 2，收到 {window}")
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol.rename(f"realized_vol_{window}")


# --------------------------------------------------------------------------- #
# Bollinger 通道
# --------------------------------------------------------------------------- #
def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """布林带。

    mid = SMA(close, w)；sd = rolling std(close, w)（**ddof=0 总体标准差**，
    John Bollinger 原版 / TA-Lib / `ta` 库同口径，已与 `ta` 交叉验证逐位一致；
    注意与 :func:`realized_volatility` 的 ddof=1 样本口径不同——那是统计估计惯例）；
    upper = mid + k·sd；lower = mid − k·sd；
    bandwidth = (upper − lower) / mid（带宽，衡量波动挤压/扩张）；
    percent_b = (close − lower) / (upper − lower) ∈ 大致 [0,1]（价格在带内的位置）。

    边界口径：
    - 价格恒定 → sd = 0 → upper = lower = mid，bandwidth = 0；percent_b 为 0/0
      未定义，置 NaN（不猜位置）。
    - mid = 0（理论上价格为 0）→ bandwidth 置 NaN。
    """
    if window < 2:
        raise ValueError(f"window 需 >= 2，收到 {window}")
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    width = upper - lower
    bandwidth = width / mid.where(mid != 0)
    percent_b = (close - lower) / width.where(width > 0)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }
    )


# --------------------------------------------------------------------------- #
# ATR
# --------------------------------------------------------------------------- #
def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """平均真实波幅（Average True Range，Wilder 口径）。

    TR_t = max(high_t − low_t, |high_t − close_{t-1}|, |low_t − close_{t-1}|)；
    ATR = Wilder 平滑(TR, w) = `ewm(alpha=1/w, adjust=False)`（与 RSI 同一平滑）。

    边界口径（审计后收紧，2026-07-01）：
    - 无前收（首日，或前一日 close 缺失）→ 回落 TR = high − low（首日惯例的推广，
      跳空分量不可知就不计，**不再用 skipna 静默吞掉**旧实现会把 NaN 行低估成数值）。
    - 当日 high 或 low 缺失 → 该日区间不可知，TR 与 ATR 输出 **NaN**；平滑在洞后
      从上一有效状态恢复递推（同 :func:`quantai.analysis.trend.rsi` 的洞口径）。
    - 前 window 个值置 NaN（平滑热身）。
    """
    _require_positive(window=window)
    prev_close = close.shift(1)
    base = high - low  # 当日区间：h/l 任一缺失 -> NaN（诚实）
    gap = pd.concat(
        [(high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)  # 无前收 -> NaN -> 回落用 base
    tr = pd.Series(np.fmax(base.to_numpy(), gap.to_numpy()), index=base.index)
    tr = tr.where(base.notna())  # 区间不可知的行强制 NaN（fmax 会偏袒非 NaN 一侧）
    out = tr.ewm(alpha=1.0 / window, adjust=False).mean().where(tr.notna())
    out.iloc[: min(window, len(out))] = np.nan
    return out.rename(f"atr_{window}")


# --------------------------------------------------------------------------- #
# 相关性
# --------------------------------------------------------------------------- #
def rolling_correlation(
    close_a: pd.Series,
    close_b: pd.Series,
    window: int = 60,
    on_returns: bool = True,
) -> pd.Series:
    """两标的的滚动 Pearson 相关。

    默认在**收益**上算（`on_returns=True`）：价格序列多为非平稳（共同漂移导致
    价格相关系数虚高，spurious correlation），收益相关才是风控/配对里有意义的量。
    `on_returns=False` 保留价格相关（诊断用）。

    索引按交集对齐（inner join）后计算；窗口内任一侧方差为 0 → 相关未定义，pandas
    给 NaN（保留该口径）。前导 NaN 数：收益口径恰 window 个（首收益 NaN 与窗口热身
    重叠）、价格口径恰 window−1 个（实测校正过的精确计数）。
    """
    if window < 2:
        raise ValueError(f"window 需 >= 2，收到 {window}")
    a, b = close_a.align(close_b, join="inner")
    if on_returns:
        a = a.pct_change(fill_method=None)
        b = b.pct_change(fill_method=None)
    return a.rolling(window).corr(b).rename(f"rolling_corr_{window}")
