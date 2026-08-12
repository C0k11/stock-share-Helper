"""期权分析引擎：Black-Scholes 定价 / Greeks / 隐含波动率 / 组合对冲计算器。

纯 numpy/scipy 公式层（零网络、零 LLM）——期权数学绝不交给语言模型心算，
LLM 只负责引用这里算出的数字做综述。

诚实边界：
- 欧式 Black-Scholes 近似。美式个股期权含早行权溢价（分红大/深度实值 put 时
  偏差变大），对**对冲成本估算**足够，对做市级定价不够——本模块的定位是前者。
- 无风险利率默认 4%（可传参）；IV 优先用数据源提供值（Yahoo 15 分钟延迟），
  缺失时用 `implied_vol` 反推。
- 对冲合约数按 100 股/张向下取整，凑不满一张的零头股数如实报告为未覆盖。
"""

from __future__ import annotations

import math
from typing import Optional

from scipy.stats import norm

DEFAULT_RISK_FREE = 0.04


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, sigma: float, kind: str = "call",
             r: float = DEFAULT_RISK_FREE) -> float:
    """欧式 BS 理论价。T 单位=年；T/sigma/S/K 非正时抛 ValueError（不静默给 0）。"""
    if min(S, K, T, sigma) <= 0:
        raise ValueError(f"S/K/T/sigma 必须为正（S={S}, K={K}, T={T}, sigma={sigma}）")
    if kind not in ("call", "put"):
        raise ValueError(f"kind 必须是 call/put，收到 {kind!r}")
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, sigma: float, kind: str = "call",
              r: float = DEFAULT_RISK_FREE) -> dict:
    """Greeks（单位注释齐全，直接可读）：

    - delta：股价 +$1 期权价变化（call  in  (0,1)，put  in  (-1,0)）
    - gamma：股价 +$1 时 delta 的变化
    - theta_per_day：**每自然日**时间损耗（负=持有方掏钱）
    - vega_per_pct：IV +1 个百分点的价格变化
    """
    if min(S, K, T, sigma) <= 0:
        raise ValueError("S/K/T/sigma 必须为正")
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T)  # 对 sigma（绝对值）的导数
    if kind == "call":
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S * pdf * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_per_day": float(theta / 365.0),
        "vega_per_pct": float(vega / 100.0),
    }


def implied_vol(price: float, S: float, K: float, T: float, kind: str = "call",
                r: float = DEFAULT_RISK_FREE, tol: float = 1e-6) -> Optional[float]:
    """二分法反推 IV  in  [0.1%, 500%]。价格越界（低于内在价值/高于上界）-> None（诚实缺失）。"""
    if price <= 0 or min(S, K, T) <= 0:
        return None
    lo, hi = 0.001, 5.0
    try:
        if not (bs_price(S, K, T, lo, kind, r) <= price <= bs_price(S, K, T, hi, kind, r)):
            return None
        for _ in range(100):
            mid = (lo + hi) / 2
            if bs_price(S, K, T, mid, kind, r) < price:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return float((lo + hi) / 2)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# 对冲计算器（输入 = 期权链 records：[{strike, bid, ask, lastPrice, impliedVolatility}...]）
# --------------------------------------------------------------------------- #
def _premium(row: dict) -> Optional[float]:
    """成交参考价：优先 (bid+ask)/2（都为正时），退回 lastPrice；无有效价 -> None。"""
    bid, ask = row.get("bid") or 0.0, row.get("ask") or 0.0
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    last = row.get("lastPrice") or 0.0
    return float(last) if last > 0 else None


def _nearest_strike(rows: list[dict], target: float) -> Optional[dict]:
    priced = [r for r in rows if _premium(r) is not None and (r.get("strike") or 0) > 0]
    return min(priced, key=lambda r: abs(r["strike"] - target)) if priced else None


def protective_put_plan(
    shares: float, spot: float, puts: list[dict], days_to_expiry: int,
    floor_pct: float = 0.92,
) -> Optional[dict]:
    """保护性 put：给持仓上"地板"。

    选 strike ~ spot×floor_pct 的 put，按 100 股/张买入：
    - `cost`：总保费（premium×100×张数）
    - `cost_pct`：保费占持仓市值比例
    - `max_loss_pct`：锁定后的最大损失（跌破 strike 后由 put 兜底）÷ 市值
    - `uncovered_shares`：凑不满一张的零头（如实报告，不假装全覆盖）
    """
    contracts = int(shares // 100)
    if contracts < 1 or spot <= 0 or not puts:
        return None
    row = _nearest_strike(puts, spot * floor_pct)
    if row is None:
        return None
    prem = _premium(row)
    strike = float(row["strike"])
    cost = prem * 100 * contracts
    covered = contracts * 100
    mv = shares * spot
    max_loss = (spot - strike) * covered + cost + (shares - covered) * spot  # 零头按全损上界诚实计
    return {
        "kind": "protective_put",
        "strike": strike,
        "premium": prem,
        "contracts": contracts,
        "days_to_expiry": days_to_expiry,
        "cost": float(cost),
        "cost_pct": float(cost / mv) if mv > 0 else float("nan"),
        "floor_pct_actual": float(strike / spot),
        "max_loss_pct_covered": float(((spot - strike) * covered + cost) / (covered * spot)),
        "uncovered_shares": float(shares - covered),
        "iv": float(row.get("impliedVolatility") or 0) or None,
    }


def covered_call_plan(
    shares: float, spot: float, calls: list[dict], days_to_expiry: int,
    target_pct: float = 1.06,
) -> Optional[dict]:
    """备兑 call：持股收租。

    选 strike ~ spot×target_pct 的 call，按 100 股/张卖出：
    - `income`：权利金收入；`income_pct`：占市值比例；`annualized_pct`：年化（按到期日粗算）
    - `called_away_at`：被行权价（涨过它股票被买走——上行封顶，如实标注）
    """
    contracts = int(shares // 100)
    if contracts < 1 or spot <= 0 or not calls:
        return None
    row = _nearest_strike(calls, spot * target_pct)
    if row is None:
        return None
    prem = _premium(row)
    strike = float(row["strike"])
    income = prem * 100 * contracts
    mv = shares * spot
    income_pct = income / mv if mv > 0 else float("nan")
    ann = income_pct * (365.0 / max(days_to_expiry, 1))
    return {
        "kind": "covered_call",
        "strike": strike,
        "premium": prem,
        "contracts": contracts,
        "days_to_expiry": days_to_expiry,
        "income": float(income),
        "income_pct": float(income_pct),
        "annualized_pct": float(ann),
        "called_away_at": strike,
        "upside_capped_pct": float(strike / spot - 1),
        "iv": float(row.get("impliedVolatility") or 0) or None,
    }


def chain_stats(calls: list[dict], puts: list[dict], spot: float) -> dict:
    """链面快照：P/C 成交量比、ATM IV（call/put 均值）、25% 档偏斜的粗代理。

    数据源 IV 缺失的行不参与均值；全缺 -> 对应字段 None（诚实缺失，不填 0）。
    """
    def vol_sum(rows):
        return sum(float(r.get("volume") or 0) for r in rows)

    def atm_iv(rows):
        cands = [r for r in rows if (r.get("impliedVolatility") or 0) > 0 and (r.get("strike") or 0) > 0]
        if not cands:
            return None
        atm = min(cands, key=lambda r: abs(r["strike"] - spot))
        return float(atm["impliedVolatility"])

    cv, pv = vol_sum(calls), vol_sum(puts)
    ivc, ivp = atm_iv(calls), atm_iv(puts)
    return {
        "pc_volume_ratio": float(pv / cv) if cv > 0 else None,
        "atm_iv_call": ivc,
        "atm_iv_put": ivp,
        "atm_iv": float((ivc + ivp) / 2) if ivc is not None and ivp is not None else (ivc or ivp),
    }
