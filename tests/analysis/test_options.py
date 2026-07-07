"""期权引擎单测：BS 定价对教科书值、put-call parity、Greeks 符号/量纲、
IV 反推收敛、对冲计划算术（合约取整/成本/锁损/零头诚实）。"""

from __future__ import annotations

import pytest

from quantai.analysis.options import (
    bs_greeks,
    bs_price,
    chain_stats,
    covered_call_plan,
    implied_vol,
    protective_put_plan,
)


class TestBlackScholes:
    def test_textbook_values(self):
        # 经典基准：S=100, K=100, T=1y, r=5%, σ=20% → call≈10.4506, put≈5.5735
        assert bs_price(100, 100, 1.0, 0.20, "call", r=0.05) == pytest.approx(10.4506, abs=1e-3)
        assert bs_price(100, 100, 1.0, 0.20, "put", r=0.05) == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity(self):
        import math

        S, K, T, r, sig = 152.0, 145.0, 0.12, 0.04, 0.55
        c = bs_price(S, K, T, sig, "call", r)
        p = bs_price(S, K, T, sig, "put", r)
        assert c - p == pytest.approx(S - K * math.exp(-r * T), abs=1e-9)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            bs_price(100, 100, 0.0, 0.2)
        with pytest.raises(ValueError):
            bs_price(100, 100, 1.0, 0.2, "straddle")

    def test_greeks_signs_and_magnitudes(self):
        g_call = bs_greeks(100, 100, 1.0, 0.20, "call", r=0.05)
        g_put = bs_greeks(100, 100, 1.0, 0.20, "put", r=0.05)
        assert g_call["delta"] == pytest.approx(0.6368, abs=1e-3)
        assert g_put["delta"] == pytest.approx(g_call["delta"] - 1.0, abs=1e-9)
        assert g_call["gamma"] > 0 and g_call["gamma"] == pytest.approx(g_put["gamma"], abs=1e-12)
        assert g_call["theta_per_day"] < 0  # 买方每天掏时间价值
        assert g_call["vega_per_pct"] > 0


class TestImpliedVol:
    def test_roundtrip(self):
        price = bs_price(150, 140, 0.25, 0.45, "put", r=0.04)
        iv = implied_vol(price, 150, 140, 0.25, "put", r=0.04)
        assert iv == pytest.approx(0.45, abs=1e-4)

    def test_below_intrinsic_returns_none(self):
        # put 内在价值 = 10；报价 5 低于内在 → 无解，诚实 None
        assert implied_vol(5.0, 130, 140, 0.25, "put") is None


def _chain(spot=152.0):
    puts = [
        {"strike": 130, "bid": 2.0, "ask": 2.4, "volume": 50, "impliedVolatility": 0.62},
        {"strike": 140, "bid": 4.6, "ask": 5.0, "volume": 120, "impliedVolatility": 0.58},
        {"strike": 150, "bid": 8.8, "ask": 9.4, "volume": 200, "impliedVolatility": 0.55},
    ]
    calls = [
        {"strike": 155, "bid": 6.0, "ask": 6.6, "volume": 300, "impliedVolatility": 0.53},
        {"strike": 160, "bid": 4.2, "ask": 4.8, "volume": 180, "impliedVolatility": 0.54},
        {"strike": 170, "bid": 2.0, "ask": 2.6, "volume": 90, "impliedVolatility": 0.57},
    ]
    return calls, puts


class TestHedgePlans:
    def test_protective_put_math(self):
        _, puts = _chain()
        # 114 股：1 张合约覆盖 100 股，14 股零头如实报告
        plan = protective_put_plan(114, 152.0, puts, days_to_expiry=32, floor_pct=0.92)
        assert plan["strike"] == 140  # 152*0.92≈139.8 → 最近 140
        assert plan["contracts"] == 1
        assert plan["premium"] == pytest.approx(4.8)  # (4.6+5.0)/2
        assert plan["cost"] == pytest.approx(480.0)
        assert plan["uncovered_shares"] == pytest.approx(14)
        # 覆盖部分锁损：(152-140)*100 + 480 = 1680 → /15200 ≈ 11.05%
        assert plan["max_loss_pct_covered"] == pytest.approx(1680 / 15200, rel=1e-9)

    def test_covered_call_math(self):
        calls, _ = _chain()
        plan = covered_call_plan(114, 152.0, calls, days_to_expiry=32, target_pct=1.06)
        assert plan["strike"] == 160  # 152*1.06≈161.1 → 最近 160
        assert plan["income"] == pytest.approx(450.0)  # (4.2+4.8)/2 ×100
        assert plan["annualized_pct"] == pytest.approx((450 / (114 * 152)) * 365 / 32, rel=1e-9)
        assert plan["upside_capped_pct"] == pytest.approx(160 / 152 - 1)

    def test_under_100_shares_returns_none(self):
        calls, puts = _chain()
        assert protective_put_plan(60, 152.0, puts, 30) is None
        assert covered_call_plan(60, 152.0, calls, 30) is None

    def test_unpriced_rows_skipped(self):
        puts = [{"strike": 140, "bid": 0, "ask": 0, "lastPrice": 0}]
        assert protective_put_plan(100, 152.0, puts, 30) is None


class TestChainStats:
    def test_pc_ratio_and_atm_iv(self):
        calls, puts = _chain()
        s = chain_stats(calls, puts, spot=152.0)
        assert s["pc_volume_ratio"] == pytest.approx((50 + 120 + 200) / (300 + 180 + 90))
        assert s["atm_iv_put"] == pytest.approx(0.55)   # 150 最贴 152
        assert s["atm_iv_call"] == pytest.approx(0.53)  # 155 最贴 152
        assert s["atm_iv"] == pytest.approx(0.54)

    def test_missing_iv_honest_none(self):
        s = chain_stats([{"strike": 155, "volume": 1}], [{"strike": 150, "volume": 1}], 152.0)
        assert s["atm_iv"] is None
