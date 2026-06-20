"""quantai.agents.router.HeuristicRouter 测试（纯规则，等价迁移自 _moe_route）。"""

from __future__ import annotations

from quantai.agents.router import HeuristicRouter
from quantai.config import AppConfig


def _feat(vol=20.0, news_score=0.0, news_new=0.0, news_count=0.0):
    return {
        "volatility_ann_pct": vol,
        "signal": {
            "news_score": news_score,
            "news_new_count": news_new,
            "news_count": news_count,
        },
    }


def test_default_routes_scalper_on_calm_market():
    r = HeuristicRouter(vol_threshold=60.0)
    expert, meta = r.route(_feat(vol=20.0))
    assert expert == "scalper"
    assert meta["triggers"] == []


def test_high_vol_routes_analyst():
    r = HeuristicRouter(vol_threshold=60.0)
    expert, meta = r.route(_feat(vol=75.0))
    assert expert == "analyst"
    assert any("vol=" in t for t in meta["triggers"])


def test_vol_threshold_zero_never_triggers_on_vol():
    r = HeuristicRouter(vol_threshold=0.0)
    expert, _ = r.route(_feat(vol=999.0))
    assert expert == "scalper"


def test_any_news_new_count_routes_analyst():
    r = HeuristicRouter(vol_threshold=60.0, any_news=True)
    expert, meta = r.route(_feat(vol=20.0, news_new=2.0))
    assert expert == "analyst"
    assert any("news_new=" in t for t in meta["triggers"])


def test_any_news_false_uses_news_score_threshold():
    r = HeuristicRouter(vol_threshold=60.0, any_news=False, news_threshold=0.8)
    assert r.route(_feat(news_score=0.5))[0] == "scalper"
    assert r.route(_feat(news_score=0.9))[0] == "analyst"
    assert r.route(_feat(news_score=-0.9))[0] == "analyst"  # 绝对值


def test_news_adapter_available_routes_news():
    r = HeuristicRouter(
        vol_threshold=60.0, any_news=True, news_adapter_available=True
    )
    expert, _ = r.route(_feat(news_new=1.0))
    assert expert == "news"


def test_news_adapter_unavailable_falls_back_to_analyst():
    r = HeuristicRouter(
        vol_threshold=60.0, any_news=True, news_adapter_available=False
    )
    expert, _ = r.route(_feat(news_new=1.0))
    assert expert == "analyst"


def test_meta_contains_thresholds_and_scores():
    r = HeuristicRouter(vol_threshold=60.0, news_threshold=0.8)
    _, meta = r.route(_feat(vol=30.0, news_score=0.3, news_count=4.0))
    assert meta["thr_vol"] == 60.0
    assert meta["thr_news"] == 0.8
    assert meta["news_score"] == 0.3
    assert meta["news_count"] == 4.0
    assert meta["vol"] == 30.0


def test_bad_feature_values_do_not_crash():
    r = HeuristicRouter()
    expert, meta = r.route({"volatility_ann_pct": "x", "signal": {"news_score": None}})
    assert expert == "scalper"
    assert meta["vol"] == 0.0


def test_from_config_maps_fields():
    cfg = AppConfig()
    r = HeuristicRouter.from_config(cfg.agents.router, news_adapter_available=True)
    assert r.vol_threshold == cfg.agents.router.vol_threshold
    assert r.news_threshold == cfg.agents.router.news_threshold
    assert r.any_news == cfg.agents.router.any_news
    assert r.news_adapter_available is True
