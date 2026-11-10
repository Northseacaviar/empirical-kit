"""机制测试：DID 动态效应与预趋势检验。

已知答案：
- 处理效应 0.5 且只出现在 k>=0 → 处理后系数均值 ≈ 0.5，预趋势 p 大
- 处理组在处理前已有上升趋势 → 预趋势检验应报警
"""
import numpy as np
import pandas as pd

from epkit.estimate.did import did


def make(effect=0.5, pre_trend=0.0, seed=3, n_firm=120, years=range(2000, 2020)):
    rng = np.random.default_rng(seed)
    years = list(years)
    cohort_year = 2010
    rows = []
    firm_fe = rng.normal(0, 0.5, n_firm)
    for i in range(n_firm):
        treated = i % 2 == 0
        g = cohort_year if treated else 0
        for ti, yr in enumerate(years):
            post = 1 if (treated and yr >= cohort_year) else 0
            trend = pre_trend * (ti - 10) if treated else 0.0
            y = firm_fe[i] + 0.05 * ti + effect * post + trend + rng.normal(0, 0.3)
            rows.append({"firm": i, "year": yr, "g": g, "y": y})
    return pd.DataFrame(rows)


def test_recovers_effect_and_passes_pretrend():
    f = did(make(), {"id": "d", "y": "y", "cohort": "g", "period": "year",
                     "idvar": "firm", "window": [-4, 4]})
    post = [r["coef"] for r in f.data["event_time"] if r["k"] >= 0]
    assert np.mean(post) == __import__("pytest").approx(0.5, abs=0.2), f"处理后系数 {np.mean(post):.3f}"
    assert f.data["pre_trend_wald"]["p"] > 0.05
    assert f.status in ("ok", "warn")


def test_detects_parallel_trend_violation():
    f = did(make(pre_trend=0.12), {"id": "d", "y": "y", "cohort": "g", "period": "year",
                                   "idvar": "firm", "window": [-4, 4]})
    assert f.data["pre_trend_wald"]["p"] < 0.05
    assert f.status == "warn"
    assert any("预趋势" in w for w in f.warnings)


def test_staggered_is_flagged():
    d = make()
    # 造两个处理时点
    d.loc[(d.firm % 4 == 0), "g"] = 2010
    d.loc[(d.firm % 4 == 2), "g"] = 2014
    d.loc[(d.g > 0), "g"] = d["g"]
    f = did(d, {"id": "d", "y": "y", "cohort": "g", "period": "year",
                "idvar": "firm", "window": [-3, 3]})
    assert f.data["staggered"] is True
    assert any("负权重" in w for w in f.warnings)
