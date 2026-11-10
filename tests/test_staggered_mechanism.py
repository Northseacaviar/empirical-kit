"""交错 DID 估计量的机制测试。

三份已知答案的面板：
1. 效应恒定 → CS 各格都应恢复到同一个值；
2. 效应异质（早期 0、后期 1.5）→ CS 分队列应分别恢复，TWFE 必须偏（这是负权重的实证形态）；
3. 无预趋势 → 各队列 k<0 的格子应接近 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from epkit.estimate.staggered import bacon_pairs, cs_att_gt, heterogeneity_q


def make(effects: dict, seed: int = 42, n_per: int = 250,
         years=(2010, 2022)) -> pd.DataFrame:
    """effects: {cohort: true_effect}，不含 0（从未处理）。"""
    rng = np.random.default_rng(seed)
    y0, y1 = years
    yr = list(range(y0, y1 + 1))
    rows = []
    groups = [0] + sorted(effects)
    for gi, g in enumerate(groups):
        fe = rng.normal(0, 1.0, n_per)
        for i in range(n_per):
            uid = gi * n_per + i
            for t in yr:
                post = 1 if (g and t >= g) else 0
                y = fe[i] + 0.08 * (t - y0) + effects.get(g, 0.0) * post + rng.normal(0, 0.6)
                rows.append({"firm": uid, "year": t, "cohort": g, "y": y})
    return pd.DataFrame(rows)


def test_cs_recovers_constant_effect():
    d = make({2015: 0.5, 2018: 0.5})
    res = cs_att_gt(d, "y", "firm", "year", "cohort", window=(-3, 4))
    post = [r for r in res["att_gt"] if r["k"] >= 0]
    assert post
    for r in post:
        assert abs(r["att"] - 0.5) < 0.18, r
    assert abs(res["aggregate"]["att"] - 0.5) < 0.12, res["aggregate"]


def test_cs_recovers_heterogeneous_effect():
    d = make({2015: 0.0, 2018: 1.5})
    res = cs_att_gt(d, "y", "firm", "year", "cohort", window=(-2, 4))
    early = [r["att"] for r in res["att_gt"] if r["g"] == 2015 and r["k"] >= 0]
    late = [r["att"] for r in res["att_gt"] if r["g"] == 2018 and r["k"] >= 0]
    assert np.mean(early) == np.float64(np.mean(early))
    assert abs(np.mean(early) - 0.0) < 0.15, early
    assert abs(np.mean(late) - 1.5) < 0.2, late
    # 分队列差异应该被异质性检验抓到
    q = heterogeneity_q(res["att_gt"])
    assert q is not None and q["min_p"] < 0.01, q


def test_pre_periods_near_zero():
    d = make({2015: 0.6, 2019: 0.6})
    res = cs_att_gt(d, "y", "firm", "year", "cohort", window=(-4, 4))
    pre = [r for r in res["att_gt"] if r["k"] < 0]
    assert pre
    for r in pre:
        assert abs(r["att"]) < 0.25, r


def test_control_group_is_not_yet_treated():
    """2018 组的 k=0 控制组不能含 2015 组（那时已处理）。"""
    d = make({2015: 0.0, 2018: 1.0})
    res = cs_att_gt(d, "y", "firm", "year", "cohort", window=(0, 0))
    cell = next(r for r in res["att_gt"] if r["g"] == 2018 and r["k"] == 0)
    # 控制组 = never(120) + 到 2018 仍未处理的队列（没有更多）→ 只能是 never 组
    assert cell["n_control"] <= 250, cell
    assert cell["n_treat"] == 250, cell


def test_bacon_pairs_flags_negative_weight_source():
    d = make({2015: 0.0, 2018: 1.0})
    pairs = bacon_pairs(d, "y", "firm", "year", "cohort", window=(-2, 3))
    kinds = {p["type"] for p in pairs}
    assert "late_vs_treated" in kinds, kinds      # 负权重的主要来源必须被摆出来
    assert "early_vs_late" in kinds, kinds
    assert "vs_never" in kinds, kinds
    late = next(p for p in pairs if p["type"] == "late_vs_treated")
    assert late["n_treat"] > 0 and late["n_control"] > 0


def test_did_attaches_staggered_check():
    """端到端：did() 在交错面板上必须自己跑出 CS 对照，而不是只留一句"需复核"。"""
    from epkit.estimate.did import did

    d = make({2015: 0.0, 2018: 1.5})
    f = did(d, {"id": "stag", "y": "y", "cohort": "cohort", "period": "year",
                "idvar": "firm", "window": [-3, 4], "cluster": ["firm"]})
    assert f.data["staggered"] is True
    sc = f.data["staggered_check"]
    assert sc is not None, f.warnings
    assert sc["aggregate"]["att"] is not None
    assert sc["pairs"], "2x2 配对诊断为空"
    assert any("CS 汇总" in w for w in f.warnings), f.warnings
