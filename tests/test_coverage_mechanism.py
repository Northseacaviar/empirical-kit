"""规格曲线、子样本网格、分位数回归的机制测试。

三份都是"已知答案"的合成数据：真效应为 0 时不能报显著，
异质效应必须被分出来，分位效应必须随分位单调。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from epkit.checks.quantile import quantile
from epkit.checks.spec_curve import spec_curve
from epkit.checks.subsample import subsample


def panel(n=1500, effect=1.0, seed=1, hetero: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    ind = rng.integers(0, 8, n)
    yr = rng.integers(2015, 2021, n)
    c1 = rng.normal(size=n)
    c2 = rng.normal(size=n)
    gid = rng.integers(0, 100, n)
    eff = np.full(n, float(effect))
    if hetero:
        for k, v in hetero.items():
            eff = np.where(ind == k, v, eff)
    y = eff * x + 0.5 * c1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "c1": c1, "c2": c2,
                         "ind": ind, "year": yr, "gid": gid})


CFG = {"y": "y", "x": "x", "controls": [[], ["c1"], ["c1", "c2"]],
       "fe": [[], ["ind"], ["ind", "year"]],
       "vcov": ["hetero", "cluster:gid"]}


def test_spec_curve_detects_null():
    # 真效应 0：多数设定不应显著，工具应报"对设定敏感"
    f = spec_curve(panel(effect=0.0, seed=2), dict(CFG, id="null"))
    assert f.data["summary"]["n_specs"] == 18   # 3 控制组 × 3 FE 组合 × 2 推断口径
    assert f.data["summary"]["share_sig"] < 0.5, f.data["summary"]
    assert f.status == "warn"


def test_spec_curve_recovers_true_effect():
    f = spec_curve(panel(effect=1.0, seed=3), dict(CFG, id="real"))
    s = f.data["summary"]
    assert abs(s["median"] - 1.0) < 0.15, s
    assert s["share_sig"] > 0.9, s
    assert abs(s["share_positive"] - 1.0) < 1e-9


def test_spec_curve_respects_max_specs():
    f = spec_curve(panel(), dict(CFG, id="big", max_specs=3))
    assert f.status == "fail"
    assert "超过上限" in f.why


def test_subsample_recovers_group_effects():
    # ind==0 效应 1.5，其余 0.2
    d = panel(effect=0.2, hetero={0: 1.5}, seed=4)
    f = subsample(d, {"id": "sub", "y": "y", "x": "x", "by": "ind", "min_n": 30})
    rows = {r["group"]: r for r in f.data["rows"]}
    assert abs(rows["0"]["coef"] - 1.5) < 0.25, rows["0"]
    assert abs(rows["1"]["coef"] - 0.2) < 0.25, rows["1"]
    assert f.status == "ok"


def test_subsample_warns_on_sign_flip():
    d = panel(effect=1.0, seed=5)
    d.loc[d.ind == 3, "y"] = -0.8 * d.loc[d.ind == 3, "x"] + 0.5 * d.loc[d.ind == 3, "c1"]
    f = subsample(d, {"id": "sub", "y": "y", "x": "x", "by": "ind", "min_n": 30})
    assert f.status == "warn"
    assert any("符号翻转" in w for w in f.warnings), f.warnings


def test_subsample_reports_thin_groups():
    d = panel(n=400, seed=6)
    d = pd.concat([d, d.iloc[:5].assign(ind=99)])   # 造一个只有 5 个观测的组
    f = subsample(d, {"id": "sub", "y": "y", "x": "x", "by": "ind", "min_n": 30})
    thin = [r for r in f.data["rows"] if r.get("coef") is None]
    assert thin and thin[0]["group"] == "99", f.data["rows"]


def test_quantile_recovers_slope_by_quantile():
    # y = u·x，u~U(0,1)，x>0 → 条件分位 Q_q(y|x)=q·x，斜率必须随分位从 0 走到 1。
    # （用对称的 x 会把这个结构抹平：正负两侧互相抵消，斜率全落在 1 附近）
    rng = np.random.default_rng(7)
    n = 4000
    x = np.exp(rng.normal(size=n))
    u = rng.uniform(size=n)
    y = u * x
    d = pd.DataFrame({"y": y, "x": x})
    f = quantile(d, {"id": "qr", "y": "y", "x": "x", "q": [0.1, 0.5, 0.9]})
    coefs = [r["coef"] for r in f.data["rows"]]
    assert coefs[0] < coefs[1] < coefs[2], coefs
    assert abs(coefs[0] - 0.1) < 0.06, coefs
    assert abs(coefs[2] - 0.9) < 0.06, coefs
    assert f.status == "ok"


def test_quantile_with_fe_flags_approximation():
    f = quantile(panel(seed=8), {"id": "qr2", "y": "y", "x": "x", "fe": ["ind"],
                                 "q": [0.25, 0.5, 0.75]})
    assert f.data["approximation"] == "within-demeaning"
    assert any("去均值" in w for w in f.warnings), f.warnings
