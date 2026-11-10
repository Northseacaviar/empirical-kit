"""离散因变量的机制测试：系数能不能被恢复、边际效应算得对不对、口径有没有漂。

已知答案的做法：按真参数生成 logit/probit/poisson，检查恢复值；
再用 statsmodels 在同一份数据上跑一遍对照（无 FE 时应当一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.estimate.reg import reg


def dgp(kind="logit", n=4000, beta=0.8, seed=0, fe_strength=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    ind = rng.integers(0, 6, n)
    gid = rng.integers(0, 100, n)
    fe = rng.normal(0, fe_strength, 6)[ind] if fe_strength else np.zeros(n)
    lp = -0.4 + beta * x + fe
    if kind == "logit":
        y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    elif kind == "probit":
        from scipy.stats import norm
        y = (rng.random(n) < norm.cdf(lp)).astype(int)
    else:  # poisson
        y = rng.poisson(np.exp(lp))
    return pd.DataFrame({"y": y, "x": x, "ind": ind, "gid": gid})


def test_logit_recovers_coefficient():
    f = reg(dgp("logit"), {"id": "l", "y": "y", "x": ["x"], "model": "logit",
                           "cluster": ["gid"]})
    assert f.data["model"] == "logit"
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    coef = st["by_vcov"]["cluster_gid"]["terms"]["x"]["coef"]
    assert abs(coef - 0.8) < 0.08, coef


def test_logit_matches_statsmodels_without_fe():
    import statsmodels.api as sm

    d = dgp("logit", n=3000, seed=3)
    f = reg(d, {"id": "l", "y": "y", "x": ["x"], "model": "logit"})
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    ours = st["by_vcov"][f.data["main_vcov"]]["terms"]["x"]["coef"]

    m = sm.Logit(d["y"], sm.add_constant(d[["x"]])).fit(disp=0)
    assert ours == pytest.approx(float(m.params["x"]), abs=1e-3), (ours, m.params)


def test_logit_with_fixed_effects():
    d = dgp("logit", beta=0.7, fe_strength=1.5, seed=4)
    f = reg(d, {"id": "l", "y": "y", "x": ["x"], "model": "logit",
                "fe": ["ind"], "cluster": ["gid"]})
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    coef = st["by_vcov"]["cluster_gid"]["terms"]["x"]["coef"]
    assert abs(coef - 0.7) < 0.15, coef


def test_probit_recovers_coefficient():
    f = reg(dgp("probit"), {"id": "p", "y": "y", "x": ["x"], "model": "probit",
                            "cluster": ["gid"]})
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    coef = st["by_vcov"]["cluster_gid"]["terms"]["x"]["coef"]
    assert abs(coef - 0.8) < 0.1, coef


def test_poisson_recovers_coefficient():
    f = reg(dgp("poisson"), {"id": "pp", "y": "y", "x": ["x"], "model": "poisson",
                             "cluster": ["gid"]})
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    coef = st["by_vcov"]["cluster_gid"]["terms"]["x"]["coef"]
    assert abs(coef - 0.8) < 0.08, coef


def test_logit_marginal_effect_is_probability_not_slope():
    d = dgp("logit", seed=5)
    f = reg(d, {"id": "l", "y": "y", "x": ["x"], "model": "logit"})
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    coef = st["by_vcov"][f.data["main_vcov"]]["terms"]["x"]["coef"]
    dy = f.data["marginals"]["x"]["dy_dx"]
    # 平均概率约 0.4，边际效应应约为系数的 0.2 倍量级，绝不该等于系数
    assert 0 < dy < coef, (dy, coef)
    assert dy / coef == pytest.approx(0.2, abs=0.08), dy / coef


def test_pseudo_r2_reported():
    f = reg(dgp("logit", seed=6), {"id": "l", "y": "y", "x": ["x"], "model": "logit"})
    assert 0 < f.data["pseudo_r2"] < 1, f.data.get("pseudo_r2")


def test_ols_path_still_works():
    rng = np.random.default_rng(7)
    d = pd.DataFrame({"y": rng.normal(size=800), "x": rng.normal(size=800),
                      "gid": rng.integers(0, 50, 800)})
    f = reg(d, {"id": "o", "y": "y", "x": ["x"], "cluster": ["gid"]})
    st = next(s for s in f.data["steps"] if "by_vcov" in s)
    assert abs(st["by_vcov"]["cluster_gid"]["terms"]["x"]["coef"]) < 0.2


def test_unknown_model_fails_loudly():
    f = reg(dgp(), {"id": "bad", "y": "y", "x": ["x"], "model": "heckman"})
    assert f.status == "fail"
    assert "heckman" in f.why


def test_iv_with_logit_is_refused():
    f = reg(dgp(), {"id": "bad2", "y": "y", "x": [], "model": "logit",
                    "iv": {"endog": ["x"], "instruments": ["ind"]}})
    assert f.status == "fail"
    assert "IV" in f.why


def test_negative_binomial_recovers_coefficient():
    rng = np.random.default_rng(11)
    n = 3000
    x = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x)
    y = rng.poisson(rng.gamma(2, mu / 2))         # 过散布计数
    gid = rng.integers(0, 80, n)
    d = pd.DataFrame({"y": y, "x": x, "gid": gid})
    f = reg(d, {"id": "nb", "y": "y", "x": ["x"], "model": "nb", "cluster": ["gid"]})
    step = f.data["steps"][0]
    coef = step["by_vcov"][f.data["main_vcov"]]["terms"]["x"]["coef"]
    assert abs(coef - 0.5) < 0.1, coef
    assert f.data["alpha"] > 0


def test_negative_binomial_refuses_fixed_effects():
    d = dgp("poisson")
    f = reg(d, {"id": "nb", "y": "y", "x": ["x"], "model": "nb", "fe": ["ind"]})
    assert f.status == "fail"
    assert "固定效应" in f.why
