"""Tobit 的机制测试：截尾数据要能恢复真参数，而 OLS 在同一份数据上应当有偏。

这是判据的关键：只断「能跑」没有意义，要断「比 OLS 更接近真值」——
否则一个退化成 OLS 的实现也能通过。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.estimate.tobit import tobit


def censored(n=4000, beta=1.5, sigma=1.0, seed=0, lower=0.0, upper=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(1.0, 1.0, n)          # 均值大于 0，才会有约一半落在下界
    ystar = 0.5 + beta * x + rng.normal(0, sigma, n)
    y = np.maximum(ystar, lower) if lower is not None else ystar
    if upper is not None:
        y = np.minimum(y, upper)
    ind = rng.integers(0, 5, n)
    return pd.DataFrame({"y": y, "x": x, "ind": ind, "ystar": ystar})


def test_tobit_recovers_slope_on_left_censored_data():
    d = censored()
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"], "tobit": {"lower": 0}})
    row = next(r for r in f.data["rows"] if r["var"] == "x")
    assert abs(row["coef"] - 1.5) < 0.12, row
    assert f.data["n_left"] > 200


def test_ols_is_biased_where_tobit_is_not():
    """同一份截尾数据：OLS 的斜率会被压向 0，Tobit 不会。"""
    d = censored()
    import statsmodels.api as sm

    ols = sm.OLS(d["y"], sm.add_constant(d[["x"]])).fit()
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"], "tobit": {"lower": 0}})
    tb = next(r for r in f.data["rows"] if r["var"] == "x")["coef"]
    assert abs(tb - 1.5) < abs(float(ols.params["x"]) - 1.5), (tb, ols.params["x"])


def test_two_sided_censoring():
    d = censored(beta=1.2, lower=0.0, upper=3.0)
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"],
                  "tobit": {"lower": 0, "upper": 3}})
    row = next(r for r in f.data["rows"] if r["var"] == "x")
    assert abs(row["coef"] - 1.2) < 0.2, row
    assert f.data["n_right"] > 0


def test_low_dim_fe_allowed():
    d = censored()
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"], "fe": ["ind"],
                  "tobit": {"lower": 0}})
    row = next(r for r in f.data["rows"] if r["var"] == "x")
    assert abs(row["coef"] - 1.5) < 0.15, row


def test_high_dim_fe_refused():
    d = censored(n=600)
    d["firm"] = np.arange(len(d))            # 每个观测一个水平
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"], "fe": ["firm"],
                  "tobit": {"lower": 0}})
    assert f.status == "fail"
    assert "高维固定效应" in f.why


def test_no_censoring_warns_to_use_ols():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"y": rng.normal(5, 1, 500), "x": rng.normal(size=500)})
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"], "tobit": {"lower": 0}})
    assert f.status == "warn"
    assert "没有删失" in f.why


def test_warns_about_iid_standard_errors():
    d = censored()
    f = tobit(d, {"id": "tb", "y": "y", "x": ["x"], "tobit": {"lower": 0}})
    assert any("iid" in w for w in f.warnings), f.warnings
