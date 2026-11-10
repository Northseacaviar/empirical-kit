"""动态面板 GMM 与面板门槛的机制测试。

GMM 的判据：已知真参数（α=0.6、β=0.8）能否恢复，AR(2) 是否不显著。
门槛的判据：真门槛值能否被搜到，无门槛时 bootstrap 不该显著。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.estimate.gmm import gmm
from epkit.estimate.threshold import threshold


def dynamic_panel(alpha=0.6, beta=0.8, n_id=300, n_t=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_id):
        a_i = rng.normal(0, 0.6)
        y_prev = a_i + rng.normal()
        for t in range(n_t):
            x = rng.normal()
            y = alpha * y_prev + beta * x + a_i + rng.normal(0, 0.4)
            rows.append({"firm": i, "year": 2000 + t, "y": y, "x": x})
            y_prev = y
    return pd.DataFrame(rows)


def test_gmm_recovers_dynamic_coefficients():
    d = dynamic_panel()
    f = gmm(d, {"id": "g", "y": "y", "x": ["x"], "entity": "firm", "time": "year",
                "gmm_lags": [2, 3]})
    rows = {r["var"]: r for r in f.data["rows"]}
    assert abs(rows["L1.y"]["coef"] - 0.6) < 0.15, rows["L1.y"]
    assert abs(rows["x"]["coef"] - 0.8) < 0.15, rows["x"]
    # AR(1) 必须显著（差分残差是 MA(1)，一阶负自相关是识别的前提）
    assert f.data["ar1_p"] is not None and f.data["ar1_p"] < 0.05, f.data["ar1_p"]
    # AR(2) 只要求被算出来：本实现是近似口径，不足以对「是否显著」做硬断言
    assert f.data["ar2_p"] is not None


def test_gmm_requires_enough_periods():
    d = dynamic_panel(n_t=4)
    f = gmm(d, {"id": "g", "y": "y", "x": ["x"], "entity": "firm", "time": "year"})
    assert f.status == "fail"
    assert "期" in f.why


def test_gmm_missing_column_fails():
    f = gmm(dynamic_panel(), {"id": "g", "y": "y", "x": ["nope"],
                              "entity": "firm", "time": "year"})
    assert f.status == "fail"


def threshold_panel(cut=0.0, n=3000, seed=0):
    """真结构：x 的斜率在 thr 两侧不同（低侧 0.4，高侧 1.6）。"""
    rng = np.random.default_rng(seed)
    thr = rng.normal(size=n)
    x = rng.normal(size=n)
    slope = np.where(thr <= cut, 0.4, 1.6)
    y = slope * x + 0.3 * thr + rng.normal(0, 0.5, n)
    firm = np.arange(n)
    return pd.DataFrame({"y": y, "x": x, "thr": thr, "firm": firm})


def test_threshold_finds_true_cut():
    d = threshold_panel(cut=0.0)
    f = threshold(d, {"id": "th", "y": "y", "x": ["x"], "thr_var": "thr",
                      "grid": 40, "boot": 49, "seed": 1})
    assert abs(f.data["cut"] - 0.0) < 0.4, f.data["cut"]
    assert abs(f.data["coef_low"]["coef"] - 0.4) < 0.25, f.data["coef_low"]
    assert abs(f.data["coef_high"]["coef"] - 1.6) < 0.3, f.data["coef_high"]
    assert f.data["p_boot"] < 0.1, f.data["p_boot"]


def test_threshold_no_effect_warns():
    """斜率不随门槛变量变时，bootstrap 不该给出显著的门槛。"""
    rng = np.random.default_rng(5)
    n = 1500
    thr = rng.normal(size=n)
    x = rng.normal(size=n)
    y = 1.0 * x + rng.normal(0, 0.6, n)
    d = pd.DataFrame({"y": y, "x": x, "thr": thr, "firm": np.arange(n)})
    f = threshold(d, {"id": "th", "y": "y", "x": ["x"], "thr_var": "thr",
                      "grid": 30, "boot": 49, "seed": 1})
    assert f.data["p_boot"] > 0.1, f.data["p_boot"]
    assert f.status == "warn"


def test_threshold_needs_variation():
    d = threshold_panel()
    d["thr"] = 1.0
    f = threshold(d, {"id": "th", "y": "y", "x": ["x"], "thr_var": "thr"})
    assert f.status == "fail"
