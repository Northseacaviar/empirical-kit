"""机制测试：留一法必须对单点极端值报警。"""
import numpy as np
import pandas as pd

from epkit.checks.leave_one import leave_one_out


def make(outlier=False, seed=2, n=200):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    y = rng.normal(0, 1, n) + 1.0 * g
    if outlier:
        y[g == 1] = y[g == 1]
        y[np.where(g == 1)[0][0]] = 40.0     # 一个极端点
    return pd.DataFrame({"y": y, "g": g})


def test_no_outlier_is_ok():
    f = leave_one_out(make(False), "y", "g")
    assert f.status == "ok"
    assert f.data["relative_shift"] < 0.2


def test_outlier_warns():
    f = leave_one_out(make(True), "y", "g")
    assert f.status == "warn"
    assert f.data["relative_shift"] > 0.2
    assert f.data["worst_obs_group"] == "1"
