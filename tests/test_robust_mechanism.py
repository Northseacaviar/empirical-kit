"""机制测试：缩尾与剔除必须真的改变样本/分布，而不是静默无效。"""
import numpy as np
import pandas as pd

from epkit.checks.group import group_diff


def make(seed=5, n=400):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n) + 0.5 * g
    x[0] = 50.0            # 一个极端值
    return pd.DataFrame({"y": x, "g": g, "big": rng.integers(0, 2, n)}), x


def test_winsor_shrinks_extremes():
    df, raw = make()
    f = group_diff(df, "y", "g", winsor=[0.01, 0.99])
    lo, hi = f.data["ci95_welch"]
    assert abs(f.data["diff"]) < abs(raw[df.g == 1].mean() - raw[df.g == 0].mean())
    assert f.data["winsor"] == [0.01, 0.99]


def test_drop_where_removes_rows():
    df, _ = make()
    f = group_diff(df, "y", "g", drop_where="big == 1")
    assert f.data["n_before_robustness"] == len(df)
    assert sum(g["n"] for g in f.data["groups"]) < len(df)


def test_unknown_column_fails_loudly():
    df, _ = make()
    f = group_diff(df, "nope", "g")
    assert f.status == "fail"
