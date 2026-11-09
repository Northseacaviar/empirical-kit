"""机制测试：描述统计、相关、VIF 的已知答案。"""
import numpy as np
import pandas as pd

from epkit.checks.desc import corr, describe
from epkit.checks.vif import vif


def make(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(10, 2, n)
    x2 = 2 * x1 + rng.normal(0, 0.01, n)      # 与 x1 几乎完全共线
    x3 = rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3,
                         "g": rng.integers(0, 2, n)})


def test_describe_matches_pandas():
    df = make()
    f = describe(df, ["x1"], fid="d")
    it = f.data["vars"][0]
    assert it["n"] == len(df)
    assert it["mean"] == np.mean(df.x1)
    assert it["median"] == np.median(df.x1)


def test_corr_flags_collinear_pair():
    f = corr(make(), ["x1", "x2", "x3"], fid="c")
    assert f.status == "warn"
    assert {(p["a"], p["b"]) for p in f.data["high_pairs"]} == {("x1", "x2")}


def test_vif_explodes_on_collinear():
    f = vif(make(), ["x1", "x2", "x3"], fid="v")
    v = {i["var"]: i["vif"] for i in f.data["items"]}
    assert v["x1"] > 50 and v["x2"] > 50
    assert v["x3"] < 2
    assert f.status == "warn"


def test_vif_low_for_independent():
    df = make()
    f = vif(df, ["x1", "x3"], fid="v2")
    for i in f.data["items"]:
        assert i["vif"] < 2
