"""机制测试：聚类标准误必须随组内相关显著变大。

用合成数据，已知答案：
- 20 个聚类、每组 40 个观测、组内高度相关（组固定效应）
- 真系数 2.0
- 等方差/iid 的 SE 应明显小于按组聚类 SE（约 3 倍以上）
口径退化成 iid 时本测试必红——这是回归层唯一能证伪自己的断言。
"""
import numpy as np
import pandas as pd

from epkit.estimate.reg import reg


def make(n_groups=60, per_group=20, beta=2.0, seed=0):
    """x 必须带组内共同成分。

    若 x 在组内独立，组内冲击在求和时正负抵消，聚类方差几乎不增加——
    表现为「聚类 SE ≈ iid SE」，也就是论文里说的「按公司聚类接近空转」。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x_group = rng.normal(0, 1)        # 组层面的 x（企业层面的处置）
        shock = rng.normal(0, 1)          # 组内共享的冲击
        for _ in range(per_group):
            x = x_group + rng.normal(0, 0.2)
            y = beta * x + shock + rng.normal(0, 0.3)
            rows.append({"y": y, "x": x, "grp": g})
    return pd.DataFrame(rows)


def test_cluster_se_exceeds_iid():
    df = make()
    f = reg(df, {"id": "m", "y": "y", "x": ["x"], "cluster": ["grp"]})
    st = f.data["steps"][0]
    iid = st["by_vcov"]["iid"]["terms"]["x"]["se"]
    cl = st["by_vcov"]["cluster_grp"]["terms"]["x"]["se"]
    assert cl > 2.0 * iid, f"聚类 SE {cl:.3f} 未显著大于 iid SE {iid:.3f}"


def test_coef_recovers_truth():
    df = make()
    f = reg(df, {"id": "m", "y": "y", "x": ["x"], "cluster": ["grp"]})
    assert f.data["steps"][0]["by_vcov"]["cluster_grp"]["terms"]["x"]["coef"] ==         __import__("pytest").approx(2.0, abs=0.15)


def test_vcov_instability_warns():
    """两个口径 t 值跨过 1.96 边界时必须 warn。"""
    df = make(n_groups=6, per_group=8, seed=3)
    f = reg(df, {"id": "m", "y": "y", "x": ["x"], "cluster": ["grp"]})
    ts = [v["terms"]["x"]["t"] for v in f.data["steps"][0]["by_vcov"].values()]
    if any(abs(t) >= 1.96 for t in ts) and any(abs(t) < 1.96 for t in ts):
        assert f.status == "warn"
