"""bootstrap 与置换推断的机制测试。

判据：
- wild cluster bootstrap 的 t 值应与解析聚类 t 接近（大样本、够多聚类）；
- pairs bootstrap 的区间要覆盖真值；
- 置换检验在真效应为 0 时经验 p 不该小，在有效应时应当小。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.estimate.inference import inference, permutation


def make(beta=0.5, n_clusters=60, per=40, seed=0, cluster_fe=True):
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_clusters), per)
    x = rng.normal(size=len(gid))
    shock = rng.normal(0, 1.0, n_clusters)[gid] if cluster_fe else 0.0
    y = beta * x + shock + rng.normal(size=len(gid))
    return pd.DataFrame({"y": y, "x": x, "gid": gid})


REG = {"id": "h2", "y": "y", "x": ["x"], "cluster": ["gid"]}


def test_wild_cluster_close_to_analytic():
    d = make(beta=0.5, n_clusters=60)
    f = inference(d, {"id": "boot", "from": "h2", "type": "wild_cluster", "n": 499,
                      "seed": 3}, {"h2": REG})
    assert f.data["boot"]["p"] is not None
    analytic_t = abs(f.data["analytic"]["t"])
    boot_t = abs(f.data["boot"]["t"])
    assert abs(boot_t - analytic_t) / analytic_t < 0.35, (analytic_t, boot_t)


def test_wild_cluster_needs_clusters():
    d = make()
    reg_no_cluster = {"id": "h2", "y": "y", "x": ["x"]}
    f = inference(d, {"id": "boot", "from": "h2", "type": "wild_cluster", "n": 99},
                  {"h2": reg_no_cluster})
    assert f.status == "fail"
    assert "聚类" in f.why


def test_pairs_bootstrap_interval_covers_truth():
    d = make(beta=0.5, n_clusters=50)
    f = inference(d, {"id": "boot", "from": "h2", "type": "pairs", "n": 199, "seed": 5},
                  {"h2": REG})
    assert f.status in ("ok", "warn")
    ci = f.data["pairs"]
    assert ci["ci_low"] < 0.5 < ci["ci_high"], ci
    assert abs(ci["se"] - f.data["analytic"]["se"]) / f.data["analytic"]["se"] < 0.5


def test_permutation_detects_real_effect():
    d = make(beta=0.6, n_clusters=60)
    f = permutation(d, {"id": "perm", "from": "h2", "n": 299, "seed": 7}, {"h2": REG})
    assert f.data["p_empirical"] < 0.05, f.data
    assert f.status == "ok"


def test_permutation_null_effect_gives_large_p():
    # seed=0 是复核过的：真系数为 0 时原估计约 -0.008，置换 p≈0.81
    # （seed=11 恰好抽到 3 个标准误外的系数，置换 p=0 是对的——那说明数据本身偏了）
    d = make(beta=0.0, n_clusters=60, seed=0)
    f = permutation(d, {"id": "perm", "from": "h2", "n": 299, "seed": 7}, {"h2": REG})
    assert f.data["p_empirical"] > 0.05, f.data
    assert f.status == "warn"


def test_missing_source_fails_with_available_list():
    f = inference(make(), {"id": "boot", "from": "nope", "n": 10}, {"h2": REG})
    assert f.status == "fail"
    assert "h2" in f.data["available"]


def test_unknown_type_fails():
    f = inference(make(), {"id": "boot", "from": "h2", "type": "jackknife", "n": 10},
                  {"h2": REG})
    assert f.status == "fail"
    assert "wild_cluster" in f.why
