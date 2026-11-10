"""中介效应与跨组系数差异的机制测试。

判据是「能不能区分有中介和没中介」「能不能区分两组一样和不一样」，
以及 bootstrap 区间在真值为 0 时不该不含 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.checks.group_test import group_test
from epkit.checks.mediation import mediation


def dgp(n=2500, a=0.5, b=0.4, cprime=0.2, seed=0):
    """真结构：x → m（a）、m → y（b）、x → y 直接（cprime）。总效应 c = a*b + cprime。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = a * x + rng.normal(size=n)
    y = b * m + cprime * x + rng.normal(size=n)
    gid = rng.integers(0, 100, n)
    return pd.DataFrame({"y": y, "x": x, "m": m, "gid": gid})


def test_mediation_recovers_ab_and_ci_excludes_zero():
    d = dgp()
    f = mediation(d, {"id": "med", "y": "y", "x": "x", "m": ["m"], "boot": 199,
                      "cluster": ["gid"], "seed": 7})
    r = f.data["rows"][0]
    assert abs(r["a"]["coef"] - 0.5) < 0.1, r["a"]
    assert abs(r["b"]["coef"] - 0.4) < 0.05, r["b"]
    assert abs(r["c_prime"]["coef"] - 0.2) < 0.05, r["c_prime"]
    assert abs(r["ab"] - 0.2) < 0.06, r["ab"]
    assert r["ci_low"] > 0, (r["ci_low"], r["ci_high"])       # CI 不含 0
    assert f.status == "ok"


def test_mediation_null_path_ci_contains_zero():
    d = dgp(a=0.0, b=0.4, cprime=1.0, seed=3)                 # x 对 m 无影响 → 无中介
    f = mediation(d, {"id": "med", "y": "y", "x": "x", "m": ["m"], "boot": 199,
                      "cluster": ["gid"], "seed": 7})
    r = f.data["rows"][0]
    assert r["ci_low"] < 0 < r["ci_high"], (r["ci_low"], r["ci_high"])
    assert any("含 0" in w for w in f.warnings), f.warnings


def test_mediation_warns_against_stepwise_only():
    d = dgp()
    f = mediation(d, {"id": "med", "y": "y", "x": "x", "m": ["m"], "boot": 49, "seed": 1})
    assert any("逐步" in w for w in f.warnings), f.warnings


def test_mediation_reports_share():
    d = dgp()
    f = mediation(d, {"id": "med", "y": "y", "x": "x", "m": ["m"], "boot": 49, "seed": 1})
    r = f.data["rows"][0]
    assert 0 < r["share"] < 1, r["share"]


def test_mediation_missing_column_fails(tmp_path):
    d = dgp(200)
    f = mediation(d, {"id": "med", "y": "y", "x": "x", "m": ["nope"]})
    assert f.status == "fail"
    assert "nope" in f.why


# —— 跨组系数差异 ——

def group_dgp(beta_low=0.5, beta_high=0.5, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    soe = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = np.where(soe == 1, beta_high, beta_low) * x + rng.normal(size=n)
    gid = rng.integers(0, 80, n)
    return pd.DataFrame({"y": y, "x": x, "soe": soe, "gid": gid})


def test_group_test_detects_difference():
    d = group_dgp(beta_low=0.5, beta_high=1.5)
    f = group_test(d, {"id": "chow", "y": "y", "x": ["x"], "by": "soe", "cluster": ["gid"]})
    assert f.data["p_f"] < 0.01, f.data
    assert abs(f.data["coef_low"] - 0.5) < 0.1
    assert abs(f.data["coef_high"] - 1.5) < 0.1
    assert f.status == "warn"


def test_group_test_finds_no_difference():
    d = group_dgp(beta_low=0.5, beta_high=0.5)
    f = group_test(d, {"id": "chow", "y": "y", "x": ["x"], "by": "soe", "cluster": ["gid"]})
    assert f.data["p_f"] > 0.05, f.data
    assert f.status == "ok"


def test_group_test_needs_two_groups():
    d = group_dgp()
    d["soe"] = np.random.default_rng(1).integers(0, 3, len(d))
    f = group_test(d, {"id": "chow", "y": "y", "x": ["x"], "by": "soe"})
    assert f.status == "fail"
    assert "只做两组" in f.why
