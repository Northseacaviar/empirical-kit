"""PSM、RDD、合成控制、Heckman 的机制测试。

四块都用「已知答案」的数据：真 ATT 已知、断点效应已知、选择偏差构造得出来。
只断「跑通」没有意义——这四个方法最危险的失败模式都是「给出了一个像模像样的数」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.estimate.heckman import heckman
from epkit.estimate.psm import psm
from epkit.estimate.rdd import rdd
from epkit.estimate.synth import synth


# —— PSM ——

def psm_dgp(att=2.0, n=2000, seed=0):
    """处理分配依赖 x（选择偏差真实存在），处理效应恒为 att。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(0.8 * x1 + 0.5 * x2)))
    t = (rng.random(n) < ps).astype(int)
    y = att * t + 1.5 * x1 + 1.0 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_psm_recovers_att_and_balances_covariates():
    d = psm_dgp(att=2.0)
    f = psm(d, {"id": "psm", "treat": "t", "y": "y", "covariates": ["x1", "x2"],
                "caliper": 0.1, "boot": 99, "seed": 1})
    assert abs(f.data["att"] - 2.0) < 0.35, f.data["att"]
    # 匹配后协变量应当比匹配前更平衡
    for c in ("x1", "x2"):
        assert abs(f.data["balance_after"][c]) <= abs(f.data["balance_before"][c]) + 1e-9
    assert f.data["n_matched"] > 100


def test_psm_raw_att_is_biased_where_matched_is_not():
    """不做匹配直接比均值会被选择偏差污染——这是 PSM 存在的理由。"""
    d = psm_dgp(att=2.0)
    naive = float(d.loc[d.t == 1, "y"].mean() - d.loc[d.t == 0, "y"].mean())
    f = psm(d, {"id": "psm", "treat": "t", "y": "y", "covariates": ["x1", "x2"],
                "caliper": 0.1, "boot": 49, "seed": 1})
    assert abs(naive - 2.0) > 0.3                      # 朴素差确实偏
    assert abs(f.data["att"] - 2.0) < abs(naive - 2.0)


def test_psm_tight_caliper_reports_common_support_problem():
    d = psm_dgp(att=1.0, seed=2)
    f = psm(d, {"id": "psm", "treat": "t", "y": "y", "covariates": ["x1", "x2"],
                "caliper": 0.0001, "boot": 20, "seed": 1})
    assert f.data["n_unmatched"] > 0
    assert any("共同支撑" in w for w in f.warnings), f.warnings


def test_psm_needs_binary_treatment():
    d = psm_dgp()
    d["t"] = d["t"] + d["t"] * 2
    f = psm(d, {"id": "psm", "treat": "t", "y": "y", "covariates": ["x1"]})
    assert f.status == "fail"


# —— RDD ——

def rdd_dgp(jump=1.0, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = jump * (x >= 0) + 0.6 * x + rng.normal(0, 0.25, n)
    return pd.DataFrame({"y": y, "score": x})


def test_rdd_recovers_jump():
    f = rdd(rdd_dgp(jump=1.0), {"id": "rd", "y": "y", "x": "score", "cut": 0.0})
    assert abs(f.data["coef_robust"] - 1.0) < 0.25, f.data
    assert f.data["p_robust"] < 0.01
    assert f.data["bw_left"] > 0


def test_rdd_no_jump_is_not_significant():
    f = rdd(rdd_dgp(jump=0.0, seed=5), {"id": "rd", "y": "y", "x": "score", "cut": 0.0})
    assert f.data["p_robust"] > 0.05, f.data
    assert f.status == "warn"


def test_rdd_missing_column_fails():
    f = rdd(pd.DataFrame({"y": [1, 2]}), {"id": "rd", "y": "y", "x": "nope"})
    assert f.status == "fail"
    assert "nope" in f.why


# —— 合成控制 ——

def synth_dgp(att=2.0, t0=2015, n_donors=8, seed=0):
    rng = np.random.default_rng(seed)
    years = list(range(2010, 2021))
    common = {t: rng.normal() for t in years}
    rows = []
    base = rng.normal(5, 0.5)
    for u in range(n_donors + 1):
        unit = "treated" if u == 0 else f"d{u}"
        offset = base if u == 0 else rng.normal(5, 0.6)
        for t in years:
            val = offset + 0.8 * common[t] + rng.normal(0, 0.2)
            if unit == "treated" and t >= t0:
                val += att
            rows.append({"unit": unit, "year": t, "y": val})
    return pd.DataFrame(rows)


def test_synth_recovers_att():
    d = synth_dgp(att=2.0)
    f = synth(d, {"id": "sc", "y": "y", "unit": "unit", "time": "year",
                  "treated_unit": "treated", "treat_time": 2015})
    assert abs(f.data["att_post"] - 2.0) < 0.6, f.data
    assert f.data["rmspe_pre"] < f.data["rmspe_post"]
    assert sum(f.data["weights"].values()) == pytest.approx(1.0)


def test_synth_placebo_flags_fake_effect():
    d = synth_dgp(att=0.0, seed=3)
    f = synth(d, {"id": "sc", "y": "y", "unit": "unit", "time": "year",
                  "treated_unit": "treated", "treat_time": 2015})
    assert f.data["p_rank"] > 0.1, f.data
    assert f.status == "warn"


def test_synth_needs_pre_periods():
    d = synth_dgp()
    f = synth(d, {"id": "sc", "y": "y", "unit": "unit", "time": "year",
                  "treated_unit": "treated", "treat_time": 2010})
    assert f.status == "fail"
    assert "3 期" in f.why


# —— Heckman ——

def heckman_dgp(selection=True, n=4000, seed=0):
    """selection=True 时，未观测的选择与结果误差相关 → 存在选择偏差。"""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    u = rng.normal(size=n)
    obs = (0.8 * z + u * (1.0 if selection else 0.0) + rng.normal(size=n)) > 0
    y = 1.0 * x + 2.0 + u * 1.5 + rng.normal(size=n)
    return pd.DataFrame({"y": np.where(obs, y, np.nan), "x": x, "z": z,
                         "observed": obs.astype(int)})


def test_heckman_detects_selection_bias():
    d = heckman_dgp(selection=True)
    f = heckman(d, {"id": "hk", "y": "y", "x": ["x"], "select": ["z"],
                    "observed": "observed"})
    lam = f.data["lambda"]
    assert lam["p"] < 0.05, lam
    assert any("选择偏差" in w for w in f.warnings), f.warnings


def test_heckman_finds_no_bias_when_none():
    d = heckman_dgp(selection=False, seed=4)
    f = heckman(d, {"id": "hk", "y": "y", "x": ["x"], "select": ["z"],
                    "observed": "observed"})
    assert f.data["lambda"]["p"] > 0.05, f.data["lambda"]


def test_heckman_recovers_slope():
    d = heckman_dgp(selection=True)
    f = heckman(d, {"id": "hk", "y": "y", "x": ["x"], "select": ["z"],
                    "observed": "observed"})
    row = next(r for r in f.data["rows"] if r["var"] == "x")
    assert abs(row["coef"] - 1.0) < 0.15, row
