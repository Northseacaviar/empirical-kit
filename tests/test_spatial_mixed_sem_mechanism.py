"""空间计量、多层模型、结构方程的机制测试。

三块都以「已知参数能否恢复」为判据：
- 空间：生成含 rho 的 SAR 数据，估计的 rho 应接近真值；
- 多层：生成组间方差已知的数据，ICC 应接近理论值；
- SEM：生成已知路径系数的数据，系数应接近真值且拟合指标达标。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.estimate.mixed import mixed
from epkit.estimate.sem import sem
from epkit.estimate.spatial import spatial


# —— 空间计量 ——

def sar_data(rho=0.5, beta=1.0, k=8, seed=0, w_path=None):
    """k×k 网格上的 SAR：y = rho*W y + beta*x + e。"""
    rng = np.random.default_rng(seed)
    n = k * k
    labels = [f"u{i:02d}" for i in range(n)]
    mat = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            idx = i * k + j
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < k and 0 <= nj < k:
                    mat[idx, ni * k + nj] = 1.0
    wdf = pd.DataFrame(mat, index=labels, columns=labels)
    wdf.to_csv(w_path, encoding="utf-8-sig")
    W = mat / mat.sum(axis=1, keepdims=True)
    x = rng.normal(size=n)
    y = np.linalg.inv(np.eye(n) - rho * W) @ (beta * x + rng.normal(0, 0.5, n))
    return pd.DataFrame({"unit": labels, "x": x, "y": y})


def test_spatial_recovers_rho(tmp_path):
    w_path = tmp_path / "w.csv"
    d = sar_data(rho=0.5, beta=1.0, w_path=str(w_path))
    f = spatial(d, {"id": "sar", "y": "y", "x": ["x"], "unit": "unit",
                    "w": str(w_path), "model": "lag"})
    assert f.status in ("ok", "warn")
    assert abs(f.data["rho"] - 0.5) < 0.2, f.data["rho"]
    xrow = next(r for r in f.data["rows"] if r["var"] == "x")
    assert abs(xrow["coef"] - 1.0) < 0.2, xrow


def test_spatial_requires_matrix_labels_match(tmp_path):
    w_path = tmp_path / "w.csv"
    d = sar_data(w_path=str(w_path))
    bad = pd.read_csv(w_path, index_col=0)
    bad.index = [f"v{i}" for i in range(len(bad))]      # 标签对不上
    bad.to_csv(w_path, encoding="utf-8-sig")
    f = spatial(d, {"id": "sar", "y": "y", "x": ["x"], "unit": "unit",
                    "w": str(w_path)})
    assert f.status == "fail"
    assert "标签" in f.why


def test_spatial_ols_model_runs(tmp_path):
    w_path = tmp_path / "w.csv"
    d = sar_data(rho=0.0, w_path=str(w_path))
    f = spatial(d, {"id": "ols", "y": "y", "x": ["x"], "unit": "unit",
                    "w": str(w_path), "model": "ols"})
    assert f.data["model"] == "ols"
    assert f.data["rows"]


# —— 多层模型 ——

def hlm_data(icc_target=0.4, n_group=30, per=40, seed=0):
    """按目标 ICC 反推组间/组内方差。"""
    rng = np.random.default_rng(seed)
    var_resid = 1.0
    var_group = var_resid * icc_target / (1 - icc_target)
    g = np.repeat(np.arange(n_group), per)
    a = rng.normal(0, np.sqrt(var_group), n_group)[g]
    x = rng.normal(size=len(g))
    y = 0.8 * x + a + rng.normal(0, np.sqrt(var_resid), len(g))
    return pd.DataFrame({"y": y, "x": x, "grp": g})


def test_mixed_recovers_icc():
    d = hlm_data(icc_target=0.4)
    f = mixed(d, {"id": "hlm", "y": "y", "x": ["x"], "group": "grp"})
    assert abs(f.data["icc"] - 0.4) < 0.15, f.data["icc"]
    xrow = next(r for r in f.data["rows"] if r["var"] == "x")
    assert abs(xrow["coef"] - 0.8) < 0.1, xrow


def test_mixed_warns_when_no_group_structure():
    d = hlm_data(icc_target=0.001, seed=3)
    f = mixed(d, {"id": "hlm", "y": "y", "x": ["x"], "group": "grp"})
    assert f.data["icc"] < 0.05
    assert f.status == "warn"


def test_mixed_needs_groups():
    d = hlm_data()
    d["grp"] = 0
    f = mixed(d, {"id": "hlm", "y": "y", "x": ["x"], "group": "grp"})
    assert f.status == "fail"
    assert "随机效应" in f.why


# —— 结构方程 ——

def test_sem_recovers_path_and_fits():
    rng = np.random.default_rng(7)
    n = 600
    f1 = rng.normal(size=n)
    f2 = 0.6 * f1 + rng.normal(0, 0.5, n)
    d = pd.DataFrame({
        "x1": f1 + rng.normal(0, 0.3, n), "x2": f1 + rng.normal(0, 0.3, n),
        "x3": f1 + rng.normal(0, 0.3, n),
        "y1": f2 + rng.normal(0, 0.3, n), "y2": f2 + rng.normal(0, 0.3, n)})
    desc = """
    f1 =~ x1 + x2 + x3
    f2 =~ y1 + y2
    f2 ~ f1
    """
    f = sem(d, {"id": "sem", "model": desc})
    assert f.status in ("ok", "warn"), f.warnings
    path = next(p for p in f.data["paths"] if p["op"] == "~")
    assert abs(path["coef"] - 0.6) < 0.15, path
    assert f.data["fit"].get("CFI", 0) > 0.9, f.data["fit"]


def test_sem_bad_spec_fails():
    f = sem(pd.DataFrame({"a": [1, 2, 3]}), {"id": "sem", "model": "f1 =~ nonexistent"})
    assert f.status == "fail"
