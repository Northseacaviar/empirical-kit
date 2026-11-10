"""审查报告中「中等」项的回归测试。

核心原则：诊断没做成必须变成 warn，不能变成「没问题」；
口径写在结果里，不能嘴上说一套、算另一套。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.checks.multitest import collect_p, multitest
from epkit.checks.reliability import krippendorff_alpha
from epkit.data_sources.baostock import _to_bs_code
from epkit.data_sources.csmar import normalize
from epkit.findings import Finding, Findings


# —— M1/M2：Krippendorff α 在 m_u 不等时也要对，level 不能静默 ——

def oracle_alpha(units: list[list], level: str = "nominal") -> float:
    """按 coincidence matrix 的教科书定义独立实现，用作对照。"""
    clean = [[v for v in u if v is not None and not (isinstance(v, float) and np.isnan(v))]
             for u in units]
    vals = sorted({v for u in clean for v in u})
    idx = {c: i for i, c in enumerate(vals)}
    K = len(vals)
    o = np.zeros((K, K))
    for u in clean:
        m = len(u)
        if m < 2:
            continue
        c = np.zeros(K)
        for x in u:
            c[idx[x]] += 1
        for i in range(K):
            for j in range(K):
                o[i, j] += c[i] * (c[j] - (1 if i == j else 0)) / (m - 1)
    n = o.sum()
    nc = o.sum(axis=1)

    def d(i, j):
        if level == "nominal":
            return 0.0 if i == j else 1.0
        if level == "interval":
            return float((i - j) ** 2)
        lo, hi = min(i, j), max(i, j)
        return float((nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2)

    Do = sum(o[i, j] * d(i, j) for i in range(K) for j in range(K) if i != j) / n
    De = sum(nc[i] * nc[j] * d(i, j) for i in range(K) for j in range(K) if i != j) / (n * (n - 1))
    return 1 - Do / De


def test_alpha_matches_oracle_with_unequal_coders():
    units = [[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1], [2, 2, 2], [2, 1]]
    got = krippendorff_alpha(units, "nominal")
    assert got == pytest.approx(oracle_alpha(units, "nominal"), abs=1e-9), got


def pooled_alpha_old(units: list[list]) -> float:
    """旧实现：池化有序对，按 m_u(m_u-1) 加权。用来证明两种算法确实不同。"""
    import itertools
    tot, dis, vals = 0.0, 0.0, []
    for u in units:
        u = [v for v in u if v is not None]
        if len(u) < 2:
            continue
        tot += len(u) * (len(u) - 1)
        dis += 2 * sum(1 for a, b in itertools.combinations(u, 2) if a != b)
        vals.extend(u)
    s = pd.Series(vals)
    N = len(s)
    vc = s.value_counts()
    Do = dis / tot
    De = 1 - float((vc * (vc - 1)).sum()) / (N * (N - 1))
    return 1 - Do / De


def test_alpha_unequal_coders_differs_from_pooled_definition():
    units = ([[1, 1, 1]] * 30 + [[0, 0, 0]] * 30 + [[1, 0]] * 20
             + [[1, 0, 0]] * 10 + [[0, 1, 1]] * 10)
    a = krippendorff_alpha(units, "nominal")
    assert a == pytest.approx(oracle_alpha(units, "nominal"), abs=1e-9)
    # m_u 不等时两种定义必然不同；差多少不重要，重要的是别把旧算法当成标准定义
    assert a != pytest.approx(pooled_alpha_old(units), abs=1e-6)


def test_alpha_ordinal_penalises_distant_disagreement():
    # 分歧全部发生在 1 与 3 之间：ordinal 的惩罚比 nominal 重，α 应更低
    units = [[1, 3], [1, 3], [2, 2], [1, 1], [3, 3], [2, 3], [1, 1]]
    nom = krippendorff_alpha(units, "nominal")
    ordi = krippendorff_alpha(units, "ordinal")
    assert ordi < nom, (nom, ordi)
    assert ordi == pytest.approx(oracle_alpha(units, "ordinal"), abs=1e-9)


def test_alpha_rejects_unknown_level():
    with pytest.raises(ValueError):
        krippendorff_alpha([[1, 1]], "ratio")


# —— M5/M6：多重检验族不能含预期间；nan 不能污染汇总 ——

def test_multitest_family_excludes_pre_period():
    F = Findings("t", "h")
    F.add(Finding("d1", "did", "ok", "", {"event_time": [
        {"k": -2, "coef": 0.1, "se": 0.1, "t": 1.0, "p": 0.02},
        {"k": -1, "coef": 0.0, "se": 0.1, "t": 0.0, "p": 0.9},
        {"k": 0, "coef": 0.5, "se": 0.1, "t": 5.0, "p": 0.001},
        {"k": 1, "coef": 0.5, "se": 0.1, "t": 5.0, "p": 0.001}]}))
    labels = [a for a, _ in collect_p(F)]
    assert labels == ["d1:k=+0", "d1:k=+1"], labels


def test_multitest_drops_nan_p():
    F = Findings("t", "h")
    for i, p in enumerate([0.01, np.nan, 0.2]):
        F.add(Finding(f"g{i}", "group_diff", "ok", "", {
            "var": "x", "p_welch": p, "diff": 0.0,
            "groups": [{"mean": 0.0, "n": 50}, {"mean": 0.0, "n": 50}],
            "t_welch": 0.0, "t_equal": 0.0}))
    labels = [a for a, _ in collect_p(F)]
    assert labels == ["g0", "g2"], labels
    f = multitest(F, {"id": "mt", "method": "bh"})
    assert all(np.isfinite(t["p_adj"]) for t in f.data["tests"])
    assert f.data["n_sig_adj"] == 1


# —— M7：没有 statsmodels 时回退实现要按 method 各自算 ——

def test_fallback_matches_statsmodels_per_method(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("statsmodels.stats.multitest"):
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    p = [0.01, 0.02, 0.03, 0.04, 0.2]
    F = Findings("t", "h")
    for i, pv in enumerate(p):
        F.add(Finding(f"g{i}", "group_diff", "ok", "", {
            "var": "x", "p_welch": pv, "diff": 0.0,
            "groups": [{"mean": 0.0, "n": 50}, {"mean": 0.0, "n": 50}],
            "t_welch": 0.0, "t_equal": 0.0}))

    from statsmodels.stats.multitest import multipletests

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for method, sm_name in [("bh", "fdr_bh"), ("holm", "holm"), ("bonferroni", "bonferroni")]:
        f = multitest(F, {"id": "mt", "method": method})
        expect = multipletests(np.array(p), alpha=0.05, method=sm_name)[1]
        got = [t["p_adj"] for t in f.data["tests"]]
        assert np.allclose(got, expect, atol=1e-12), (method, got, list(expect))


# —— M8/M9：CSMAR 重复目标列不崩、代码补零 ——

def test_csmar_duplicate_target_columns_do_not_crash():
    df = pd.DataFrame({"Stkcd": ["000001"], "Symbol": ["000001"],
                       "Trddt": ["2023-01-03"], "Clprc": [10.0], "Clsprc": [10.0]})
    out, info = normalize(df)
    assert "code" in out.columns
    assert out["code"].iloc[0] == "000001"
    assert info.get("conflicts"), info


def test_csmar_numeric_code_gets_zero_filled():
    df = pd.DataFrame({"Stkcd": [1, 600000, 300750], "Clsprc": [1.0, 2.0, 3.0]})
    out, _ = normalize(df)
    assert list(out["code"]) == ["000001", "600000", "300750"]


# —— M10：停牌不打断复牌首日收益 ——

def test_market_prefix_covers_funds_and_b_shares():
    assert _to_bs_code("159915") == "sz.159915"      # 深市 ETF
    assert _to_bs_code("200002") == "sz.200002"      # 深 B 股
    assert _to_bs_code("510300") == "sh.510300"      # 沪市 ETF
    with pytest.raises(ValueError):
        _to_bs_code("700001")


# —— M11：预趋势在聚类时用 F 口径 ——

def test_did_pre_trend_uses_f_distribution_with_clusters():
    from epkit.estimate.did import did

    rng = np.random.default_rng(4)
    rows = []
    for i in range(120):
        fe = rng.normal(0, 0.5)
        g = 2015 if i % 2 == 0 else 0
        for t in range(2010, 2021):
            post = 1 if (g and t >= g) else 0
            rows.append({"firm": i, "year": t, "cohort": g,
                         "y": fe + 0.6 * post + rng.normal(0, 0.5)})
    d = pd.DataFrame(rows)
    f = did(d, {"id": "d", "y": "y", "cohort": "cohort", "period": "year",
                "idvar": "firm", "window": [-4, 4], "cluster": ["firm"]})
    wt = f.data["pre_trend_wald"]
    assert wt is not None
    assert wt["distribution"] == "F" and wt["df2"] == 119, wt
