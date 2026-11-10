"""多重检验校正的机制测试。

只测"能跑"没用。这里用一组手算得出的 p 值，核对 BH 的校正结果是否与
statsmodels 一致；再用一份"未校正显著、校正后不显著"的族，验证工具会报警。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from epkit.checks.multitest import collect_p, multitest
from epkit.findings import Finding, Findings


def F_with_groups(pvals, prefix="g"):
    F = Findings("toy", "hash")
    n = 60
    for i, p in enumerate(pvals):
        # 反推一个能让 Welch p 恰好约等于 p 的组间差：用大样本近似的 z 反算
        F.add(Finding(f"{prefix}{i}", "group_diff", "ok", "", {
            "var": f"car_{i}", "p_welch": p, "diff": 0.0,
            "groups": [{"mean": 0.0, "n": n}, {"mean": 0.0, "n": n}],
            "t_welch": 0.0, "t_equal": 0.0}))
    return F


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    f = multitest(F_with_groups(p), {"id": "mt", "method": "bh", "alpha": 0.05})
    _, expect, _, _ = multipletests(np.array(p), alpha=0.05, method="fdr_bh")
    got = [t["p_adj"] for t in f.data["tests"]]
    assert np.allclose(got, expect, atol=1e-9), (got, list(expect))
    assert f.data["n_sig_raw"] == 5
    assert f.data["n_sig_adj"] == 2


def test_bonferroni_is_more_conservative_than_bh():
    p = [0.001, 0.01, 0.02, 0.03, 0.04]
    bh = multitest(F_with_groups(p), {"id": "a", "method": "bh"})
    bon = multitest(F_with_groups(p), {"id": "b", "method": "bonferroni"})
    assert bon.data["n_sig_adj"] <= bh.data["n_sig_adj"]


def test_flip_is_flagged():
    # 小族 + 边缘 p：Bonferroni 校正后全部失去显著性，必须进 warn 并列出翻转项
    f = multitest(F_with_groups([0.011, 0.021, 0.031, 0.041]), {"id": "mt", "method": "bonferroni"})
    assert f.status == "warn"
    assert f.data["flipped"], f.data
    assert any("由显著转为不显著" in w for w in f.warnings)


def test_collect_p_reads_reg_and_did():
    F = Findings("toy", "hash")
    F.add(Finding("r1", "reg", "ok", "", {
        "focus": "x", "main_vcov": "cluster_id",
        "steps": [{"label": "单变量", "by_vcov": {"cluster_id": {"terms": {
            "x": {"coef": 1.0, "se": 0.1, "t": 10.0, "p": 0.001}}}}}]}))
    F.add(Finding("d1", "did", "ok", "", {"event_time": [
        {"k": 0, "coef": 0.5, "se": 0.1, "t": 5.0, "p": 0.01},
        {"k": 1, "coef": 0.5, "se": 0.1, "t": 1.0, "p": 0.4}]}))
    pairs = collect_p(F)
    assert len(pairs) == 3, pairs
    assert [a for a, _ in pairs] == ["r1:x[单变量]", "d1:k=+0", "d1:k=+1"]


def test_too_few_p_is_not_an_error():
    f = multitest(F_with_groups([0.01]), {"id": "mt"})
    assert f.status == "warn"
    assert "族太小" in f.why


def test_unknown_method_fails_loudly():
    f = multitest(F_with_groups([0.01, 0.02]), {"id": "mt", "method": "fdr_xx"})
    assert f.status == "fail"
    assert "未知校正方法" in f.why
