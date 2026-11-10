"""审查报告（2026-11-10）里五条阻塞项的回归测试。

每条都构造「旧实现会给出看似正常但错误的结果」的数据：
1. CS 的 k<0 格子自纳入 → 预期间 ATT 被压到一半；
2. warn 早退路径的空 data 让 report.md 崩；
3. spec_curve 的 drop_where 反语义；
4. FE 丢 singleton 时 Sargan 长度不匹配；
5. 第一阶段算不出来却报 ok。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.checks.confound import confound
from epkit.checks.pileup import pileup
from epkit.checks.spec_curve import spec_curve
from epkit.estimate.iv import iv_diagnostics, iv_warnings
from epkit.estimate.staggered import cs_att_gt
from epkit.findings import Finding, Findings
from epkit.report.tables import report_md
from epkit.spec import StudySpec


# —— B1：CS 的预期间控制组不能含处理组自己 ——

def ramp_panel(seed: int = 11, n_per: int = 300) -> pd.DataFrame:
    """2015 队列在 2010–2014 有 +0.5/年 的差分预趋势，基准期之后持平；其他组全程平坦。

    基准期是 g-1=2014，所以 k=-3（2012）的长差应约 -1.0（两年 × 0.5，方向为
    「早一点减晚一点」）。旧实现把处理组自己算进控制组，会把幅度按样本比例
    压到 -0.5 ~ -0.67；k>=0 各格本应约为 0（趋势在 2015 就停了）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g, ids in [(2015, range(n_per)), (2018, range(n_per, 2 * n_per)),
                   (0, range(2 * n_per, 3 * n_per))]:
        for i in ids:
            fe = rng.normal(0, 0.5)
            for t in range(2010, 2023):
                # 趋势在基准期（2014）就停：否则处理当年还被算进「效应」
                trend = 0.5 * (min(t, 2014) - 2010) if g == 2015 else 0.0
                rows.append({"firm": i, "year": t, "cohort": g,
                             "y": fe + trend + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def test_cs_pre_period_excludes_own_treatment_group():
    d = ramp_panel()
    res = cs_att_gt(d, "y", "firm", "year", "cohort", window=(-3, 2))
    c = next(r for r in res["att_gt"] if r["g"] == 2015 and r["k"] == -3)
    # 两年的差分预趋势 = 2 × 0.5 = 1.0（方向：2012 − 2014 = −1.0）
    assert c["att"] < -0.75, c
    # 控制组只能来自「2015 之后才处理」的队列与 never 组，绝不能含自己那 300 家
    assert c["n_control"] <= 600, c


def test_cs_post_period_unchanged_by_fix():
    d = ramp_panel()
    res = cs_att_gt(d, "y", "firm", "year", "cohort", window=(-2, 2))
    post = [r for r in res["att_gt"] if r["k"] >= 0]
    assert post
    assert all(abs(r["att"]) < 0.3 for r in post), post   # 没有处理效应


def test_staggered_rejects_nan_never_encoding():
    d = ramp_panel()
    d["cohort"] = d["cohort"].replace({0: np.nan})
    with pytest.raises(ValueError, match="never"):
        cs_att_gt(d, "y", "firm", "year", "cohort", window=(0, 1))


# —— B2：warn 早退路径不能把 report.md 打崩 ——

def test_report_md_survives_early_exit_findings():
    F = Findings("toy", "h")
    F.add(Finding("pu", "pileup", "warn", "观测太少",
                  {"var": "div", "cut": 0.3, "bw": 0.05, "n_near": 12,
                   "f_left": None, "f_right": None, "theta": None, "z": None,
                   "p": None, "bins": []}))
    F.add(Finding("cf", "confound", "warn", "主事件不是单一日期",
                  {"main_event": None, "windows": [], "events": [], "n_overlap": 0}))
    md = report_md(F)
    assert "观测太少" in md and "主事件不是单一日期" in md


def test_pileup_early_exit_has_full_schema():
    d = pd.DataFrame({"div": np.random.default_rng(0).uniform(0.29, 0.31, 20)})
    f = pileup(d, {"id": "pu", "var": "div", "cut": 0.3, "bw": 0.05})
    assert f.status == "warn"
    assert f.data["bins"] == [] and f.data["theta"] is None
    assert "bins" in f.data


def test_pileup_rejects_too_few_bins():
    d = pd.DataFrame({"div": np.random.default_rng(1).uniform(0.1, 0.5, 2000)})
    f = pileup(d, {"id": "pu", "var": "div", "cut": 0.3, "bw": 0.05, "bins": 3})
    assert f.status == "fail"
    assert "太少" in f.why


def test_confound_fail_path_has_full_schema(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(f"""name: t
data: {{panel: x.csv}}
event: {{mode: column, column: d}}
returns: {{windows: [[-1, 1]]}}
tests: []
output_dir: {tmp_path / 'out'}
""", encoding="utf-8")
    spec = StudySpec.load(p)
    f = confound(spec, {"id": "cf", "other_events": [{"date": "2024-01-01"}]})
    assert f.status == "warn"
    assert f.data["events"] == []
    report_md(Findings("t", "h"))   # 空 findings 也不能崩


# —— B3：spec_curve 的 drop_where 是剔除，不是保留 ——

def _spec_curve_df(n=400, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    size = rng.uniform(10, 30, n)
    y = x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "size": size})


def test_spec_curve_drop_where_removes_rows():
    d = _spec_curve_df()
    n_all = len(d)
    n_drop = int((d["size"] > 20).sum())
    f = spec_curve(d, {"id": "sc", "y": "y", "x": "x", "controls": [[]], "fe": [[]],
                       "vcov": ["hetero"], "sample": [{"drop_where": "size > 20"}]})
    assert f.data["specs"][0]["n_sample"] == n_all - n_drop, f.data["specs"][0]


def test_spec_curve_keep_where_keeps_rows():
    d = _spec_curve_df()
    n_keep = int((d["size"] > 20).sum())
    f = spec_curve(d, {"id": "sc", "y": "y", "x": "x", "controls": [[]], "fe": [[]],
                       "vcov": ["hetero"], "sample": [{"keep_where": "size > 20"}]})
    assert f.data["specs"][0]["n_sample"] == n_keep


# —— B4 / B5：IV 诊断在 FE 与失败路径上必须给出结论 ——

def panel_iv(n_firm=150, years=6, seed=0, singleton=True):
    rng = np.random.default_rng(seed)
    n = n_firm * years
    firm = np.repeat(np.arange(n_firm), years)
    year = np.tile(np.arange(2018, 2018 + years), n_firm)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.6 * z1 + 0.5 * z2 + 0.4 * u + rng.normal(size=n)
    y = 0.8 * x + u + rng.normal(size=n)
    d = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "firm": firm, "year": year})
    if singleton:
        extra = pd.DataFrame({"y": [1.0, 2.0], "x": [0.1, 0.2], "z1": [0.3, 0.4],
                              "z2": [0.1, 0.2], "firm": [9901, 9902], "year": [2018, 2019]})
        d = pd.concat([d, extra], ignore_index=True)
    return d


IV_CFG = {"y": "y", "x": [], "controls": [], "fe": ["firm", "year"],
          "cluster": ["firm"], "iv": {"endog": ["x"], "instruments": ["z1", "z2"]}}


def test_sargan_runs_with_singleton_drop():
    d = panel_iv()
    diag = iv_diagnostics(d, IV_CFG)
    assert diag["sargan"] is not None, diag["notes"]
    assert diag["sargan"]["df"] == 1


def test_first_stage_failure_is_reported_not_silent():
    d = panel_iv()
    d["z_firm"] = d["firm"].astype(float)          # 个体层面不变 → 被 FE 吸收
    cfg = dict(IV_CFG, iv={"endog": ["x"], "instruments": ["z_firm", "z1"]})
    diag = iv_diagnostics(d, cfg)
    warns, bad = iv_warnings(diag)
    assert bad is True, diag
    assert any("第一阶段没算出来" in w for w in warns), warns


def test_overid_missing_becomes_warning():
    diag = {"first_stage": [], "sargan": None, "overid_needed": True}
    warns, bad = iv_warnings(diag)
    assert bad is True
    assert any("过度识别检验未完成" in w for w in warns), warns
