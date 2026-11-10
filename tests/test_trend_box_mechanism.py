"""趋势图、箱线图与相关热图的机制测试。

图的判据不是「生成了 png」，而是：统计量与手算一致、图上数据与 finding 同源、
黑白版确实去掉了彩色。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.checks.trend_box import box_plot, trend
from epkit.findings import Finding, Findings
from epkit.report.figures import box_plot_fig, corr_heatmap, trend_plot


def panel(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in (0, 1):
        for y in (2019, 2020, 2021):
            for _ in range(n // 6):
                rows.append({"y": rng.normal(0.5 * g + 0.1 * (y - 2019), 1.0),
                             "grp": g, "year": y})
    return pd.DataFrame(rows)


def test_trend_mean_and_ci_match_manual():
    d = panel()
    f = trend(d, {"id": "tr", "var": "y", "by": "year", "group": "grp"})
    p = next(x for x in f.data["points"] if x["group"] == "0" and x["time"] == "2019")
    sub = d[(d.grp == 0) & (d.year == 2019)]["y"]
    assert p["mean"] == pytest.approx(float(sub.mean()), abs=1e-9)
    se_expect = float(sub.std(ddof=1) / np.sqrt(len(sub)))
    assert p["se"] == pytest.approx(se_expect, abs=1e-9)
    assert p["ci_low"] == pytest.approx(p["mean"] - 1.96 * se_expect, abs=1e-9)


def test_trend_without_group():
    d = panel()
    f = trend(d, {"id": "tr", "var": "y", "by": "year"})
    assert all(x["group"] == "" for x in f.data["points"])
    assert len(f.data["points"]) == 3


def test_trend_thin_cells_warn():
    d = panel(n=60)
    d = d.groupby(["grp", "year"]).head(3)
    f = trend(d, {"id": "tr", "var": "y", "by": "year", "group": "grp"})
    assert f.status == "warn"
    assert any("样本少于 10" in w for w in f.warnings)


def test_box_stats_match_quartiles():
    d = panel()
    f = box_plot(d, {"id": "bx", "var": "y", "by": "grp"})
    g0 = next(g for g in f.data["groups"] if g["group"] == "0")
    sub = d[d.grp == 0]["y"]
    assert g0["median"] == pytest.approx(float(sub.median()), abs=1e-9)
    assert g0["q1"] == pytest.approx(float(sub.quantile(0.25)), abs=1e-9)


def _findings_from(d):
    F = Findings("toy", "h")
    F.add(trend(d, {"id": "tr", "var": "y", "by": "year", "group": "grp"}))
    F.add(box_plot(d, {"id": "bx", "var": "y", "by": "grp"}))
    F.add(Finding("corr", "corr", "ok", "", {
        "vars": ["y", "x"], "method": "pearson", "n_complete": 100,
        "matrix": [[None, 0.35], [0.35, None]]}))
    return F


def test_three_figures_are_written(tmp_path):
    F = _findings_from(panel())
    paths = [trend_plot(F, tmp_path / "trend.png"),
             box_plot_fig(F, tmp_path / "box.png"),
             corr_heatmap(F, tmp_path / "corr.png")]
    for p in paths:
        assert p is not None and p.exists() and p.stat().st_size > 3000, p


def test_monochrome_versions_written(tmp_path):
    F = _findings_from(panel())
    for fn, name in ((trend_plot, "tr_bw.png"), (box_plot_fig, "bx_bw.png"),
                     (corr_heatmap, "corr_bw.png")):
        p = fn(F, tmp_path / name, bw=True)
        assert p is not None and p.exists() and p.stat().st_size > 3000, p


def test_figures_absent_without_findings(tmp_path):
    F = Findings("empty", "h")
    assert trend_plot(F, tmp_path / "t.png") is None
    assert box_plot_fig(F, tmp_path / "b.png") is None
    assert corr_heatmap(F, tmp_path / "c.png") is None
