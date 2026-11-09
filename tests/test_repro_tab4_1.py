"""验收用例：复现 gov-subsidy 论文表 4-1（主样本累计异常收益）。

基线：均值（%）、中位数（%）、t 值、正收益占比。源数据是小数，spec 里 scale=100。
"""
from pathlib import Path

import pytest

from epkit.pipeline import run_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "studies" / "gov_subsidy_tab4_1.yaml"

# (均值, 中位数, t, 正收益占比)
PAPER = {
    "tab4_1_m3_3": (-0.79, -0.63, -2.34, 0.446),
    "tab4_1_m1_1": (-0.14, -0.05, -0.61, 0.497),
    "tab4_1_0_1":  (-0.18, -0.34, -0.94, 0.438),
    "tab4_1_0_3":  (-0.56, -0.72, -2.09, 0.426),
    "tab4_1_m5_5": (-1.14, -0.74, -2.40, 0.452),
}


@pytest.fixture(scope="module")
def findings():
    return run_spec(SPEC, verbose=False)


@pytest.mark.parametrize("fid", list(PAPER))
def test_mean_median_t_pctpos(findings, fid):
    f = next(x for x in findings.items if x.id == fid)
    mean, med, t, pos = PAPER[fid]
    assert f.data["mean"] == pytest.approx(mean, abs=0.02), f"{fid} 均值"
    assert f.data["median"] == pytest.approx(med, abs=0.02), f"{fid} 中位数"
    assert f.data["t"] == pytest.approx(t, abs=0.05), f"{fid} t"
    assert f.data["pct_pos"] == pytest.approx(pos, abs=0.005), f"{fid} 正收益占比"


def test_n_matches_paper(findings):
    for f in findings.items:
        assert f.data["n"] == 352, f"{f.id} 样本量 {f.data['n']} != 352"
