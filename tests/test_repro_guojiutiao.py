"""验收用例：用 epkit 复现 guojiutiao 论文表 2A。

基线数字取自论文正文表 2A（零分红 vs 有分红，diff 与 Welch t）。
容差：均值差 0.01 pp，t 值 0.05——超出说明 CAR 口径、样本或 t 检验被改动过。
"""
from pathlib import Path

import pytest

from epkit.pipeline import run_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "studies" / "guojiutiao_tab2a.yaml"

PAPER = {
    "tab2a_1d":  (-3.082, -10.75),
    "tab2a_3d":  (-5.463, -12.89),
    "tab2a_5d":  (-6.797, -13.00),
    "tab2a_10d": (-5.501, -8.87),
    "tab2a_20d": (-7.797, -8.83),
    "tab2a_30d": (-7.980, -7.03),
    "tab2a_40d": (-10.383, -8.48),
    "tab2a_60d": (-9.406, -7.41),
}


@pytest.fixture(scope="module")
def findings():
    return run_spec(SPEC, verbose=False)


def test_sample_split(findings):
    # 组 0 = 有分红，组 1 = 零分红；差定义为「组1 - 组0」
    gd = next(x for x in findings.items if x.type == "group_diff")
    g = {x["label"]: x["n"] for x in gd.data["groups"]}
    assert g["0"] == 521 and g["1"] == 279


@pytest.mark.parametrize("fid", list(PAPER))
def test_diff_and_t_match_paper(findings, fid):
    item = next(x for x in findings.items if x.id == fid)
    d, t = PAPER[fid]
    assert item.data["diff"] == pytest.approx(d, abs=0.01), f"{fid} diff"
    assert item.data["t_welch"] == pytest.approx(t, abs=0.05), f"{fid} t"


def test_every_finding_has_provenance(findings):
    for f in findings.items:
        assert f.provenance.get("spec_hash"), f"{f.id} 缺 spec hash"
