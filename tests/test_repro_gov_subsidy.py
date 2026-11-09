"""验收用例：复现 gov-subsidy 论文 output/cluster_sensitivity.md 的 H2/H3。

基线来自该论文的聚类敏感性表（系数与标准误，单位为百分点）：
H2 disclosed_asset：公司 5.76%(2.26%)、公告日 5.76%(2.28%)、双向 5.76%(2.26%)
H3 log_amount×lev ：公司 -3.35%(1.39%)、公告日 -3.35%(1.37%)、双向 -3.35%(1.39%)
"""
from pathlib import Path

import pytest

from epkit.pipeline import run_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "studies" / "gov_subsidy_h2_cluster.yaml"

BASE = {
    ("h2_disclose", "disclosed_asset"): {"cluster_sec_code": (0.0576, 0.0226),
                                         "cluster_announcement_date": (0.0576, 0.0228),
                                         "cluster_twoway": (0.0576, 0.0226)},
    ("h3_amount_lev", "log_amount:lev_w"): {"cluster_sec_code": (-0.0335, 0.0139),
                                            "cluster_announcement_date": (-0.0335, 0.0137),
                                            "cluster_twoway": (-0.0335, 0.0139)},
}


@pytest.fixture(scope="module")
def findings():
    return run_spec(SPEC, verbose=False)


@pytest.mark.parametrize("key", list(BASE))
def test_coef_and_se_by_cluster(findings, key):
    fid, term = key
    item = next(x for x in findings.items if x.id == fid)
    st = item.data["steps"][0]
    for vname, (coef, se) in BASE[key].items():
        got = st["by_vcov"][vname]["terms"][term]
        assert got["coef"] == pytest.approx(coef, abs=0.001), f"{fid}/{vname} coef"
        assert got["se"] == pytest.approx(se, abs=0.001), f"{fid}/{vname} se"


def test_all_five_vcovs_present(findings):
    for item in findings.items:
        if item.type == "reg":
            assert {"iid", "hetero", "cluster_sec_code",
                    "cluster_announcement_date", "cluster_twoway"} <= set(
                item.data["steps"][0]["by_vcov"])
