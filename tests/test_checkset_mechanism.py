"""混杂事件、阈值堆积、双编码信度的机制测试。

三份都造了"有/没有"两种情形的已知答案：查不出来和乱报警同样算失败。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.checks.confound import confound
from epkit.checks.pileup import pileup
from epkit.checks.reliability import krippendorff_alpha, reliability
from epkit.spec import StudySpec


# —— 混杂事件 ——

def _spec(tmp_path, event_date="2023-08-28"):
    p = tmp_path / "s.yaml"
    p.write_text(f"""name: t
data: {{panel: x.csv}}
event: {{mode: single, date: "{event_date}"}}
returns: {{windows: [[-1, 1], [0, 5]]}}
tests: []
output_dir: {tmp_path / 'out'}
""", encoding="utf-8")
    return StudySpec.load(p)


def test_confound_flags_overlap(tmp_path):
    spec = _spec(tmp_path)
    f = confound(spec, {"id": "cf", "other_events": [
        {"date": "2023-08-28", "label": "同日降准"},
        {"date": "2023-06-01", "label": "无关事件"}]})
    assert f.data["n_overlap"] == 1
    assert f.status == "warn"
    assert any("重叠" in w for w in f.warnings)


def test_confound_clean_when_far_away(tmp_path):
    spec = _spec(tmp_path)
    f = confound(spec, {"id": "cf", "other_events": [
        {"date": "2020-01-02", "label": "很久以前"}]})
    assert f.data["n_overlap"] == 0
    assert f.status == "ok"


def test_confound_needs_events(tmp_path):
    spec = _spec(tmp_path)
    f = confound(spec, {"id": "cf", "other_events": []})
    assert f.status == "fail"


# —— 阈值堆积 ——

def test_pileup_detects_manipulation():
    # 阈值 0.3 处：右侧人为加倍（公司为了让分红率"达标"而调整）
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.5, 4000)
    boost = rng.uniform(0.3, 0.5, 4000)
    d = pd.DataFrame({"div": np.concatenate([base, boost])})
    f = pileup(d, {"id": "pu", "var": "div", "cut": 0.3, "bw": 0.05})
    assert f.status == "warn", f.why
    assert f.data["theta"] > 0, f.data["theta"]
    assert f.data["p"] < 0.05


def test_pileup_quiet_on_smooth_density():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"div": rng.uniform(0.1, 0.5, 6000)})
    f = pileup(d, {"id": "pu", "var": "div", "cut": 0.3, "bw": 0.05})
    assert f.status == "ok", (f.why, f.warnings)
    assert f.data["p"] > 0.05


def test_pileup_needs_enough_points():
    d = pd.DataFrame({"div": np.random.default_rng(2).uniform(0.28, 0.32, 30)})
    f = pileup(d, {"id": "pu", "var": "div", "cut": 0.3, "bw": 0.05})
    assert f.status == "warn"
    assert "没有意义" in f.why


# —— 双编码信度 ——

def test_alpha_perfect_agreement():
    units = [[1, 1] for _ in range(50)] + [[0, 0] for _ in range(50)]
    assert krippendorff_alpha(units) == pytest.approx(1.0)


def test_alpha_zero_on_independent_coding():
    rng = np.random.default_rng(3)
    units = [[int(rng.integers(0, 2)), int(rng.integers(0, 2))] for _ in range(3000)]
    a = krippendorff_alpha(units)
    assert abs(a) < 0.1, a


def test_alpha_handles_missing_coder():
    units = [[1, 1], [0, None], [1, 1], [0, 0], [1, None]]
    a = krippendorff_alpha(units)
    assert np.isfinite(a) and a > 0.5


def test_reliability_warns_below_threshold():
    rng = np.random.default_rng(4)
    d = pd.DataFrame({"a": rng.integers(0, 3, 200), "b": rng.integers(0, 3, 200)})
    f = reliability(d, {"id": "rel", "coders": ["a", "b"]})
    assert f.status == "warn"
    assert any("一致性不足" in w for w in f.warnings)


def test_reliability_passes_on_good_coding():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 3, 300)
    b = a.copy()
    flip = rng.random(300) < 0.05          # 5% 分歧
    b[flip] = rng.integers(0, 3, flip.sum())
    f = reliability(pd.DataFrame({"a": a, "b": b}), {"id": "rel", "coders": ["a", "b"]})
    assert f.data["alpha"] > 0.8, f.data
    assert f.status == "ok"


def test_reliability_needs_two_coders():
    f = reliability(pd.DataFrame({"a": [1, 0]}), {"id": "rel", "coders": ["a"]})
    assert f.status == "fail"
