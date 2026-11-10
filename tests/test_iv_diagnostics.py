"""IV 诊断的机制测试：诊断能不能认出强工具、弱工具、坏工具。

只断"跑得通"没有意义——三个阶段各造一份已知答案的数据：
强工具 F 应该大、弱工具 F 应该小、违反排他性的工具应该让 Sargan 拒绝。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from epkit.estimate.iv import iv_diagnostics, iv_warnings
from epkit.estimate.reg import reg


def make(n: int = 3000, first_stage: float = 0.6, bad_instrument: float = 0.0,
         seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    x = first_stage * z1 + 0.5 * z2 + 0.4 * u + rng.normal(size=n)
    # bad_instrument>0 时，z2 直接影响 y，违反排他性
    y = 1.0 * x + u + bad_instrument * z2 + rng.normal(size=n)
    gid = rng.integers(0, 60, n)
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "gid": gid})


CFG = {"y": "y", "x": [], "controls": [], "fe": [], "cluster": ["gid"],
       "iv": {"endog": ["x"], "instruments": ["z1", "z2"]}}


def test_strong_instrument_passes():
    diag = iv_diagnostics(make(first_stage=0.6), CFG)
    fs = diag["first_stage"][0]
    assert fs["f"] > 30, fs
    assert diag["weak"] is False
    assert diag["sargan"]["p"] > 0.05, diag["sargan"]


WEAK_CFG = dict(CFG, iv={"endog": ["x"], "instruments": ["z1"]})


def test_weak_instrument_flagged():
    # 单工具 + 与 x 几乎无关：联合 F 必然小。用两个工具时要一起弱，否则强的那支会撑起 F。
    diag = iv_diagnostics(make(first_stage=0.02, seed=11), WEAK_CFG)
    fs = diag["first_stage"][0]
    assert fs["f"] < 10, fs
    assert diag["weak"] is True
    warns, bad = iv_warnings(diag)
    assert bad is True
    assert any("弱工具" in w for w in warns), warns


def test_bad_instrument_rejected_by_sargan():
    # z2 直接进 y：工具不满足排他性，Sargan 应该拒绝
    diag = iv_diagnostics(make(bad_instrument=0.35, seed=3), CFG)
    assert diag["sargan"] is not None
    assert diag["sargan"]["p"] < 0.01, diag["sargan"]
    warns, bad = iv_warnings(diag)
    assert any("过度识别" in w for w in warns), warns


def test_exactly_identified_has_no_sargan():
    cfg = dict(CFG, iv={"endog": ["x"], "instruments": ["z1"]})
    diag = iv_diagnostics(make(), cfg)
    assert diag["sargan"] is None
    assert any("恰好识别" in n for n in diag["notes"]), diag["notes"]


def test_reg_attaches_diagnostics_and_warns():
    f = reg(make(first_stage=0.02, seed=5), dict(WEAK_CFG, id="iv_weak"))
    assert f.type == "reg"
    assert f.data["iv"] is not None
    assert f.data["iv"]["weak"] is True
    assert f.status == "warn"
    assert any("弱工具" in w for w in f.warnings), f.warnings


def test_reg_iv_coefficient_recovered():
    f = reg(make(first_stage=0.6), dict(CFG, id="iv_ok"))
    step = next(s for s in f.data["steps"] if "by_vcov" in s)
    coef = step["coef"]
    assert abs(coef - 1.0) < 0.15, coef


def test_iv_warnings_flags_sargan_rejection():
    """Sargan 判据的守备：p 在 0.05 附近也要报，不能只在极小 p 时才报。"""
    diag = {"first_stage": [],
            "sargan": {"stat": 4.2, "df": 1, "p": 0.04, "n": 500, "r2": 0.0084}}
    warns, bad = iv_warnings(diag)
    assert bad is True
    assert any("过度识别" in w for w in warns), warns
