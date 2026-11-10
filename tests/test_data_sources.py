"""数据源层测试：列名映射、收益率单位识别、代码规范化。

单位这条用真实撞过的坑做回归：CSMAR 给小数（0.008）、自建面板给百分点（0.8），
差 100 倍且不报错。必须能在读数阶段就判出来。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.data_sources.baostock import _to_bs_code as to_bs_code
from epkit.data_sources.csmar import audit_units, detect_return_units, normalize


def csmar_like(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Stkcd": [f"{i:06d}" for i in range(n)],
        "Trddt": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Clsprc": rng.uniform(5, 20, n),
        "Dretwd": rng.normal(0, 0.02, n),
        "Nindcd": rng.integers(1, 20, n),
    })


def test_normalize_renames_csmar_columns():
    df, info = normalize(csmar_like())
    assert {"code", "date", "close", "ret", "industry"} <= set(df.columns)
    assert info["renamed"]["Stkcd"] == "code"
    assert df["date"].dtype.kind == "M"


def test_normalize_strips_exchange_prefix_and_float_code():
    df = pd.DataFrame({"Stkcd": ["sh.600000", "000001.0"], "Clsprc": [1.0, 2.0]})
    out, _ = normalize(df)
    assert list(out["code"]) == ["600000", "000001"]


def test_detect_units_decimal_vs_percent():
    rng = np.random.default_rng(1)
    dec = pd.Series(rng.normal(0, 0.015, 500))
    pct = dec * 100
    assert detect_return_units(dec) == "decimal"
    assert detect_return_units(pct) == "percent"


def test_detect_units_on_skewed_percent_car():
    # 百分点口径的 CAR，但分布集中在小值（中位数 0.2 左右）、尾巴到 3：
    # 用中位数判会落进盲区，只有 P99 判据认得出来。
    rng = np.random.default_rng(9)
    n = 2000
    v = np.where(rng.random(n) < 0.7, rng.normal(0, 0.25, n), rng.normal(0, 1.8, n))
    s = pd.Series(v)
    assert 0.2 <= abs(s).median() <= 0.5, abs(s).median()
    assert detect_return_units(s) == "percent"


def test_detect_units_gives_up_on_short_series():
    assert detect_return_units(pd.Series([0.01, 0.02])) is None


def test_audit_flags_decimal_without_scale():
    df, _ = normalize(csmar_like())
    notes = audit_units(df, scale=1.0, unit="")
    assert any("小数口径" in n and "scale: 100" in n for n in notes), notes


def test_audit_flags_double_scaling():
    df, _ = normalize(csmar_like())
    df["ret"] = df["ret"] * 100          # 已经换成百分点
    notes = audit_units(df, scale=100, unit="%")
    assert any("已经是百分点口径" in n for n in notes), notes


def test_audit_checks_car_columns_too():
    # 小数口径的 CAR：标准差 0.03，也就是 3 个百分点
    df = pd.DataFrame({"car_0_5": np.random.default_rng(2).normal(0, 0.03, 400)})
    notes = audit_units(df, scale=1.0, unit="")
    assert any("car_0_5" in n for n in notes), notes


def test_audit_quiet_when_scale_matches():
    # 两种自洽组合都不该报警：小数口径配 scale=100；百分点口径配 scale=1
    dec = pd.DataFrame({"car_0_5": np.random.default_rng(3).normal(0, 0.03, 400)})
    assert audit_units(dec, scale=100, unit="%") == []
    pct = pd.DataFrame({"car_0_5": np.random.default_rng(3).normal(0, 3.0, 400)})
    assert audit_units(pct, scale=1.0, unit="%") == []


def test_read_table_applies_column_map(tmp_path):
    from epkit.data import read_table

    src = tmp_path / "csmar.csv"
    csmar_like().to_csv(src, index=False, encoding="utf-8-sig")
    df = read_table(src, column_map="csmar")
    assert "code" in df.columns and "ret" in df.columns


@pytest.mark.parametrize("raw,expect", [
    ("600000", "sh.600000"), ("000001", "sz.000001"), ("300750", "sz.300750"),
    ("430047", "bj.430047"), ("sh.600000", "sh.600000"),
])
def test_to_bs_code(raw, expect):
    assert to_bs_code(raw) == expect
