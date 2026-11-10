"""多表合并的机制测试。

判据不是「能跑」，而是「该报错的地方必须报错」：
右表键重复、连接键格式不一致、撞名、缺列——这四种在真实数据里都会出现，
而且静默通过时的后果都是「样本量悄悄变化」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epkit.pipeline import run_spec
from epkit.sample import build_sample
from epkit.spec import DataSpec, MergeSpec, StudySpec


def write(path, df):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def base_panel(n_codes=50, years=(2019, 2020, 2021), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_codes):
        code = f"{c:06d}"
        fe = rng.normal(0, 0.5)
        for y in years:
            rows.append({"code": code, "year": y, "y": 0.8 * fe + rng.normal(),
                         "treat": int(c % 2 == 0)})
    return pd.DataFrame(rows)


def make_spec(tmp_path, merge_yaml: str, panel=None):
    if panel is None:
        panel = base_panel()
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    write(pdir / "main.csv", panel)
    spec = tmp_path / "s.yaml"
    spec.write_text(f"""name: t
data:
  panel: {pdir / 'main.csv'}
{merge_yaml}
tests: []
output_dir: {tmp_path / 'out'}
""", encoding="utf-8")
    return spec


def test_left_merge_keeps_rows_and_records_flow(tmp_path):
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    fin = pd.DataFrame({"code": [f"{c:06d}" for c in range(50)] * 1,
                        "roa": np.random.default_rng(1).normal(size=50)})
    write(pdir / "fin.csv", fin)
    spec = make_spec(tmp_path, f"""  merge:
    - with: {pdir / 'fin.csv'}
      on: [code]
      how: left
      keep: [roa]
      validate: many_to_one""")
    F = run_spec(spec, verbose=False)
    step = next(s for s in F.sample_flow if s["step"].startswith("merge_"))
    assert step["n_before"] == 150 and step["n_after"] == 150
    assert "匹配 100.0%" in step["reason"], step


def test_merge_zero_rows_fails_with_code_hint(tmp_path):
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    # 右表代码带交易所前缀，与主表的 6 位纯数字对不上
    fin = pd.DataFrame({"code": [f"sh.{c:06d}" for c in range(50)], "roa": np.arange(50.0)})
    write(pdir / "fin.csv", fin)
    spec = make_spec(tmp_path, f"""  merge:
    - with: {pdir / 'fin.csv'}
      on: [code]
      how: inner
      keep: [roa]""")
    with pytest.raises(ValueError, match="0 行"):
        run_spec(spec, verbose=False)


def test_duplicate_right_keys_are_caught(tmp_path):
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    # 右表同一 code 有两行，却声明 many_to_one
    fin = pd.DataFrame({"code": ["000000", "000000"], "roa": [0.1, 0.2]})
    write(pdir / "fin.csv", fin)
    spec = make_spec(tmp_path, f"""  merge:
    - with: {pdir / 'fin.csv'}
      on: [code]
      how: left
      keep: [roa]
      validate: many_to_one""")
    with pytest.raises(ValueError, match="validate"):
        run_spec(spec, verbose=False)


def test_name_clash_requires_rename_or_suffix(tmp_path):
    """撞名且没有 suffix 时必须报错，而不是让 pandas 静默加 _x/_y。"""
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    fin = pd.DataFrame({"code": [f"{c:06d}" for c in range(50)], "treat": [0] * 50})
    write(pdir / "fin.csv", fin)
    main = base_panel()
    write(pdir / "main.csv", main)
    # 用 API 构造，避免在 YAML 里写空字符串（三引号会打架）
    spec = StudySpec(
        name="t",
        data=DataSpec(panel=str(pdir / "main.csv"),
                      merge=[MergeSpec(**{"with": str(pdir / "fin.csv"),
                                          "on": ["code"], "keep": ["treat"],
                                          "suffix": ""})]),
        tests=[],
        output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="同名"):
        build_sample(spec)


def test_missing_column_is_named(tmp_path):
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    fin = pd.DataFrame({"code": [f"{c:06d}" for c in range(50)], "roa": [0.1] * 50})
    write(pdir / "fin.csv", fin)
    spec = make_spec(tmp_path, f"""  merge:
    - with: {pdir / 'fin.csv'}
      on: [code]
      keep: [roa, lev]""")
    with pytest.raises(KeyError, match="lev"):
        run_spec(spec, verbose=False)


def test_low_match_rate_is_flagged_in_reason(tmp_path):
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    # 只有一半公司有财务数据
    fin = pd.DataFrame({"code": [f"{c:06d}" for c in range(25)], "roa": [0.1] * 25})
    write(pdir / "fin.csv", fin)
    spec = make_spec(tmp_path, f"""  merge:
    - with: {pdir / 'fin.csv'}
      on: [code]
      how: left
      keep: [roa]
      min_match: 0.9""")
    F = run_spec(spec, verbose=False)
    step = next(s for s in F.sample_flow if s["step"].startswith("merge_"))
    assert "匹配 50.0%" in step["reason"]
    assert "低于" in step["reason"]


def test_rename_applied(tmp_path):
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    fin = pd.DataFrame({"code": [f"{c:06d}" for c in range(50)], "treat": [1] * 50})
    write(pdir / "fin.csv", fin)
    spec = make_spec(tmp_path, f"""  merge:
    - with: {pdir / 'fin.csv'}
      on: [code]
      keep: [treat]
      rename: {{treat: treat_fin}}""")
    F = run_spec(spec, verbose=False)
    assert any(s["step"].startswith("merge_") for s in F.sample_flow)


def test_leading_zeros_are_restored_before_merge(tmp_path):
    """右表代码被 Excel 吃掉前导零（1 而不是 000001）时，合并前要补回来。"""
    pdir = tmp_path / "data"
    pdir.mkdir(exist_ok=True)
    main = base_panel()
    write(pdir / "main.csv", main)
    fin = pd.DataFrame({"code": [c for c in range(50)], "roa": [0.5] * 50})
    write(pdir / "fin.csv", fin)
    spec = StudySpec(
        name="t",
        data=DataSpec(panel=str(pdir / "main.csv"),
                      merge=[MergeSpec(**{"with": str(pdir / "fin.csv"),
                                          "on": ["code"], "keep": ["roa"]})]),
        tests=[],
        output_dir=str(tmp_path / "out"))
    df, flow = build_sample(spec)
    step = next(s for s in flow.as_list() if s["step"].startswith("merge_"))
    assert "匹配 100.0%" in step["reason"], step
    assert df["roa"].notna().all()
