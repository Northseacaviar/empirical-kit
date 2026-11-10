"""多层/混合效应模型（随机截距、可选随机斜率）。

为什么要它：管理学与教育学里的数据天然分层（学生嵌套在班级、公司嵌套在行业），
普通 OLS 会把组内相关当成独立信息，标准误偏小。MixedLM 是 statsmodels 的现成实现。

口径：固定效应照常报；随机效应用方差成分表示（组间方差 / 残差方差），
并给 ICC = 组间方差 / (组间方差 + 残差方差)——论文里说明「有多少变异在组间」就靠它。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def mixed(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "mixed")
    try:
        import statsmodels.formula.api as smf
    except ImportError:  # pragma: no cover
        return Finding(fid, "mixed", "fail", "未安装 statsmodels", {})

    y, xs, group = cfg["y"], cfg.get("x", []), cfg["group"]
    if isinstance(xs, str):
        xs = [xs]
    re_slope = cfg.get("re_slope") or []
    if isinstance(re_slope, str):
        re_slope = [re_slope]
    need = [y, group] + list(xs) + list(re_slope)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "mixed", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    if d[group].nunique() < 3:
        return Finding(fid, "mixed", "fail",
                       f"分组变量 {group} 只有 {d[group].nunique()} 组，随机效应无法估计", {})

    rhs = " + ".join(list(xs)) if xs else "1"
    fml = f"{y} ~ {rhs}"
    kwargs: dict[str, Any] = {}
    if re_slope:
        kwargs["re_formula"] = "~ " + " + ".join(re_slope)
    try:
        # groups 传列名字符串：传 Series 时索引容易与 data 错位，随机效应会被估成 0
        m = smf.mixedlm(fml, data=d, groups=group, **kwargs).fit(reml=True)
    except Exception as exc:
        return Finding(fid, "mixed", "fail",
                       f"混合效应模型估计失败：{type(exc).__name__}: {exc}", {})

    rows = []
    for nm in m.params.index:
        if nm in ("Group Var", "Intercept") and nm not in xs:
            continue
        rows.append({"var": nm, "coef": float(m.params[nm]), "se": float(m.bse[nm]),
                     "z": float(m.tvalues[nm]), "p": float(m.pvalues[nm])})

    # 随机截距的方差就是 cov_re 的第一格（只放随机截距时是 1×1）
    try:
        var_group = float(m.cov_re.iloc[0, 0])
    except Exception:
        var_group = None
    var_resid = float(m.scale)
    icc = (var_group / (var_group + var_resid)) if var_group is not None and (var_group + var_resid) > 0 else None

    warnings, status = [], "ok"
    if icc is not None and icc < 0.05:
        status = "warn"
        warnings.append(f"ICC={icc:.3f}：组间变异只占 {icc:.1%}，多层结构几乎不存在，"
                        "用普通 OLS 加聚类标准误即可")
    if icc is not None and icc > 0.3:
        warnings.append(f"ICC={icc:.3f}：组间变异很大，不能忽略分层结构")

    why = (f"混合效应模型（{len(d)} 观测 / {d[group].nunique()} 组）："
           + "、".join(f"{r['var']} {r['coef']:+.4f}" for r in rows if r["var"] in xs)
           + (f"，ICC={icc:.3f}" if icc is not None else ""))
    return Finding(fid, "mixed", status, why,
                   {"y": y, "x": xs, "group": group, "rows": rows,
                    "var_group": var_group, "var_resid": var_resid, "icc": icc,
                    "n": len(d), "n_groups": int(d[group].nunique()),
                    "random_slope": re_slope},
                   warnings=warnings)
