"""组间差异检验：均值差、t 值、置信区间。

主口径用 Welch（不假设等方差），等方差 t 作对照一并给出：
两组样本量差一倍以上时，等方差 t 会系统性偏大——实测同一份数据
从 -8.87 偏到 -9.68，足以改变结论的严格程度。两种都给，写论文时选一个并写明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding


def group_diff(df: pd.DataFrame, var: str, group: str, fid: str | None = None,
               min_n: int = 30) -> Finding:
    d = df[[var, group]].dropna()
    fid = fid or f"diff_{var}_by_{group}"
    # 组顺序固定为取值升序：否则 diff 的符号取决于数据里谁先出现，
    # 同一份数据换个行序就会把"负效应"读成"正效应"。
    levels = sorted(pd.unique(d[group]), key=lambda x: (str(type(x)), x))
    if len(levels) != 2:
        return Finding(fid, "group_diff", "fail",
                       f"分组变量 {group} 有 {len(levels)} 个取值（{levels[:5]}），本检验需要两组",
                       {"var": var, "group": group, "levels": [str(x) for x in levels]})

    g0 = d.loc[d[group] == levels[0], var].astype(float)
    g1 = d.loc[d[group] == levels[1], var].astype(float)
    if len(g0) < 2 or len(g1) < 2:
        return Finding(fid, "group_diff", "fail", "某一组样本不足 2 个观测",
                       {"var": var, "n": [len(g0), len(g1)]})

    t_eq, p_eq = stats.ttest_ind(g1, g0, equal_var=True)
    t_w, p_w = stats.ttest_ind(g1, g0, equal_var=False)
    # 主口径 = Welch
    t_main, p_main = t_w, p_w
    diff = float(g1.mean() - g0.mean())
    se_w = float(np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0)))
    ci = (diff - 1.96 * se_w, diff + 1.96 * se_w)

    warnings, status = [], "ok"
    if min(len(g0), len(g1)) < min_n:
        status = "warn"
        warnings.append(f"小样本：组样本量 {min(len(g0), len(g1))} < {min_n}，t 值近似性差")
    if np.isnan(t_eq):
        status = "warn"
        warnings.append("组内方差为 0，t 值无定义")

    data = {
        "var": var, "by": group,
        "groups": [
            {"label": str(levels[0]), "n": int(len(g0)), "mean": float(g0.mean()),
             "sd": float(g0.std(ddof=1)), "median": float(g0.median())},
            {"label": str(levels[1]), "n": int(len(g1)), "mean": float(g1.mean()),
             "sd": float(g1.std(ddof=1)), "median": float(g1.median())},
        ],
        "diff": diff,
        "t": float(t_main), "p": float(p_main), "t_test": "welch",
        "t_equal": float(t_eq), "p_equal": float(p_eq),
        "t_welch": float(t_w), "p_welch": float(p_w), "ci95_welch": [float(ci[0]), float(ci[1])],
        "unit": "pp" if var.startswith("car") else "",
    }
    why = (f"差 = {levels[1]} - {levels[0]}：{diff:.3f}，t={t_main:.2f}（Welch，组 n={len(g0)}/{len(g1)}）")
    if data["unit"]:
        why += "，单位 个百分点"
    return Finding(fid, "group_diff", status, why, data, warnings=warnings)
