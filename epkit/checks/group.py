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
               min_n: int = 30, winsor: list[float] | None = None,
               drop_where: str | None = None, scale: float = 1.0,
               unit: str = "") -> Finding:
    """组间均值差。

    winsor     : [lo, hi] 分位点，先缩尾再检验（记原作 vs 缩尾后两个差）
    drop_where : pandas query，先剔除再检验（稳健性：去掉大市值、ST、次新等）
    """
    fid = fid or f"diff_{var}_by_{group}"
    missing = [c for c in (var, group) if c not in df.columns]
    if missing:
        near = [c for c in df.columns if any(m.split("_")[0] in c for m in missing)][:8]
        return Finding(fid, "group_diff", "fail",
                       f"列不存在：{missing}；可能是 {near}",
                       {"var": var, "by": group, "missing": missing, "n_cols": len(df.columns)})

    d = df.copy()
    n_raw = len(d)
    note = ""
    if drop_where:
        mask = d.eval(drop_where).fillna(False).astype(bool)
        d = d[~mask]
        note = f"；先剔除 {n_raw - len(d)} 条（{drop_where}）"
    d = d[[var, group]].dropna()
    if winsor:
        lo, hi = winsor
        a, b = d[var].quantile(lo), d[var].quantile(hi)
        d[var] = d[var].clip(a, b)
        note += f"；缩尾到 [{a:.3f}, {b:.3f}]"
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
            {"label": str(levels[0]), "n": int(len(g0)), "mean": float(g0.mean()) * scale,
             "sd": float(g0.std(ddof=1)) * scale, "median": float(g0.median()) * scale},
            {"label": str(levels[1]), "n": int(len(g1)), "mean": float(g1.mean()) * scale,
             "sd": float(g1.std(ddof=1)) * scale, "median": float(g1.median()) * scale},
        ],
        "diff": diff * scale,
        "scale": scale, "unit": unit,
        "t": float(t_main), "p": float(p_main), "t_test": "welch",
        "t_equal": float(t_eq), "p_equal": float(p_eq),
        "t_welch": float(t_w), "p_welch": float(p_w), "ci95_welch": [float(ci[0]), float(ci[1])],
        "n_before_robustness": n_raw,
        "winsor": winsor,
        "drop_where": drop_where,
    }
    why = (f"差 = {levels[1]} - {levels[0]}：{diff * scale:.3f}，t={t_main:.2f}"
           f"（Welch，组 n={len(g0)}/{len(g1)}）{note}")
    if unit:
        why += f"，单位 {unit}"
    return Finding(fid, "group_diff", status, why, data, warnings=warnings)
