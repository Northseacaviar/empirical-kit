"""留一法：逐个剔除观测，看组间差的最大位移。

O(n) 解析计算，不用真的循环 n 次重估：
去掉组 k 的一个观测后，均值变化 = (x_i - mean_k) / (n_k - 1)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding


def leave_one_out(df: pd.DataFrame, var: str, by: str, fid: str | None = None,
                  threshold: float = 0.2) -> Finding:
    fid = fid or f"loo_{var}_by_{by}"
    missing = [c for c in (var, by) if c not in df.columns]
    if missing:
        return Finding(fid, "leave_one_out", "fail", f"列不存在：{missing}", {"missing": missing})

    d = df[[var, by]].dropna()
    levels = sorted(pd.unique(d[by]), key=lambda x: (str(type(x)), x))
    if len(levels) != 2:
        return Finding(fid, "leave_one_out", "fail", f"{by} 需要两组，实际 {len(levels)} 组", {})

    g0 = d.loc[d[by] == levels[0], var].astype(float).to_numpy()
    g1 = d.loc[d[by] == levels[1], var].astype(float).to_numpy()
    n0, n1 = len(g0), len(g1)
    if n0 < 2 or n1 < 2:
        return Finding(fid, "leave_one_out", "fail", "组样本不足", {"n": [n0, n1]})

    m0, m1 = g0.mean(), g1.mean()
    diff = m1 - m0

    # 去掉组1某点：Δdiff = (m1 - x)/(n1-1)；去掉组0某点：Δdiff = (x - m0)/(n0-1)
    d1 = (m1 - g1) / (n1 - 1)
    d0 = (g0 - m0) / (n0 - 1)
    shift = np.concatenate([d1, d0])
    abs_shift = np.abs(shift)
    idx = int(np.argmax(abs_shift))
    max_shift = float(shift[idx])
    drop = diff - shift[idx]

    rel = abs(max_shift) / abs(diff) if diff != 0 else float("inf")
    warnings, status = [], "ok"
    if rel > threshold:
        status = "warn"
        warnings.append(f"单点影响过大：剔除一个观测后组间差移动 {max_shift:+.3f}"
                        f"（占原差 {rel:.1%}）")
    if max_shift * diff < 0 and abs(drop) > abs(diff):
        status = "warn"
        warnings.append("剔除最敏感观测后组间差绝对值反而变大，说明原差被稀释")

    why = (f"{var}：原差 {diff:+.3f}；最敏感单点使差变为 {drop:+.3f}"
           f"（移动 {max_shift:+.3f}，占 {rel:.1%}）")
    return Finding(fid, "leave_one_out", status, why,
                   {"var": var, "by": by, "diff": float(diff), "max_shift": max_shift,
                    "diff_without_extreme": float(drop), "relative_shift": float(rel),
                    "n": [n0, n1], "worst_obs_group": str(levels[1] if idx < n1 else levels[0])},
                   warnings=warnings)
