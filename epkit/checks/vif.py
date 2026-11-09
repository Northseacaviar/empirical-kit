"""共线性诊断：VIF。

有固定效应时先做组内去均值再算——否则 FE 吸收掉的组间差异会污染判断。
经验阈值：VIF > 10 视为严重共线，5~10 提示关注。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding


def vif(df: pd.DataFrame, vars: list[str], fid: str | None = None,
        fe: list[str] | None = None, warn_at: float = 10.0) -> Finding:
    fid = fid or "vif"
    missing = [v for v in vars if v not in df.columns]
    if missing:
        near = [c for c in df.columns if any(m in c for m in missing)][:8]
        return Finding(fid, "vif", "fail", f"列不存在：{missing}；可能是 {near}", {})
    if len(vars) < 2:
        return Finding(fid, "vif", "fail", "VIF 至少需要两个变量", {})

    fe = fe or []
    d = df[vars + fe].dropna().copy()
    if len(d) < len(vars) + 2:
        return Finding(fid, "vif", "fail", f"可用观测 {len(d)} 过少", {})
    for f in fe:
        for v in vars:
            d[v] = d[v] - d.groupby(f)[v].transform("mean")

    items, warnings, status = [], [], "ok"
    for v in vars:
        y = d[v].to_numpy(dtype=float)
        X = d[[c for c in vars if c != v]].to_numpy(dtype=float)
        if X.shape[1] == 0 or np.var(y) == 0:
            items.append({"var": v, "vif": None})
            continue
        X1 = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ coef
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
        v_ = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
        items.append({"var": v, "vif": v_, "r2": r2})
        if v_ > warn_at:
            warnings.append(f"{v} 的 VIF={v_:.1f} > {warn_at}，与其他解释变量严重共线")
            status = "warn"
        elif v_ > 5:
            warnings.append(f"{v} 的 VIF={v_:.1f}，需要关注")

    worst = max((i for i in items if i["vif"]), key=lambda i: i["vif"], default=None)
    why = (f"{len(vars)} 个变量的 VIF" + (f"（组内去均值：{'/'.join(fe)}）" if fe else "")
           + (f"；最大 {worst['var']}={worst['vif']:.1f}" if worst else ""))
    return Finding(fid, "vif", status, why,
                   {"items": items, "fe": fe, "warn_at": warn_at, "n": int(len(d))},
                   warnings=warnings)
