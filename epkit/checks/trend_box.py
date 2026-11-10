"""时间趋势图的数据：按时间（和可选的组）给均值与 95% 区间。

为什么要有：论文的第二张图几乎总是「组间差异随时间怎么变」。
均值的置信区间用 Welch 口径（与大样本正态近似一致），分组时各画一条线。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding


def trend(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "trend")
    var = cfg["var"]
    by = cfg["by"]
    group = cfg.get("group")
    need = [c for c in (var, by, group) if c]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "trend", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need)
    if d.empty:
        return Finding(fid, "trend", "fail", "没有可用观测", {})

    def _agg(sub: pd.DataFrame) -> dict:
        s = sub[var].astype(float)
        n = len(s)
        if n < 2:
            return {"n": n, "mean": float(s.mean()) if n else None, "se": None,
                    "ci_low": None, "ci_high": None}
        se = float(s.std(ddof=1) / np.sqrt(n))
        return {"n": n, "mean": float(s.mean()), "se": se,
                "ci_low": float(s.mean() - 1.96 * se),
                "ci_high": float(s.mean() + 1.96 * se)}

    points = []
    if group:
        for gval, sub in d.groupby(group):
            for tval, s2 in sub.groupby(by):
                rec = _agg(s2)
                rec.update({"time": str(tval), "group": str(gval)})
                points.append(rec)
    else:
        for tval, s2 in d.groupby(by):
            rec = _agg(s2)
            rec.update({"time": str(tval), "group": ""})
            points.append(rec)

    thin = [p for p in points if p["n"] < 10]
    warnings = []
    if thin:
        warnings.append(f"{len(thin)} 个时间点的样本少于 10：区间会很宽，图上会显得抖动")
    status = "warn" if thin else "ok"
    why = (f"{var} 按 {by}" + (f" × {group}" if group else "") +
           f"：{len(points)} 个点，均值 {min(p['mean'] for p in points if p['mean'] is not None):.3f}"
           f" ~ {max(p['mean'] for p in points if p['mean'] is not None):.3f}")
    return Finding(fid, "trend", status, why,
                   {"var": var, "by": by, "group": group, "points": points},
                   warnings=warnings)


def box_plot(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "box")
    var, by = cfg["var"], cfg["by"]
    missing = [c for c in (var, by) if c not in df.columns]
    if missing:
        return Finding(fid, "box", "fail", f"列不存在：{missing}", {})
    d = df.dropna(subset=[var, by])
    groups = []
    for gval, sub in d.groupby(by):
        s = sub[var].astype(float)
        if len(s) < 5:
            groups.append({"group": str(gval), "n": int(len(s))})
            continue
        groups.append({"group": str(gval), "n": int(len(s)),
                       "min": float(s.min()), "q1": float(s.quantile(0.25)),
                       "median": float(s.median()), "q3": float(s.quantile(0.75)),
                       "max": float(s.max()), "mean": float(s.mean())})
    thin = [g for g in groups if g["n"] < 20]
    why = f"{var} 按 {by}：{len(groups)} 组" + (f"，{len(thin)} 组样本 < 20" if thin else "")
    return Finding(fid, "box", "warn" if thin else "ok", why,
                   {"var": var, "by": by, "groups": groups},
                   warnings=[f"{g['group']} 只有 {g['n']} 个观测，箱线图意义有限" for g in thin])
