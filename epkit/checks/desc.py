"""描述统计与相关矩阵。

描述统计同时给分位数，是因为均值 + 标准差在偏态变量（补贴金额、市值）上会骗人；
写论文的变量表也要求分位数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding


def _stats(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return {"n": 0}
    return {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std(ddof=1)),
            "p25": float(s.quantile(0.25)), "median": float(s.median()),
            "p75": float(s.quantile(0.75)), "min": float(s.min()), "max": float(s.max())}


def describe(df: pd.DataFrame, vars: list[str], fid: str | None = None,
             by: str | None = None, scale: float = 1.0,
             scale_fn=None) -> Finding:
    """scale_fn(var) 可逐变量给换算系数；不给就统一用 scale。"""
    if scale_fn is None:
        def scale_fn(_v):
            return scale
    fid = fid or "describe"
    missing = [v for v in vars if v not in df.columns]
    if missing:
        near = [c for c in df.columns if any(m in c for m in missing)][:8]
        return Finding(fid, "describe", "fail", f"列不存在：{missing}；可能是 {near}", {})
    if not vars:
        return Finding(fid, "describe", "fail", "没有指定变量", {})

    def sc(st: dict, k: float) -> dict:
        return {x: (v * k if x != "n" else v) for x, v in st.items()}

    rows, warnings, status = [], [], "ok"
    if by:
        levels = sorted(pd.unique(df[by].dropna()), key=lambda x: (str(type(x)), x))
        for v in vars:
            item = {"var": v, "groups": []}
            for lv in levels:
                st = _stats(df.loc[df[by] == lv, v])
                if st.get("n", 0) == 0:
                    continue
                item["groups"].append({"label": str(lv), **sc(st, scale_fn(v))})
            rows.append(item)
    else:
        for v in vars:
            st = _stats(df[v])
            if st.get("n", 0) < 2:
                warnings.append(f"{v} 可用观测不足")
                status = "warn"
                continue
            rows.append({"var": v, **sc(st, scale_fn(v))})

    # 缺失率高的变量提示
    for v in vars:
        if len(df) and df[v].isna().mean() > 0.1:
            warnings.append(f"{v} 缺失率 {df[v].isna().mean():.1%}")
            status = "warn"

    why = f"{len(vars)} 个变量的描述统计" + (f"（按 {by} 分组）" if by else "")
    return Finding(fid, "describe", status, why,
                   {"vars": rows, "by": by, "scale": scale, "n_rows": int(len(df))},
                   warnings=warnings)


def corr(df: pd.DataFrame, vars: list[str], fid: str | None = None,
         method: str = "pearson", threshold: float = 0.8) -> Finding:
    fid = fid or "corr"
    missing = [v for v in vars if v not in df.columns]
    if missing:
        near = [c for c in df.columns if any(m in c for m in missing)][:8]
        return Finding(fid, "corr", "fail", f"列不存在：{missing}；可能是 {near}", {})
    if len(vars) < 2:
        return Finding(fid, "corr", "fail", "相关矩阵至少需要两个变量", {})

    d = df[vars].astype(float)
    m = d.corr(method=method)
    pairs = []
    for i, a in enumerate(vars):
        for b in vars[i + 1:]:
            r = m.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"a": a, "b": b, "r": float(r)})

    status, warnings = "ok", []
    if pairs:
        status = "warn"
        warnings += [f"{p['a']} 与 {p['b']} 相关 {p['r']:+.3f}，同时入模可能共线" for p in pairs]
    n_used = int(d.dropna().shape[0])
    if n_used < len(d) * 0.8:
        warnings.append(f"成对删除后仅 {n_used}/{len(d)} 行同时可用")
        status = "warn"

    return Finding(fid, "corr", status,
                   f"{method} 相关矩阵（{len(vars)} 个变量，{n_used} 行完整）；"
                   f"|r|≥{threshold} 的有 {len(pairs)} 对",
                   {"vars": vars, "method": method, "matrix": [[None if pd.isna(x) else float(x)
                                                                for x in row] for row in m.values],
                    "high_pairs": pairs, "n_complete": n_used},
                   warnings=warnings)
