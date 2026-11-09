"""单样本均值检验：事件研究的表 1（平均 CAR 与 t 值）。

报告均值、中位数、标准差、t、正收益占比。中位数与正收益占比同时给，
是因为均值可能被少数极端值拉动——三者方向一致才说明结论不是尾部驱动。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding


def mean_test(df: pd.DataFrame, var: str, fid: str | None = None, min_n: int = 30,
              by: str | None = None, scale: float = 1.0, unit: str = "") -> Finding:
    fid = fid or f"mean_{var}"
    if var not in df.columns:
        near = [c for c in df.columns if var.split("_")[0] in c][:8]
        return Finding(fid, "mean_test", "fail", f"列不存在：{var}；可能是 {near}", {})

    if by:
        # 分组各自的均值检验
        levels = sorted(pd.unique(df[by].dropna()), key=lambda x: (str(type(x)), x))
        rows = []
        for lv in levels:
            s = df.loc[df[by] == lv, var].dropna().astype(float)
            if len(s) < 2:
                continue
            t, p = stats.ttest_1samp(s, 0)
            rows.append({"label": str(lv), "n": int(len(s)), "mean": float(s.mean()) * scale,
                         "median": float(s.median()) * scale, "sd": float(s.std(ddof=1)) * scale,
                         "t": float(t), "p": float(p),
                         "pct_pos": float((s > 0).mean())})
        if not rows:
            return Finding(fid, "mean_test", "fail", "分组后没有可用观测", {})
        status = "warn" if min(r["n"] for r in rows) < min_n else "ok"
        why = "；".join(f"{r['label']}: {r['mean']:+.3f}（t={r['t']:.2f}, n={r['n']}）" for r in rows)
        return Finding(fid, "mean_test", status, f"{var} 分组均值：{why}",
                       {"var": var, "by": by, "groups": rows})

    s = df[var].dropna().astype(float)
    n = len(s)
    if n < 2:
        return Finding(fid, "mean_test", "fail", "可用观测不足 2 个", {"n": n})
    t, p = stats.ttest_1samp(s, 0)
    mean, sd = float(s.mean()), float(s.std(ddof=1))
    se = sd / np.sqrt(n)

    status, warnings = "ok", []
    if n < min_n:
        status = "warn"
        warnings.append(f"小样本：n={n} < {min_n}")
    # 均值与中位数方向不一致 → 可能由尾部驱动
    med = float(s.median())
    if mean * med < 0:
        warnings.append(f"均值 {mean:+.3f} 与中位数 {med:+.3f} 方向不一致，检查极端值")
        status = "warn"

    why = (f"{var}：均值 {mean * scale:+.3f}，中位数 {med * scale:+.3f}，t={t:.2f}，"
           f"n={n}，正收益占比 {(s > 0).mean():.1%}" + (f"，单位 {unit}" if unit else ""))
    return Finding(fid, "mean_test", status, why,
                   {"var": var, "n": int(n), "mean": mean * scale, "median": med * scale,
                    "sd": sd * scale, "se": float(se) * scale, "t": float(t), "p": float(p),
                    "ci95": [float(mean - 1.96 * se) * scale, float(mean + 1.96 * se) * scale],
                    "pct_pos": float((s > 0).mean()), "scale": scale, "unit": unit},
                   warnings=warnings)
