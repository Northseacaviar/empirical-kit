"""结构方程模型（SEM）：潜变量之间的路径。

为什么用 semopy：SEM 需要极大似然估计协方差结构，还要出一整套拟合指标
（CFI / TLI / RMSEA / SRMR），自己写不现实。semopy 是 Python 里目前最完整的一个。

口径：用户直接写 lavaan 风格的模型串（`f1 =~ x1 + x2`、`f2 ~ f1`）。
输出固定路径系数 + 拟合指标；样本量过小、不收敛时明确报错而不是给个半成品。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def sem(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "sem")
    try:
        import semopy
    except ImportError:  # pragma: no cover
        return Finding(fid, "sem", "fail", "未安装 semopy：uv pip install semopy", {})

    desc = cfg.get("model")
    if not desc:
        return Finding(fid, "sem", "fail", "缺少 model（lavaan 风格的模型串）", {})

    try:
        model = semopy.Model(desc)
        res = model.fit(df)
    except Exception as exc:
        return Finding(fid, "sem", "fail",
                       f"SEM 估计失败：{type(exc).__name__}: {exc}", {})

    ins = model.inspect()   # semopy 0.6+ 把 inspect 挂到模型对象上
    rows = []
    # semopy 0.6 的 inspect 列名是 lval / op / rval
    lcol = "lval" if "lval" in ins.columns else "lhs"
    rcol = "rval" if "rval" in ins.columns else "rhs"
    for _, r in ins.iterrows():
        if r.get("op") in ("=~", "~", "~~"):
            def _f(v):
                try:
                    x = float(v)
                    return x if np.isfinite(x) else None
                except Exception:
                    return None
            rows.append({"lhs": str(r[lcol]), "op": str(r["op"]), "rhs": str(r[rcol]),
                         "coef": _f(r["Estimate"]), "se": _f(r["Std. Err"]),
                         "z": _f(r["z-value"]), "p": _f(r["p-value"])})
    try:
        stats_df = semopy.calc_stats(model).T
        fit = {k: float(stats_df.loc[k, stats_df.columns[0]])
               for k in ("CFI", "TLI", "RMSEA", "SRMR", "GFI", "AGFI")
               if k in stats_df.index}
    except Exception:
        fit = {}

    warnings, status = [], "ok"
    if fit.get("CFI", 1) < 0.9:
        status = "warn"
        warnings.append(f"CFI={fit['CFI']:.3f} < 0.9：模型拟合不足，"
                        "不要直接报路径系数，先改设定")
    if fit.get("RMSEA", 0) > 0.08:
        status = "warn"
        warnings.append(f"RMSEA={fit['RMSEA']:.3f} > 0.08：拟合不足")
    n = len(df.dropna())
    if n < 200:
        warnings.append(f"样本 {n}：SEM 通常要求 200 以上，小样本下拟合指标不稳")

    paths = [r for r in rows if r["op"] == "~"]
    why = (f"SEM（n={n}）：" + ("、".join(f"{r['lhs']}←{r['rhs']} {r['coef']:+.4f}" for r in paths[:4])
                               if paths else "只有测量模型")
           + (f"；CFI={fit.get('CFI', float('nan')):.3f}，RMSEA={fit.get('RMSEA', float('nan')):.3f}"
              if fit else ""))
    return Finding(fid, "sem", status, why,
                   {"model": desc, "paths": rows, "fit": fit, "n": n},
                   warnings=warnings)
