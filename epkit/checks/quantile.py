"""分位数回归：在分布的不同位置上重新问一遍"x 对 y 的影响有多大"。

为什么值得做：均值回归只说平均效应。A 股里常见的形态是
"对小市值公司影响大、对大的几乎没影响"，均值回归会把这条信息抹平。

实现：statsmodels QuantReg。高维固定效应用组内去均值近似（去均值后再估），
这是线性分位数回归里被普遍接受的近似做法；只有少量固定效应时才等价于哑变量法。
结果里写明这一点，别让它当成精确的固定效应分位数回归。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def _within(df: pd.DataFrame, cols: list[str], fe: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c].astype(float)
        for g in fe:
            s = s - s.groupby(out[g]).transform("mean")
        out[c] = s
    return out


def quantile(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "quantile")
    try:
        import statsmodels.api as sm
    except ImportError:  # pragma: no cover
        return Finding(fid, "quantile", "fail", "未安装 statsmodels", {})

    y, x = cfg["y"], cfg["x"]
    controls = cfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = cfg.get("fe") or []
    if fe and not isinstance(fe[0], str):
        fe = []
    qs = cfg.get("q", [0.1, 0.25, 0.5, 0.75, 0.9])
    cols = [y, x] + list(controls)
    need = cols + fe
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "quantile", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=cols).copy()
    if fe:
        d = _within(d, cols, fe)
    X = sm.add_constant(d[[x] + list(controls)].astype(float))
    yy = d[y].astype(float)

    rows, errors = [], []
    for q in qs:
        try:
            res = sm.QuantReg(yy, X).fit(q=q)
            rows.append({"q": float(q), "coef": float(res.params[x]),
                         "se": float(res.bse[x]), "t": float(res.tvalues[x]),
                         "p": float(res.pvalues[x]), "n": int(res.nobs)})
        except Exception as exc:
            errors.append(f"q={q}: {type(exc).__name__}")

    if not rows:
        return Finding(fid, "quantile", "fail",
                       f"所有分位估计失败：{errors}", {})

    coefs = np.array([r["coef"] for r in rows])
    sig = [r for r in rows if r["p"] < 0.05]
    spread = float(coefs.max() - coefs.min())
    monotone = bool(np.all(np.diff(coefs) >= -1e-8) or np.all(np.diff(coefs) <= 1e-8))
    warnings, status = [], "ok"
    if coefs.min() < 0 < coefs.max():
        status = "warn"
        warnings.append(f"分位系数符号翻转（{coefs.min():+.4f} ~ {coefs.max():+.4f}）："
                        "均值回归给出的单一系数会掩盖方向差异")
    if spread > 3 * abs(float(np.median(coefs))) and abs(np.median(coefs)) > 1e-9:
        warnings.append(f"最高与最低分位的系数相差 {spread:.4f}，"
                        "约为中位系数的 3 倍以上：效应沿分布变化明显")
    if not monotone:
        warnings.append("系数随分位不单调：不要用「越大越受影响」这类线性表述")
    if fe:
        warnings.append("固定效应用组内去均值近似，不是哑变量法的精确解；"
                        "结论只在系数方向与相对大小上可靠")

    why = (f"分位回归 {len(rows)} 个分位：{qs[0]:.2f} 分位 {rows[0]['coef']:+.4f} → "
           f"{qs[-1]:.2f} 分位 {rows[-1]['coef']:+.4f}，显著 {len(sig)} 个")
    return Finding(fid, "quantile", status, why,
                   {"y": y, "x": x, "rows": rows, "fe": fe,
                    "approximation": "within-demeaning" if fe else "none"},
                   warnings=warnings)
