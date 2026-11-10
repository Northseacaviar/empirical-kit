"""跨组系数差异检验：两组的斜率到底一不一样。

为什么用交互项而不是分组回归：分组回归的两个系数各自有标准误，
但「差异是否显著」需要两者的协方差，而 split-sample 把它丢掉了。
全样本加 x×group 交互项等价于 Chow 检验，且能带固定效应与聚类标准误。

边界：只处理两组（by 取值恰为两个）。多组请先二值化，或用 subsample 分别报。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def group_test(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "group_test")
    if pf is None:
        return Finding(fid, "group_test", "fail", "未安装 pyfixest", {})
    y, by = cfg["y"], cfg["by"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    controls = cfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = cfg.get("fe") or []
    clusters = cfg.get("cluster") or []
    focus = cfg.get("focus") or (xs[0] if xs else None)

    empty = {"y": y, "x": focus, "by": by, "groups": [], "base_coef": None,
             "diff": None, "se_diff": None, "t": None, "p": None, "f": None,
             "p_f": None, "coef_low": None, "coef_high": None, "n": 0}
    need = [y, by] + list(xs) + list(controls) + list(fe) + list(clusters)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "group_test", "fail", f"列不存在：{missing}", empty)

    d = df.dropna(subset=need).copy()
    vals = sorted(d[by].unique())
    if len(vals) != 2:
        return Finding(fid, "group_test", "fail",
                       f"分组变量 {by} 有 {len(vals)} 个取值：本项只做两组比较。"
                       "多组请先二值化，或用 subsample 分别报系数",
                       {**empty, "values": [str(v) for v in vals]})
    lo, hi = vals
    d["_grp"] = (d[by] == hi).astype(int)
    if not focus:
        return Finding(fid, "group_test", "fail", "没有指定 x（要看哪个系数在两组间的差异）", empty)

    inter = f"{focus}:_grp"
    rhs = [focus, inter] + [x for x in xs if x != focus] + list(controls)
    fml = f"{y} ~ 1 + " + " + ".join(rhs)
    if fe:
        fml += " | " + " + ".join(fe)
    vcov = ({"CRV1": clusters[0]} if len(clusters) == 1
            else {"CRV1": "+".join(clusters)}) if clusters else "hetero"
    try:
        m = pf.feols(fml, data=d, vcov=vcov)
        td = m.tidy()
    except Exception as exc:
        return Finding(fid, "group_test", "fail",
                       f"估计失败：{type(exc).__name__}: {exc}", {**empty, "formula": fml})

    if inter not in td.index:
        return Finding(fid, "group_test", "fail",
                       f"交互项 {inter} 没进模型（{by} 或 {focus} 在组内没有变异）",
                       {**empty, "formula": fml})

    b_inter = float(td.loc[inter, "Estimate"])
    se_inter = float(td.loc[inter, "Std. Error"])
    t_inter = float(td.loc[inter, "t value"])
    p_inter = float(td.loc[inter, "Pr(>|t|)"])
    b_base = float(td.loc[focus, "Estimate"]) if focus in td.index else float("nan")

    # 联合检验（与 Chow 的 F 形式一致）
    try:
        idx = list(td.index)
        R = np.zeros((1, len(idx)))
        R[0, idx.index(inter)] = 1
        wald = m.wald_test(R=R, distribution="chi2")
        g = int(min(m._G)) if getattr(m, "_G", None) else None
        if g:
            from scipy import stats as sps
            df2 = max(g - 1, 1)
            f_stat = float(wald.stat) / 1
            p_f = float(sps.f.sf(f_stat, 1, df2))
        else:
            f_stat, p_f = float(wald.stat), float(wald.pvalue)
    except Exception:
        f_stat, p_f = t_inter ** 2, p_inter

    status, warnings = "ok", []
    if p_f < 0.05:
        status = "warn"
        warnings.append(
            f"{focus} 的系数在 {by}={lo} 与 {by}={hi} 两组间显著不同"
            f"（差 {b_inter:+.4f}，F={f_stat:.2f}，p={p_f:.4f}）："
            "正文要分组报，不能用全样本的单一系数")
    else:
        warnings.append(f"不能拒绝两组系数相同（p={p_f:.3f}）：全样本系数可以照用")

    why = (f"{focus} 的两组差异：{by}={lo} 组 {b_base:+.4f}，{by}={hi} 组 "
           f"{b_base + b_inter:+.4f}，差 {b_inter:+.4f}（F={f_stat:.2f}，p={p_f:.4f}）")
    return Finding(fid, "group_test", status, why,
                   {"y": y, "x": focus, "by": by, "groups": [str(lo), str(hi)],
                    "base_coef": b_base, "diff": b_inter, "se_diff": se_inter,
                    "t": t_inter, "p": p_inter, "f": f_stat, "p_f": p_f,
                    "coef_low": b_base, "coef_high": b_base + b_inter,
                    "formula": fml, "n": int(m._N), "vcov": vcov if isinstance(vcov, str) else "cluster"},
                   warnings=warnings)
