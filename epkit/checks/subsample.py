"""子样本网格：按年、按行业、按规模把主回归重跑一遍。

为什么要单独做：分组交互项只能回答"斜率有没有差异"，回答不了
"哪些组里根本不成立"。审稿人问异质性，通常要看到每组的系数与样本量。

口径：同一套公式、同一套推断口径，只换样本。样本太小的组照样报，
但标 warn —— 一个 12 个观测的组算出 t=3 不是发现，是指纹。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding
from ..vars import evaluate

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def subsample(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "subsample")
    if pf is None:
        return Finding(fid, "subsample", "fail", "未安装 pyfixest", {})
    y, x, by = cfg["y"], cfg["x"], cfg["by"]
    controls = cfg.get("controls") or []
    fe = cfg.get("fe") or []
    clusters = cfg.get("cluster") or []
    min_n = int(cfg.get("min_n", 30))
    max_groups = int(cfg.get("max_groups", 40))

    if by not in df.columns:
        try:
            g = evaluate(by, df)
            df = df.assign(**{"_by": pd.Series(g, index=df.index)})
            by = "_by"
        except Exception as exc:
            return Finding(fid, "subsample", "fail",
                           f"分组变量 {cfg['by']} 不存在且无法求值：{exc}", {})

    vals = [v for v in df[by].dropna().unique()]
    if len(vals) > max_groups:
        return Finding(fid, "subsample", "fail",
                       f"分组 {len(vals)} 个超过上限 {max_groups}："
                       "细到这种程度每组都没有样本，先粗分或换分位分组", {})

    rhs = [x] + list(controls)
    fml = f"{y} ~ 1 + " + " + ".join(rhs)
    if fe:
        fml += " | " + " + ".join(fe)
    # 没有聚类列时退回异方差稳健，而不是 pyfixest 默认的 iid——
    # 否则「主表聚类、子样本表 iid」就是典型的口径漂移
    vcov = ({"CRV1": clusters[0]} if len(clusters) == 1 else {"CRV1": "+".join(clusters)}) \
        if clusters else "hetero"

    rows, errors = [], []
    for v in sorted(vals, key=lambda z: str(z)):
        sub = df[df[by] == v]
        n = len(sub)
        rec = {"group": str(v), "n": n}
        if n < min_n:
            rec.update({"coef": None, "note": f"样本 {n} < {min_n}，不出估计"})
            rows.append(rec)
            continue
        try:
            m = pf.feols(fml, data=sub, vcov=vcov)
            td = m.tidy()
            if x not in td.index:
                rec.update({"coef": None, "note": "系数未提取"})
            else:
                rec.update({"coef": float(td.loc[x, "Estimate"]),
                            "se": float(td.loc[x, "Std. Error"]),
                            "t": float(td.loc[x, "t value"]),
                            "p": float(td.loc[x, "Pr(>|t|)"])})
        except Exception as exc:
            rec.update({"coef": None, "note": f"{type(exc).__name__}: {exc}"})
            errors.append(f"{v}: {type(exc).__name__}")
        rows.append(rec)

    est = [r for r in rows if r.get("coef") is not None]
    if not est:
        return Finding(fid, "subsample", "fail",
                       f"没有任何一组能出估计（{len(errors)} 组报错）", {"rows": rows})

    coefs = np.array([r["coef"] for r in est])
    sig = [r for r in est if r.get("p") is not None and r["p"] < 0.05]
    flip = bool(coefs.min() < 0 < coefs.max())
    warnings, status = [], "ok"
    if flip:
        status = "warn"
        warnings.append(f"分组系数符号翻转（{coefs.min():+.4f} ~ {coefs.max():+.4f}）："
                        "正文不能只报一个方向，要么交代异质性来源，要么改用交互项")
    thin = [r["group"] for r in rows if r.get("coef") is None]
    if thin:
        warnings.append(f"{len(thin)} 组样本不足未出估计：" + "、".join(thin[:8]))
    if len(sig) <= len(est) / 3:
        status = "warn" if status == "ok" else status
        warnings.append(f"{len(est)} 组里只有 {len(sig)} 组显著：主结论不是每类子样本都成立")

    why = (f"按 {cfg['by']} 分 {len(rows)} 组，{len(est)} 组出估计；"
           f"系数 {coefs.min():+.4f} ~ {coefs.max():+.4f}，显著 {len(sig)} 组")
    return Finding(fid, "subsample", status, why,
                   {"y": y, "x": x, "by": cfg["by"], "rows": rows,
                    "vcov": vcov if isinstance(vcov, str) else "cluster",
                    "formula": fml}, warnings=warnings)
