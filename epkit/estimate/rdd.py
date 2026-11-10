"""断点回归（RDD）：局部多项式 + 数据驱动带宽。

为什么用 rdrobust 而不是自己写：CCT（Calonico-Cattaneo-Titiunik）的带宽选择、
偏误校正与稳健置信区间是一整套方法，自己实现几乎必然在细节上走样，
而审稿人正盯着这几个数。这里只做 spec 到 rdrobust 的适配与结果解读。

配套必查：断点处密度是否连续（用 pileup 项的密度检验，或 rdrobust 的 rddensity）。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def rdd(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "rdd")
    try:
        from rdrobust import rdrobust
    except ImportError:  # pragma: no cover
        return Finding(fid, "rdd", "fail", "未安装 rdrobust：uv pip install rdrobust", {})

    y, x = cfg["y"], cfg["x"]
    cut = float(cfg.get("cut", 0.0))
    need = [y, x]
    for k in ("cluster", "covs"):
        v = cfg.get(k)
        if v:
            need += list(v) if isinstance(v, (list, tuple)) else [v]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "rdd", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    kwargs: dict[str, Any] = {"c": cut}
    if cfg.get("p"):
        kwargs["p"] = int(cfg["p"])
    if cfg.get("bwselect"):
        kwargs["bwselect"] = cfg["bwselect"]
    if cfg.get("bw"):
        kwargs["h"] = float(cfg["bw"])
    if cfg.get("cluster"):
        cl = cfg["cluster"]
        kwargs["cluster"] = d[cl[0] if isinstance(cl, (list, tuple)) else cl].values
    if cfg.get("covs"):
        cv = cfg["covs"]
        kwargs["covs"] = d[cv if isinstance(cv, list) else [cv]].values
    if cfg.get("fuzzy"):
        fz = cfg["fuzzy"]
        kwargs["fuzzy"] = d[fz if isinstance(fz, str) else fz[0]].astype(float).values

    try:
        r = rdrobust(d[y].values, d[x].values, **kwargs)
    except Exception as exc:
        return Finding(fid, "rdd", "fail",
                       f"rdrobust 估计失败：{type(exc).__name__}: {exc}", {})

    def _get(*names, idx=None):
        for n in names:
            v = getattr(r, n, None)
            if v is not None:
                try:
                    arr = np.asarray(v, dtype=float).ravel()
                    return float(arr[idx]) if idx is not None and len(arr) > idx else float(arr[0])
                except Exception:
                    continue
        return None

    coef_conv = _get("est", "coef", idx=0)
    coef_robust = _get("est", "coef", idx=1)
    se_conv = _get("se", "se_conv", idx=0)
    z_robust = _get("z", "z_stat", idx=1)
    p_robust = _get("pv", "pval", idx=1)
    bw = getattr(r, "bws", None)
    bw_left = float(np.asarray(bw).ravel()[0]) if bw is not None else None
    bw_right = float(np.asarray(bw).ravel()[1]) if bw is not None and np.asarray(bw).size > 1 else bw_left
    ci = getattr(r, "ci", None)
    ci_arr = np.asarray(ci, dtype=float) if ci is not None else None
    ci_robust = (float(ci_arr[1, 0]), float(ci_arr[1, 1])) if ci_arr is not None and ci_arr.shape[0] > 1 else (None, None)

    n_eff = None
    for name in ("N", "N_h", "n"):
        v = getattr(r, name, None)
        if v is not None:
            arr = np.asarray(v).ravel()
            n_eff = int(arr.sum()) if arr.size else None
            break

    if coef_conv is None:
        return Finding(fid, "rdd", "fail", "rdrobust 没有返回可解析的系数，请检查版本", {})

    warnings, status = [], "ok"
    if p_robust is not None and p_robust >= 0.05:
        status = "warn"
        warnings.append(f"断点处效应不显著（robust p={p_robust:.3f}）："
                        "不要只报 conventional 的 p 值，稳健口径才是审稿人看的那个")
    front = d[(d[x] >= cut - 0.02) & (d[x] < cut)]
    back = d[(d[x] >= cut) & (d[x] <= cut + 0.02)]
    if len(front) < 20 or len(back) < 20:
        warnings.append(f"断点两侧 ±0.02 内样本偏少（左 {len(front)} / 右 {len(back)}）："
                        "带宽内的有效样本可能不足以支撑局部多项式")

    why = (f"RDD（cut={cut}，带宽 {bw_left:.3f}/{bw_right:.3f}）："
           f"conventional {coef_conv:+.4f}，robust {coef_robust:+.4f}"
           if coef_robust is not None else f"RDD（cut={cut}）：{coef_conv:+.4f}")
    if ci_robust[0] is not None:
        why += f"，robust 95% CI [{ci_robust[0]:+.4f}, {ci_robust[1]:+.4f}]"

    return Finding(fid, "rdd", status, why,
                   {"y": y, "x": x, "cut": cut, "coef": coef_conv,
                    "coef_robust": coef_robust, "se": se_conv,
                    "z_robust": z_robust, "p_robust": p_robust,
                    "bw_left": bw_left, "bw_right": bw_right,
                    "ci_robust": list(ci_robust), "n_eff": n_eff,
                    "n": len(d),
                    "note": "报效应请用 bias-corrected 估计与 robust CI（CCT 口径）"},
                   warnings=warnings)
