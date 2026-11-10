"""空间计量：带空间滞后/空间误差的回归（SAR / SEM）。

为什么用 spreg 而不是自己写：SAR 的 rho 是极大似然估计，要对 N×N 矩阵求行列式；
自己实现既慢又容易在稀疏矩阵与行标准化上出错。spreg（PySAL 家族）是这一块的标准实现。

输入要求：用户提供空间权重矩阵 csv（行列都是研究单元，如省份/城市）。
口径：默认行标准化（每个单元的邻居权重和为 1），这是 SAR 的常用约定；
不标准化时 rho 的量级不可比，结果里会写明。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def _load_w(path: str, row_standardize: bool = True):
    from libpysal.weights import W

    mat = pd.read_csv(path, index_col=0)
    idx = [str(i) for i in mat.index]
    cols = [str(c) for c in mat.columns]
    if set(idx) != set(cols):
        raise ValueError(f"权重矩阵的行列标签不一致：行 {idx[:3]}… 列 {cols[:3]}…")
    mat.index, mat.columns = idx, idx
    neigh = {i: {j: float(mat.loc[i, j]) for j in idx
                 if j != i and np.isfinite(mat.loc[i, j]) and mat.loc[i, j] != 0}
             for i in idx}
    w = W(neigh)
    if row_standardize:
        w.transform = "r"
    return w, idx


def spatial(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "spatial")
    try:
        from spreg import ML_Error, ML_Lag, OLS
    except ImportError:  # pragma: no cover
        return Finding(fid, "spatial", "fail",
                       "未安装 spreg / libpysal：uv pip install spreg libpysal", {})
    y, xs, unit = cfg["y"], cfg.get("x", []), cfg["unit"]
    if isinstance(xs, str):
        xs = [xs]
    w_path = cfg.get("w")
    model = (cfg.get("model") or "lag").lower()
    if not w_path:
        return Finding(fid, "spatial", "fail", "缺少 w（空间权重矩阵 csv 路径）", {})
    need = [y, unit] + list(xs)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "spatial", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    d[unit] = d[unit].astype(str)
    d = d.drop_duplicates(subset=unit)          # 每个单元一行（横截面）
    try:
        w, order = _load_w(w_path, bool(cfg.get("row_standardize", True)))
    except Exception as exc:
        return Finding(fid, "spatial", "fail", f"权重矩阵读取失败：{exc}", {})

    d = d.set_index(unit).reindex(order).dropna(subset=[y])
    if len(d) < 10:
        return Finding(fid, "spatial", "fail",
                       f"与权重矩阵对得上的单元只有 {len(d)} 个", {})
    sub_w = w
    if len(d) < len(order):
        present = list(d.index)
        sub_w = w.subset(present) if hasattr(w, "subset") else w
    yv = d[[y]].values
    X = d[list(xs)].values
    try:
        if model == "ols":
            m = OLS(yv, X, w=sub_w, spat_diag=True, name_y=y, name_x=list(xs))
        elif model == "error":
            m = ML_Error(yv, X, w=sub_w, name_y=y, name_x=list(xs))
        else:
            m = ML_Lag(yv, X, w=sub_w, name_y=y, name_x=list(xs))
    except Exception as exc:
        return Finding(fid, "spatial", "fail",
                       f"空间模型估计失败：{type(exc).__name__}: {exc}", {})

    betas = np.asarray(m.betas).ravel()
    names = ["const"] + list(xs)
    zs = np.asarray(getattr(m, "z_stat", [])).ravel() if hasattr(m, "z_stat") else None
    ps = np.asarray(getattr(m, "p_z", [])).ravel() if hasattr(m, "p_z") else None
    rows = []
    for i, nm in enumerate(names[:len(betas)]):
        rows.append({"var": nm, "coef": float(betas[i]),
                     "z": float(zs[i]) if zs is not None and i < len(zs) else None,
                     "p": float(ps[i]) if ps is not None and i < len(ps) else None})

    rho = getattr(m, "rho", None)
    lam = getattr(m, "lam", None)
    warnings, status = [], "ok"
    if model != "ols":
        key = "rho" if model == "lag" else "lambda"
        val = float(rho if model == "lag" else lam)
        if abs(val) > 0.9:
            status = "warn"
            warnings.append(f"{key}={val:.3f} 接近 1：空间依赖极强，模型可能不稳定")
    if cfg.get("row_standardize", True):
        warnings.append("权重矩阵已行标准化；若正文用的是未标准化权重，rho 不可直接比较")

    why = (f"空间{('滞后' if model == 'lag' else '误差' if model == 'error' else 'OLS')}"
           f"（N={len(d)}）："
           + (f"rho={float(rho):.4f}，" if model == "lag" and rho is not None else "")
           + (f"lambda={float(lam):.4f}，" if model == "error" and lam is not None else "")
           + "、".join(f"{r['var']} {r['coef']:+.4f}" for r in rows[1:]))
    return Finding(fid, "spatial", status, why,
                   {"y": y, "x": xs, "unit": unit, "model": model, "rows": rows,
                    "rho": float(rho) if rho is not None else None,
                    "lambda": float(lam) if lam is not None else None,
                    "n": len(d), "w": str(w_path),
                    "pseudo_r2": float(getattr(m, "pr2", float("nan")))
                    if getattr(m, "pr2", None) is not None else None},
                   warnings=warnings)
