"""阈值堆积：断点两侧的样本密度是否连续。

为什么要有：凡是"按某个阈值决定处理"的设计（分红率达标才强制披露、
市值低于线才纳入某指数），只要作者能影响自己在阈值哪一侧，样本就会在
阈值一侧堆积。McCrary (2008) 的密度检验就是查这个。

实现：阈值 cut±bw 内分箱计数，两侧各做一次线性拟合，比较 cut 处的密度，
θ = ln(f_right / f_left)，SE 用二项近似 sqrt(1/c_right + 1/c_left)。
这是 McCrary 检验的核心近似，不是他原始的两步局部线性过程；
对"有没有明显堆积"这个判断够用，写论文要报精确值时请用 rddensity。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding


def pileup(df: pd.DataFrame, cfg: dict) -> Finding:
    fid = cfg.get("id", "pileup")
    var = cfg["var"]
    if var not in df.columns:
        return Finding(fid, "pileup", "fail", f"列不存在：{var}", {})
    cut = float(cfg.get("cut", 0))
    bw = float(cfg.get("bw", 0.05))
    n_bins = int(cfg.get("bins", 20))

    s = pd.to_numeric(df[var], errors="coerce").dropna()
    near = s[(s >= cut - bw) & (s <= cut + bw)]
    if len(near) < 50:
        return Finding(fid, "pileup", "warn",
                       f"阈值 ±{bw} 内只有 {len(near)} 个观测，密度检验没有意义", {})

    edges = np.linspace(cut - bw, cut + bw, n_bins + 1)
    cnt, _ = np.histogram(near, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    left = centers < cut
    right = ~left

    def _fit_and_predict(mask: bool) -> tuple[float, float] | None:
        x = centers[mask]
        y = cnt[mask].astype(float)
        if len(x) < 2:
            return None
        # 线性拟合后在 cut 处的预测密度（每箱计数）
        A = np.column_stack([np.ones_like(x), x - cut])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = float(beta[0])
        resid = y - A @ beta
        dof = max(len(x) - 2, 1)
        s2 = float((resid ** 2).sum() / dof)
        cov = s2 * np.linalg.inv(A.T @ A)
        return pred, float(np.sqrt(max(cov[0, 0], 1e-12)))

    fl, sl = _fit_and_predict(left) or (0.0, 1.0)
    fr, sr = _fit_and_predict(right) or (0.0, 1.0)
    fl, fr = max(fl, 1e-9), max(fr, 1e-9)
    theta = float(np.log(fr / fl))
    se = float(np.sqrt((sl / fl) ** 2 + (sr / fr) ** 2))
    z = theta / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")

    warnings, status = [], "ok"
    if np.isfinite(p) and p < 0.05:
        status = "warn"
        side = "右侧" if theta > 0 else "左侧"
        warnings.append(f"阈值 {cut} 处密度不连续（θ={theta:+.3f}，z={z:.2f}，p={p:.4f}）："
                        f"{side}堆积，说明样本在阈值附近不是随机分布的")
        warnings.append("这是「个体能选择自己在阈值哪一侧」的典型迹象，需在正文交代制度与操纵可能性")

    why = (f"阈值 {cut}±{bw}：左侧密度 {fl:.1f}/箱，右侧 {fr:.1f}/箱，"
           f"θ={theta:+.3f}，z={z:.2f}，p={p:.3f}（N={len(near)}）")
    return Finding(fid, "pileup", status, why,
                   {"var": var, "cut": cut, "bw": bw, "n_near": int(len(near)),
                    "f_left": fl, "f_right": fr, "theta": theta, "z": float(z), "p": p,
                    "bins": [{"center": float(c), "count": int(m)}
                             for c, m in zip(centers, cnt)]},
                   warnings=warnings)
