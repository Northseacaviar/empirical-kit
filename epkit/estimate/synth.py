"""合成控制法：一个处理单位、多个候选对照，用对照的凸组合复制处理单位。

为什么自己实现 QP 而不调包：约束是「权重非负且和为 1」，这正是 scipy 的 SLSQP
能直接解的；第三方包通常还把画图与 placebo 绑死，反而不好嵌进流水线。

口径：
- 权重解 min ||X1 - X0 w||²，s.t. w ≥ 0, Σw = 1（Abadie & Gardeazabal 2003）；
- 拟合质量看处理后前的 RMSPE，以及处理前拟合路径；
- 推断用 placebo：把每个对照单位依次当"处理单位"，算其处理后 RMSPE 与处理前的比值，
  p = 排名（Abadie 的 rank-based p），并给出"处理单位是否最极端"。

边界：只做单一处理单位、单一处理时点；多处理单位请用 SDID 或其他方法。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..findings import Finding


def _weights(X1: np.ndarray, X0: np.ndarray) -> np.ndarray:
    k = X0.shape[1]

    def obj(w):
        r = X1 - X0 @ w
        return float(r @ r)

    cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
    bounds = [(0.0, 1.0)] * k
    x0 = np.full(k, 1.0 / k)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = res.x if res.success else x0
    return w / w.sum() if w.sum() else w


def synth(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "synth")
    y, unit, time = cfg["y"], cfg["unit"], cfg["time"]
    treated_unit = cfg["treated_unit"]
    treat_time = cfg["treat_time"]
    predictors = cfg.get("predictors") or []
    need = [y, unit, time] + list(predictors)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "synth", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    if treated_unit not in set(d[unit]):
        return Finding(fid, "synth", "fail", f"处理单位 {treated_unit!r} 不在 {unit} 里", {})
    times = sorted(d[time].unique())
    pre = [t for t in times if t < treat_time]
    post = [t for t in times if t >= treat_time]
    if len(pre) < 3 or not post:
        return Finding(fid, "synth", "fail",
                       f"处理前只有 {len(pre)} 期、处理后 {len(post)} 期：合成控制至少要 3 期前史", {})

    units = list(d[unit].unique())
    donors = [u for u in units if u != treated_unit]
    if len(donors) < 2:
        return Finding(fid, "synth", "fail", "对照单位少于 2 个", {})

    wide_pre = d[d[time].isin(pre)].pivot_table(index=time, columns=unit, values=y)
    wide_all = d.pivot_table(index=time, columns=unit, values=y).sort_index()
    if treated_unit not in wide_all.columns:
        return Finding(fid, "synth", "fail", "处理单位在结果变量上没有观测", {})
    donors = [u for u in donors if u in wide_pre.columns and wide_pre[u].notna().all()]
    if len(donors) < 2:
        return Finding(fid, "synth", "fail", "可用的对照单位不足（处理前有缺失）", {})

    # 预测变量：取处理前均值
    if predictors:
        X1 = d[(d[unit] == treated_unit) & (d[time].isin(pre))][predictors].mean().values
        X0 = np.vstack([d[(d[unit] == u) & (d[time].isin(pre))][predictors].mean().values
                        for u in donors]).T
    else:
        X1 = wide_pre[treated_unit].values
        X0 = wide_pre[donors].values

    w = _weights(np.asarray(X1, dtype=float), np.asarray(X0, dtype=float))
    weight_map = {u: float(wi) for u, wi in zip(donors, w)}

    # 合成路径
    synth_path = (wide_all[donors] * w).sum(axis=1)
    actual = wide_all[treated_unit]
    gap = actual - synth_path

    def rmspe(idx) -> float:
        v = gap.loc[[t for t in idx if t in gap.index]].dropna()
        return float(np.sqrt((v ** 2).mean())) if len(v) else float("nan")

    rmspe_pre, rmspe_post = rmspe(pre), rmspe(post)
    att_post = float(gap.loc[[t for t in post if t in gap.index]].mean())

    # placebo：每个对照单位依次当处理单位
    placebo = []
    for u in donors:
        others = [v for v in donors if v != u]
        if len(others) < 2:
            continue
        try:
            wu = _weights(np.asarray(wide_pre[u].values, dtype=float),
                          np.asarray(wide_pre[others].values, dtype=float))
            g = wide_all[u] - (wide_all[others] * wu).sum(axis=1)
            r_pre = float(np.sqrt((g.loc[pre] ** 2).mean()))
            r_post = float(np.sqrt((g.loc[[t for t in post if t in g.index]] ** 2).mean()))
            placebo.append({"unit": str(u), "rmspe_pre": r_pre, "rmspe_post": r_post,
                            "ratio": r_post / r_pre if r_pre else float("inf")})
        except Exception:
            continue
    own_ratio = rmspe_post / rmspe_pre if rmspe_pre else float("inf")
    better = sum(1 for p in placebo if p["ratio"] >= own_ratio)
    p_rank = (1 + better) / (1 + 1 + len(placebo))

    warnings, status = [], "ok"
    if not np.isfinite(p_rank) or p_rank > 0.1:
        status = "warn"
        warnings.append(f"placebo 排名 p={p_rank:.3f}：处理单位并不比多数对照单位更异常，"
                        "处理效应可能只是拟合误差")
    if rmspe_pre > 0.1 * abs(float(actual.loc[pre].mean())):
        warnings.append(f"处理前 RMSPE={rmspe_pre:.4f} 相对量级偏大：合成控制没有很好复制处理单位")
    if len(pre) < 5:
        warnings.append(f"处理前只有 {len(pre)} 期，拟合与 placebo 推断都不稳")

    top = sorted(weight_map.items(), key=lambda kv: -kv[1])[:5]
    why = (f"合成控制（{treated_unit}，{treat_time} 起）：处理后平均效应 {att_post:+.4f}，"
           f"处理前 RMSPE={rmspe_pre:.4f}，处理后 RMSPE={rmspe_post:.4f}，"
           f"placebo 排名 p={p_rank:.3f}")
    return Finding(fid, "synth", status, why,
                   {"y": y, "treated_unit": str(treated_unit), "treat_time": str(treat_time),
                    "att_post": att_post, "rmspe_pre": rmspe_pre, "rmspe_post": rmspe_post,
                    "p_rank": float(p_rank), "weights": weight_map,
                    "top_weights": [{"unit": k, "w": v} for k, v in top],
                    "donors": len(donors), "placebo": placebo,
                    "gap": [{"time": str(t), "gap": float(gap.loc[t])}
                            for t in gap.index if np.isfinite(gap.loc[t])]},
                   warnings=warnings)
