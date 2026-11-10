"""Heckman 两步：当因变量只在特定子样本上被观测时，纠正选择偏差。

为什么自己实现：py4etrics 的 heckit 依赖 `statsmodels.tools.decorators`，
在 statsmodels 0.15 里已经不存在（装了也 import 失败）。两步法本身只有三十行，
自己写还能接聚类标准误。

口径（Heckman 1979）：
1. 选择方程：probit(观测与否 ~ 选择变量)，得逆米尔斯比 λ = φ(zγ)/Φ(zγ)；
2. 结果方程：在观测子样本上 OLS(y ~ X + λ)，λ 的系数即选择偏差项（rho·sigma）。
标准误是两步法的常规 OLS 标准误，**未校正第一步估计的不确定性**；
严进口径要 bootstrap 或 MLE，写论文时要交代这一点。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding


def heckman(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "heckman")
    y = cfg["y"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    sel = cfg.get("select") or []
    if isinstance(sel, str):
        sel = [sel]
    observed = cfg.get("observed")          # 0/1：因变量是否被观测
    clusters = cfg.get("cluster") or []

    if not observed:
        return Finding(fid, "heckman", "fail",
                       "缺少 observed（0/1）：Heckman 需要知道哪些观测的选择变量可见但结果缺失", {})
    need = [y, observed] + list(xs) + list(sel) + list(clusters)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "heckman", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=[observed] + list(sel) + list(clusters)).copy()
    d[observed] = d[observed].astype(int)
    if d[observed].nunique() != 2:
        return Finding(fid, "heckman", "fail", f"{observed} 必须有两个取值", {})

    import statsmodels.api as sm

    # 第一步：probit
    Xs = sm.add_constant(d[sel].astype(float))
    try:
        prob = sm.Probit(d[observed], Xs).fit(disp=0)
    except Exception as exc:
        return Finding(fid, "heckman", "fail", f"选择方程估计失败：{exc}", {})
    # statsmodels 0.15 的 BinaryModel.predict 不再接受 linear=True：
    # 取概率后用 Φ⁻¹ 反推线性预测，再算逆米尔斯比
    p_hat = np.clip(np.asarray(prob.predict(Xs)), 1e-12, 1 - 1e-12)
    xb = stats.norm.ppf(p_hat)
    imr = stats.norm.pdf(xb) / np.clip(stats.norm.cdf(xb), 1e-12, None)

    # 第二步：观测子样本上的 OLS，加 λ
    obs = d[observed] == 1
    sub = d[obs].copy()
    sub["_imr"] = imr[obs]
    Xo = sm.add_constant(sub[list(xs) + ["_imr"]].astype(float))
    try:
        ols = sm.OLS(sub[y].astype(float), Xo).fit(
            cov_type="cluster", cov_kwds={"groups": sub[clusters[0]]}) if clusters \
            else sm.OLS(sub[y].astype(float), Xo).fit()
    except Exception as exc:
        return Finding(fid, "heckman", "fail", f"结果方程估计失败：{exc}", {})

    rows = []
    for v in [*xs, "_imr"]:
        if v in ols.params.index:
            rows.append({"var": "lambda（逆米尔斯比）" if v == "_imr" else v,
                         "coef": float(ols.params[v]), "se": float(ols.bse[v]),
                         "t": float(ols.tvalues[v]), "p": float(ols.pvalues[v])})

    lam = next((r for r in rows if r["var"].startswith("lambda")), None)
    warnings, status = [], "ok"
    if lam is None:
        return Finding(fid, "heckman", "fail", "结果方程里没有 λ，检查设定", {"rows": rows})
    if lam["p"] < 0.05:
        warnings.append(f"λ 显著（p={lam['p']:.3f}）：确实存在选择偏差，OLS 的系数不可直接使用")
    else:
        warnings.append(f"λ 不显著（p={lam['p']:.3f}）：没有证据表明存在选择偏差，"
                        "可以直接报 OLS 结果并注明已检验")
    warnings.append("标准误未校正第一步（probit）的估计不确定性；严进口径请用 bootstrap 或 MLE")

    focus = cfg.get("focus") or (xs[0] if xs else None)
    fr = next((r for r in rows if r["var"] == focus), None)
    why = (f"Heckman 两步（观测 {int(obs.sum())}/{len(d)}）："
           + (f"{focus} 系数 {fr['coef']:+.4f}（t={fr['t']:.2f}），" if fr else "")
           + f"λ={lam['coef']:+.4f}（p={lam['p']:.3f}）")
    return Finding(fid, "heckman", status, why,
                   {"y": y, "x": xs, "select": sel, "observed": observed,
                    "rows": rows, "lambda": lam, "n_observed": int(obs.sum()),
                    "n_total": len(d), "focus": focus,
                    "vcov": "cluster" if clusters else "ols"},
                   warnings=warnings)
