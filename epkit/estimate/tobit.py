"""Tobit：删失正态的最大似然。

为什么自己写：statsmodels 没有内置 Tobit，GitHub 上的实现要么只支持左截尾、
要么不带固定效应处理。这里按删失正态似然直接最大化，参数化 log(σ) 保证 σ>0。

边界（刻意不做）：
- 不做高维固定效应：个体固定效应的 Tobit 存在附带参数问题，估计有偏、
  数值也常不收敛。只接受少量离散固定效应（用哑变量），超过 50 个水平直接报错。
- 标准误是 iid MLE（Hessian 逆），没有聚类口径；面板数据用之前要清楚这一点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..findings import Finding


def _negll(params: np.ndarray, y: np.ndarray, X: np.ndarray,
           lower: float, upper: float) -> float:
    beta, log_sigma = params[:-1], params[-1]
    sigma = float(np.exp(log_sigma))
    xb = X @ beta
    ll = np.zeros(len(y))

    left = np.isfinite(lower) & (y <= lower + 1e-12)
    right = np.isfinite(upper) & (y >= upper - 1e-12)
    mid = ~(left | right)
    if mid.any():
        ll[mid] = stats.norm.logpdf((y[mid] - xb[mid]) / sigma) - log_sigma
    if left.any():
        ll[left] = stats.norm.logcdf((lower - xb[left]) / sigma)
    if right.any():
        ll[right] = stats.norm.logsf((upper - xb[right]) / sigma)
    return -float(ll.sum())


def tobit(df: pd.DataFrame, cfg: dict) -> Finding:
    fid = cfg.get("id", "tobit")
    y = cfg["y"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    controls = cfg.get("controls") or []
    if controls and not isinstance(controls[0], str):
        controls = controls[0]
    fe = cfg.get("fe") or []
    if fe and not isinstance(fe[0], str):
        fe = []
    tb = cfg.get("tobit") or {}
    lower = float(tb.get("lower", 0.0)) if tb.get("lower", 0.0) is not None else -np.inf
    upper = tb.get("upper")
    upper = float(upper) if upper is not None else np.inf

    need = [y] + list(xs) + list(controls) + list(fe)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "tobit", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    for f in fe:
        if d[f].nunique() > 50:
            return Finding(fid, "tobit", "fail",
                           f"固定效应 {f} 有 {d[f].nunique()} 个水平：Tobit 不做高维固定效应"
                           "（附带参数问题会使估计有偏）。改用行业/年份这类低维 FE，"
                           "或把个体异质性交给 BLP / 加控制变量", {})
    X_parts = [np.ones(len(d)), d[list(xs) + list(controls)].astype(float).values]
    for f in fe:
        dummies = pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float)
        X_parts.append(dummies.values)
        d = pd.concat([d.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = np.column_stack(X_parts)
    yy = d[y].astype(float).values

    if np.isfinite(lower) and (yy <= lower).sum() == 0:
        return Finding(fid, "tobit", "warn",
                       f"没有观测落在下界 {lower}：数据没有删失，用 OLS 即可（Tobit 会退化为 OLS）", {})
    n_left = int((yy <= lower).sum()) if np.isfinite(lower) else 0
    n_right = int((yy >= upper).sum()) if np.isfinite(upper) else 0

    # 初值：对未删失子样本做 OLS，σ 取残差标准差
    mid = ~(np.isfinite(lower) & (yy <= lower + 1e-12))
    if np.isfinite(upper):
        mid &= ~(yy >= upper - 1e-12)
    if mid.sum() < X.shape[1] + 2:
        return Finding(fid, "tobit", "fail", "未删失观测太少，无法给初值", {})
    beta0, *_ = np.linalg.lstsq(X[mid], yy[mid], rcond=None)
    resid = yy[mid] - X[mid] @ beta0
    start = np.append(beta0, np.log(max(resid.std(), 1e-3)))

    res = minimize(_negll, start, args=(yy, X, lower, upper), method="BFGS")
    if not res.success:
        res = minimize(_negll, res.x, args=(yy, X, lower, upper), method="Nelder-Mead",
                       options={"maxiter": 20000})
    params = res.x
    beta, sigma = params[:-1], float(np.exp(params[-1]))

    # Hessian → 标准误（数值二阶差分）
    p = len(params)
    H = np.zeros((p, p))
    h = 1e-5
    f0 = _negll(params, yy, X, lower, upper)
    for i in range(p):
        for j in range(p):
            ei = np.zeros(p); ei[i] = h
            ej = np.zeros(p); ej[j] = h
            H[i, j] = (_negll(params + ei + ej, yy, X, lower, upper)
                       - _negll(params + ei - ej, yy, X, lower, upper)
                       - _negll(params - ei + ej, yy, X, lower, upper)
                       + _negll(params - ei - ej, yy, X, lower, upper)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    names = ["const"] + list(xs) + list(controls) + [c for c in d.columns
                                                    if c.split("_")[0] in fe and c not in need]
    rows = []
    for i, nm in enumerate(names[:len(beta)]):
        b, s = float(beta[i]), float(se[i]) if i < len(se) else float("nan")
        z = b / s if s and np.isfinite(s) else float("nan")
        rows.append({"var": nm, "coef": b, "se": s, "z": float(z),
                     "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")})

    focus = cfg.get("focus") or (xs[0] if xs else names[0])
    fr = next((r for r in rows if r["var"] == focus), None)
    why = (f"Tobit（下界 {lower if np.isfinite(lower) else '无'}，"
           f"上界 {upper if np.isfinite(upper) else '无'}；删失 {n_left + n_right} 个）："
           f"{focus} 系数 {fr['coef']:+.4f}，z={fr['z']:.2f}，N={len(d)}"
           if fr else "Tobit 估计完成")
    warnings = ["Tobit 的标准误是 iid MLE；面板数据要聚类时请用 bootstrap 或换 LPM 做对照",
                "Tobit 系数不能直接读成边际效应（∂E[y|x]/∂x = β·Φ(xβ/σ)）"]
    return Finding(fid, "tobit", "ok", why,
                   {"y": y, "x": xs, "lower": lower, "upper": upper,
                    "rows": rows, "sigma": sigma, "n": len(d),
                    "n_left": n_left, "n_right": n_right,
                    "loglik": -f0, "focus": focus, "model": "tobit"},
                   warnings=warnings)
