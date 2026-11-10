"""动态面板 GMM（Arellano-Bond 差分 GMM，两步估计）。

为什么自己写：pydynpd 依赖 numpy 1.x 的 API（np.in1d、标量转换），
在 numpy 2.5 上直接崩。差分 GMM 的核心只有三步——
一阶差分消个体效应、滞后水平做工具、两步权重更新——自己实现反而能把
口径（工具滞后范围、是否 collapse）写清楚，也能用已知参数验证。

边界（刻意不做）：
- 不做系统 GMM（Blundell-Bond）：水平方程那套工具（Δy 的滞后）需要额外假设，
  做半熟比不做更危险。要系统 GMM 请用 Stata 的 xtabond2 并在正文交代。
- 不做 Windmeijer 有限样本校正：两步 GMM 的 SE 在小 T 下偏小，写论文时要提一句。
- 要求个体内部的观测连续（内部自动取平衡部分），不平衡面板会丢样本。

诊断：Hansen J（过度识别）、AR(1)/AR(2)（差分残差的自相关）。
AR(2) 不显著才说明工具有效——这条比 Hansen 更容易被审稿人问。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding


def _balanced_wide(d: pd.DataFrame, entity: str, time: str, cols: list[str]):
    """转宽表并只保留个体内观测完整的区间。"""
    wide = {c: d.pivot_table(index=entity, columns=time, values=c) for c in cols}
    any_wide = next(iter(wide.values()))
    ok = any_wide.notna().all(axis=1)
    for c in wide:
        wide[c] = wide[c].loc[ok]
    return wide


def gmm(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "gmm")
    y = cfg["y"]
    entity, time = cfg["entity"], cfg["time"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    gmm_lags = cfg.get("gmm_lags") or [2, 3]
    lmin, lmax = int(gmm_lags[0]), int(gmm_lags[1])
    twostep = bool(cfg.get("twostep", True))

    need = [y, entity, time] + list(xs)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "gmm", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    wide = _balanced_wide(d, entity, time, [y] + list(xs))
    Y = wide[y]
    times = list(Y.columns)
    n_units = Y.shape[0]
    if len(times) < lmax + 3:
        return Finding(fid, "gmm", "fail",
                       f"每个个体只有 {len(times)} 期：差分 GMM 至少要 {lmax + 3} 期", {})

    dY = Y.diff(axis=1)
    dX = {c: wide[c].diff(axis=1) for c in xs}

    rows_y, rows_X, rows_Z, rows_reg = [], [], [], []
    start = max(lmax + 1, 2)
    for ti in range(start, len(times)):
        for ui in range(n_units):
            dy = dY.iloc[ui, ti]
            dylag = dY.iloc[ui, ti - 1]
            if not np.isfinite(dy) or not np.isfinite(dylag):
                continue
            dx = [dX[c].iloc[ui, ti] for c in xs]
            if any(not np.isfinite(v) for v in dx):
                continue
            z = [Y.iloc[ui, ti - l] for l in range(lmin, lmax + 1)]
            if any(not np.isfinite(v) for v in z):
                continue
            rows_y.append(dy)
            rows_X.append([dylag] + list(dx))
            rows_Z.append(z + list(dx))     # 差分外生变量作自己的工具
            rows_reg.append((Y.index[ui], times[ti]))

    if len(rows_y) < (len(xs) + 1) * 5:
        return Finding(fid, "gmm", "fail",
                       f"可用差分观测只有 {len(rows_y)} 个，GMM 无法识别", {})

    yy = np.asarray(rows_y, dtype=float)
    X = np.asarray(rows_X, dtype=float)
    Z = np.asarray(rows_Z, dtype=float)
    n, k = X.shape
    m = Z.shape[1]
    if m < k:
        return Finding(fid, "gmm", "fail",
                       f"工具数 {m} 少于参数 {k}：检查 gmm_lags 与解释变量", {})

    def gmm_step(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ZX = Z.T @ X
        Zy = Z.T @ yy
        A = ZX.T @ W @ ZX
        b = ZX.T @ W @ Zy
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(A) @ b
        resid = yy - X @ coef
        return coef, resid

    W1 = np.linalg.pinv(Z.T @ Z)
    coef1, resid1 = gmm_step(W1)
    coef, resid = coef1, resid1
    if twostep:
        # 聚类到个体的 meat 矩阵
        omega = np.zeros((m, m))
        for u in np.unique([r[0] for r in rows_reg]):
            mask = np.array([r[0] == u for r in rows_reg])
            zi = Z[mask]
            ei = resid1[mask]
            g = zi.T @ ei
            omega += np.outer(g, g)
        W2 = np.linalg.pinv(omega)
        coef, resid = gmm_step(W2)
    else:
        W2 = W1

    # 系数协方差：A^-1 ZX' W Ω W ZX A^-1（两步时用第二次的 Ω）
    omega = np.zeros((m, m))
    for u in np.unique([r[0] for r in rows_reg]):
        mask = np.array([r[0] == u for r in rows_reg])
        zi = Z[mask]
        ei = resid[mask]
        g = zi.T @ ei
        omega += np.outer(g, g)
    ZX = Z.T @ X
    A = ZX.T @ W2 @ ZX
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A)
    cov = Ainv @ (ZX.T @ W2 @ omega @ W2 @ ZX) @ Ainv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    names = [f"L1.{y}"] + list(xs)
    rows = []
    for i, nm in enumerate(names):
        zstat = coef[i] / se[i] if se[i] else float("nan")
        rows.append({"var": nm, "coef": float(coef[i]), "se": float(se[i]),
                     "z": float(zstat),
                     "p": float(2 * stats.norm.sf(abs(zstat))) if np.isfinite(zstat) else float("nan")})

    # Hansen J
    g = Z.T @ resid
    try:
        J = float(g @ np.linalg.pinv(omega) @ g)
        df_j = max(m - k, 1)
        p_j = float(stats.chi2.sf(J, df_j))
    except Exception:
        J, df_j, p_j = float("nan"), 0, float("nan")

    # AR(1)/AR(2)：差分残差按个体求滞后相关
    resid_by_unit: dict = {}
    for (u, t), e in zip(rows_reg, resid):
        resid_by_unit.setdefault(u, []).append(float(e))

    def _ar(order: int) -> float:
        """个体内自相关的近似检验。

        口径：每个个体算 rho_i = Σ e_t e_{t-k} / Σ e_t²，再对 rho_i 的均值做 t 检验。
        这是近似实现——标准做法（xtabond2、abond）用 GMM 框架下的渐近方差。
        两种更差的做法都试过并放弃：pooled 相关乘 sqrt(总观测数) 把 SE 按 N 压缩、
        直接用个体相关系数的 t 检验对微小偏离过度敏感（AR(2) 在正确 DGP 上也报 p<0.001）。
        """
        rs = []
        for seq in resid_by_unit.values():
            if len(seq) > order + 1:
                a = np.asarray(seq[:-order])
                b = np.asarray(seq[order:])
                den = float((a * a).sum())
                if den > 0:
                    rs.append(float((a * b).sum() / den))
        if len(rs) < 5:
            return float("nan")
        rs = np.asarray(rs)
        se = rs.std(ddof=1) / np.sqrt(len(rs))
        if se == 0:
            return float("nan")
        t = rs.mean() / se
        return float(2 * stats.t.sf(abs(t), len(rs) - 1))

    p_ar1, p_ar2 = _ar(1), _ar(2)

    warnings, status = [], "ok"
    if np.isfinite(p_ar2) and p_ar2 < 0.05:
        status = "warn"
        warnings.append(f"AR(2) p={p_ar2:.3f} < 0.05：差分残差存在二阶自相关，"
                        "滞后水平不是有效工具（这是比 Hansen 更致命的诊断）")
    if np.isfinite(p_j) and p_j < 0.05:
        status = "warn"
        warnings.append(f"Hansen J p={p_j:.3f}：过度识别检验被拒绝，工具有效性可疑")
    if not np.isfinite(p_ar1) or p_ar1 > 0.05:
        warnings.append("AR(1) 不显著：差分模型本应有一阶负自相关，检查差分是否正确")
    warnings.append("两步 GMM 的标准误未做 Windmeijer 有限样本校正，小 T 时偏小")
    warnings.append("AR(1)/AR(2) 是按个体比率的近似检验，不是 xtabond2 的渐近口径；"
                    "要进论文请用 Stata 复核")
    if not twostep:
        warnings.append("用的是一步 GMM：在异方差下不如两步稳健")

    focus = cfg.get("focus") or names[1] if len(names) > 1 else names[0]
    fr = next((r for r in rows if r["var"] == focus), rows[0])
    why = (f"差分 GMM（{'两步' if twostep else '一步'}，工具滞后 {lmin}-{lmax}，"
           f"n={n}，个体={n_units}）：{fr['var']} 系数 {fr['coef']:+.4f}"
           f"（z={fr['z']:.2f}），Hansen J p={p_j:.3f}，AR(2) p={p_ar2:.3f}")
    return Finding(fid, "gmm", status, why,
                   {"y": y, "x": xs, "rows": rows, "name": "difference GMM (Arellano-Bond)",
                    "twostep": twostep, "gmm_lags": [lmin, lmax],
                    "n_obs": n, "n_units": n_units, "n_instruments": m,
                    "hansen": {"stat": J, "df": df_j, "p": p_j},
                    "ar1_p": p_ar1, "ar2_p": p_ar2, "focus": focus},
                   warnings=warnings)
