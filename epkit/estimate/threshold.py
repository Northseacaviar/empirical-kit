"""面板门槛模型（Hansen 1999）：斜率在某个人为的门槛值两侧不同。

为什么自己写：没有现成的 Python 包（R 的 threshold 包、Stata 的 xthreg 都是别的语言）。
核心只有两步——网格搜索门槛值使残差平方和最小、bootstrap 检验门槛是否显著。

口径：
- 门槛值在 trim 后（默认去掉两端 10%）的样本里网格搜索；
- 固定效应用组内去均值吸收；
- 显著性用 Hansen 的自助法：从线性模型残差重抽，重算 sup-F，得到经验 p。
  这个 p 值回答的是「是否存在门槛」，不是「门槛值等于某个数」。

边界：只做单门槛；双门槛（先定第一个再搜第二个）容易过拟合，不做。
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


def _ssr(df: pd.DataFrame, y: str, rhs: list[str], fe: list[str]) -> tuple[float, dict]:
    fml = f"{y} ~ 1 + " + " + ".join(rhs)
    if fe:
        fml += " | " + " + ".join(fe)
    m = pf.feols(fml, data=df)
    ssr = float(np.sum(np.asarray(m._u_hat) ** 2))
    td = m.tidy()
    return ssr, {v: {"coef": float(td.loc[v, "Estimate"]),
                     "se": float(td.loc[v, "Std. Error"]),
                     "p": float(td.loc[v, "Pr(>|t|)"])} for v in rhs if v in td.index}


def threshold(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "threshold")
    if pf is None:
        return Finding(fid, "threshold", "fail", "未安装 pyfixest", {})
    y = cfg["y"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    controls = cfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = cfg.get("fe") or []
    thr_var = cfg["thr_var"]
    trim = float(cfg.get("trim", 0.1))
    grid_n = int(cfg.get("grid", 60))
    boot = int(cfg.get("boot", 199))
    seed = int(cfg.get("seed", 42))

    need = [y, thr_var] + list(xs) + list(controls) + list(fe)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "threshold", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    if d[thr_var].nunique() < 10:
        return Finding(fid, "threshold", "fail", f"门槛变量 {thr_var} 的取值太少，无法搜索门槛", {})
    lo, hi = d[thr_var].quantile(trim), d[thr_var].quantile(1 - trim)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return Finding(fid, "threshold", "fail", f"trim={trim} 后门槛变量没有可用区间", {})
    grid = np.unique(np.linspace(lo, hi, grid_n))

    base_rhs = list(xs) + list(controls)
    try:
        ssr0, base_coef = _ssr(d, y, base_rhs, fe)
    except Exception as exc:
        return Finding(fid, "threshold", "fail", f"线性模型估计失败：{exc}", {})

    focus = cfg.get("focus") or xs[0]
    best = None
    for c in grid:
        dd = d.copy()
        low = (dd[thr_var] <= c).astype(float)
        dd["_lo"] = dd[focus] * low
        dd["_hi"] = dd[focus] * (1 - low)
        rhs = [x for x in xs if x != focus] + ["_lo", "_hi"] + list(controls)
        try:
            ssr, coef = _ssr(dd, y, rhs, fe)
        except Exception:
            continue
        if best is None or ssr < best["ssr"]:
            best = {"ssr": ssr, "cut": float(c), "coef": coef, "rhs": rhs}
    if best is None:
        return Finding(fid, "threshold", "fail", "所有网格点都估计失败", {})

    n = len(d)
    k = len(best["rhs"]) + 1
    sigma2 = best["ssr"] / max(n - k, 1)
    f_stat = (ssr0 - best["ssr"]) / (sigma2 if sigma2 > 0 else float("nan"))

    # Hansen 自助法：从线性模型残差重抽
    rng = np.random.default_rng(seed)
    dd0 = d.copy()
    f_lin = f"{y} ~ 1 + " + " + ".join(base_rhs) + (f" | {' + '.join(fe)}" if fe else "")
    m0 = pf.feols(f_lin, data=dd0)
    resid0 = np.asarray(m0._u_hat) if hasattr(m0, "_u_hat") else None
    # Hansen 自助法：用残差重抽构造 y*（加回拟合值），重算 sup-F
    f_boot = []
    if resid0 is not None and len(resid0) == n:
        y_hat = d[y].values - resid0
        for _ in range(boot):
            ys = y_hat + rng.choice(resid0, size=n, replace=True)
            db = d.copy()
            db["_ystar"] = ys
            try:
                ssr0b, _ = _ssr(db, "_ystar", base_rhs, fe)
            except Exception:
                continue
            best_b = None
            for c in grid[::max(len(grid) // 20, 1)]:
                dbs = db.copy()
                low = (dbs[thr_var] <= c).astype(float)
                dbs["_lo"] = dbs[focus] * low
                dbs["_hi"] = dbs[focus] * (1 - low)
                rhs = [x for x in xs if x != focus] + ["_lo", "_hi"] + list(controls)
                try:
                    ssrb, _ = _ssr(dbs, "_ystar", rhs, fe)
                except Exception:
                    continue
                if best_b is None or ssrb < best_b:
                    best_b = ssrb
            if best_b is not None:
                s2 = best_b / max(n - k, 1)
                if s2 > 0:
                    f_boot.append((ssr0b - best_b) / s2)
    p_boot = float(np.mean(np.array(f_boot) >= f_stat)) if f_boot else float("nan")

    lo_c = best["coef"].get("_lo", {})
    hi_c = best["coef"].get("_hi", {})
    flip = (np.sign(lo_c.get("coef", 0)) != np.sign(hi_c.get("coef", 0)))
    warnings, status = [], "ok"
    if np.isfinite(p_boot) and p_boot > 0.1:
        status = "warn"
        warnings.append(f"门槛显著性 bootstrap p={p_boot:.3f}：没有证据表明存在门槛，"
                        "两侧系数差异可能只是分组造成的噪声")
    if flip:
        warnings.append("门槛两侧系数符号相反：这类结果要格外小心，"
                        "先确认不是门槛值把样本切成两个不同总体")
    if len(grid) < 20:
        warnings.append(f"网格只有 {len(grid)} 点，门槛值精度有限")

    why = (f"面板门槛（{thr_var} ≤ {best['cut']:.4f}）：低侧系数 {lo_c.get('coef', float('nan')):+.4f}，"
           f"高侧 {hi_c.get('coef', float('nan')):+.4f}，sup-F={f_stat:.2f}，"
           f"bootstrap p={p_boot:.3f}（{len(f_boot)} 次）")
    return Finding(fid, "threshold", status, why,
                   {"y": y, "x": xs, "thr_var": thr_var, "cut": best["cut"],
                    "coef_low": lo_c, "coef_high": hi_c,
                    "ssr_linear": ssr0, "ssr_threshold": best["ssr"],
                    "f_stat": float(f_stat), "p_boot": p_boot,
                    "n_obs": n, "n_boot_ok": len(f_boot),
                    "grid_size": int(len(grid)), "trim": trim,
                    "base_coef": base_coef},
                   warnings=warnings)
