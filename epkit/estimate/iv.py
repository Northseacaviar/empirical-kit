"""IV 诊断：第一阶段强度、过度识别、排他性交代。

为什么自己做：pyfixest 0.70 只回点估计，fitstat 里没有 ivf/sargan/wh；
statsmodels 的 IV 诊断不管高维固定效应与聚类。论文里这三项是必须交代的，
所以按主回归同一套固定效应与聚类口径自己算。

口径：
- 第一阶段 F 用聚类 Wald F（对工具变量系数联合为 0）。单内生变量时 F>10 是
  经验法则；多内生变量或多工具要用 Stock-Yogo / Effective F 临界值，这里只报警不裁决。
- 过度识别用 Sargan（n·R²，同方差假设）。聚类稳健的 Hansen J 需要 GMM，
  这里不给，写进 notes 让作者自己决定要不要补。
- 排他性无法用数据检验，只做提示。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def _first_stage(df: pd.DataFrame, endog: str, instruments: list[str],
                 exog: list[str], fe: list[str], clusters: list[str]) -> dict:
    """单个内生变量的第一阶段：工具变量联合 F。"""
    rhs = instruments + [e for e in exog if e != endog]
    fml = f"{endog} ~ 1 + " + " + ".join(rhs)
    if fe:
        fml += " | " + " + ".join(fe)
    vcov = ({"CRV1": clusters[0]} if len(clusters) == 1
            else {"CRV1": "+".join(clusters)}) if clusters else "hetero"
    m = pf.feols(fml, data=df, vcov=vcov)
    tidy = m.tidy()
    idx = list(tidy.index)
    R = np.zeros((len(instruments), len(idx)))
    for i, z in enumerate(instruments):
        if z not in idx:
            raise ValueError(f"工具变量 {z} 未进入第一阶段估计")
        R[i, idx.index(z)] = 1
    res = m.wald_test(R=R, distribution="chi2")
    # pyfixest 会把 distribution="F" 悄悄换回 chi2（R 非单位阵时），
    # 所以自己把 Wald 统计量折算成 F：F = W/df1，分母自由度取 聚类数-1（无聚类时取残差自由度）。
    from scipy import stats

    stat = float(res.stat)
    df1 = int(res.df1)
    g = int(min(m._G)) if getattr(m, "_G", None) else None
    df2 = (g - 1) if g else df_len(df, fe) - df1
    df2 = max(df2, 1)
    f_stat = stat / df1
    coefs = {z: {"coef": float(tidy.loc[z, "Estimate"]),
                 "t": float(tidy.loc[z, "t value"])} for z in instruments if z in idx}
    return {"endog": endog, "f": float(f_stat), "wald": stat,
            "df1": df1, "df2": df2,
            "p": float(stats.f.sf(f_stat, df1, df2)), "instruments": coefs,
            "n": int(m._N), "formula": fml}


def df_len(df: pd.DataFrame, fe: list[str]) -> int:
    """残差自由度近似：观测数 - 聚类数（无 fe 时减参数个数）。"""
    if fe:
        k = sum(df[f].nunique() for f in fe if f in df.columns)
    else:
        k = 1
    return max(int(len(df) - k), 1)


def _sargan(df: pd.DataFrame, uhat: np.ndarray, instruments: list[str],
            exog: list[str], fe: list[str], n_endog: int) -> dict | None:
    """Sargan：把 2SLS 残差对工具与外生变量回归，J = n·R²，χ²(q)。"""
    from scipy import stats

    q = len(instruments) - n_endog
    if q <= 0:
        return None
    d = df.copy()
    d["_uhat"] = uhat
    rhs = instruments + [e for e in exog if e not in instruments]
    fml = "_uhat ~ 1 + " + " + ".join(rhs)
    if fe:
        fml += " | " + " + ".join(fe)
    m = pf.feols(fml, data=d)
    r2 = m._r2_within if fe else m._r2
    if r2 is None or not np.isfinite(r2):
        r2 = m._r2
    n = int(m._N)
    j = float(n * r2)
    p = float(stats.chi2.sf(j, q))
    return {"stat": j, "df": q, "p": p, "n": n, "r2": float(r2)}


def iv_diagnostics(df: pd.DataFrame, cfg: dict) -> dict:
    """返回诊断字典；调用方负责并进 Finding。"""
    iv = cfg["iv"]
    endog = list(iv["endog"])
    instruments = list(iv["instruments"])
    y = cfg["y"]
    exog = [v for v in list(cfg.get("x", [])) if v not in endog]
    controls = cfg.get("controls") or []
    fe = cfg.get("fe", []) or []
    clusters = cfg.get("cluster", []) or []
    if fe and not isinstance(fe[0], str):
        fe = []
    if controls and not isinstance(controls[0], str):
        controls = []

    out: dict = {"first_stage": [], "notes": [], "endog": endog,
                 "instruments": instruments}
    exog_all = exog + controls
    fstats = []
    for e in endog:
        try:
            fs = _first_stage(df, e, instruments, exog_all, fe, clusters)
        except Exception as exc:
            out["notes"].append(f"{e} 的第一阶段估计失败：{type(exc).__name__}: {exc}")
            continue
        fs["weak"] = bool(fs["f"] < 10)
        out["first_stage"].append(fs)
        fstats.append(fs["f"])
    out["min_first_stage_f"] = float(min(fstats)) if fstats else None
    out["weak"] = bool(fstats and min(fstats) < 10)
    if len(endog) > 1 or len(instruments) > len(endog):
        out["notes"].append("多内生变量或多工具：弱工具判定应看 Stock-Yogo 临界值或 "
                            "Montiel Olea-Pflueger Effective F，单个第一阶段 F 只是参考")

    # 过度识别：用主回归的 2SLS 残差
    try:
        parts = list(exog) + list(controls) + [f"[{' + '.join(endog)} ~ {' + '.join(instruments)}]"]
        fml = f"{y} ~ 1 + " + " + ".join(parts)
        if fe:
            fml += " | " + " + ".join(fe)
        m = pf.feols(fml, data=df)
        out["sargan"] = _sargan(df, np.asarray(m._u_hat), instruments, exog_all, fe, len(endog))
        if out["sargan"] is None:
            out["notes"].append("工具数等于内生变量数：恰好识别，无法做过度识别检验")
        else:
            out["notes"].append("Sargan 用同方差假设，聚类稳健口径请自行补 Hansen J")
    except Exception as exc:
        out["sargan"] = None
        out["notes"].append(f"过度识别检验失败：{type(exc).__name__}: {exc}")

    out["notes"].append("排他性约束无法用数据检验，需在正文用制度事实论证")
    return out


def iv_warnings(diag: dict) -> tuple[list[str], bool]:
    """把诊断翻成 agent 能直接读的告警与状态。"""
    warns, weak_status = [], False
    for fs in diag.get("first_stage", []):
        if fs["weak"]:
            weak_status = True
            warns.append(f"{fs['endog']} 的第一阶段 F={fs['f']:.2f}（df1={fs['df1']}, "
                         f"df2={fs['df2']}）< 10：弱工具，需换工具或在正文交代弱识别下的推断")
    sar = diag.get("sargan")
    if sar and sar["p"] < 0.05:
        warns.append(f"过度识别检验拒绝（Sargan χ²={sar['stat']:.2f}, df={sar['df']}, "
                     f"p={sar['p']:.4f}）：工具可能不满足排他性，至少有一个是坏的")
        weak_status = True
    return warns, weak_status
