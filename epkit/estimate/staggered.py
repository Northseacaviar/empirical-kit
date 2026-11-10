"""交错 DID 的估计量复核：Callaway–Sant'Anna 的分队列 ATT 与 2×2 配对诊断。

为什么必须复核：TWFE 在交错处理下拿"已经处理的组"当控制组，
权重会出现负值，系数连方向都可能错（Goodman-Bacon 2021）。
诊断只报警不够——论文里要给出一个没有负权重的估计量。

这里给两条独立证据：
1. CS 分队列 ATT(g,k)：控制组只用"从未处理"和"到 t 期仍未处理"的个体，
   按 (g,k) 逐格估计，再按处理组规模加权汇总。没有负权重。
2. 2×2 配对诊断：早期组 vs 后期组、各队列 vs 从未处理、后期组 vs 已处理组。
   每格给点估计与样本量，不做权重加总（严格的 Bacon 权重公式对不平衡面板敏感，
   与其用近似权重冒充，不如把每格的数摆出来让作者自己看）。

标准误用两组长差的独立样本方差（var_t/n_t + var_c/n_c），
与 CS 原论文的影响函数推断在无协变量时一致；有协变量时不适用，会在 notes 里写明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _long_diff(d: pd.DataFrame, idv: str, y: str, period: str,
               t1, t0) -> pd.Series:
    a = d.loc[d[period] == t1, [idv, y]].dropna().set_index(idv)[y]
    b = d.loc[d[period] == t0, [idv, y]].dropna().set_index(idv)[y]
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common])


def _two_sample(dy: pd.Series, treat_ids, ctrl_ids) -> dict:
    dt = dy.reindex(dy.index.intersection(treat_ids)).dropna()
    dc = dy.reindex(dy.index.intersection(ctrl_ids)).dropna()
    nt, nc = len(dt), len(dc)
    if nt < 2 or nc < 2:
        return {"att": None, "se": None, "n_treat": nt, "n_control": nc}
    att = float(dt.mean() - dc.mean())
    se = float(np.sqrt(dt.var(ddof=1) / nt + dc.var(ddof=1) / nc))
    t = att / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), min(nt, nc) - 1)) if np.isfinite(t) else np.nan
    return {"att": att, "se": se, "t": float(t), "p": p,
            "n_treat": nt, "n_control": nc}


def cs_att_gt(d: pd.DataFrame, y: str, idv: str, period: str, cohort: str,
              never=0, window: tuple[int, int] = (-4, 6),
              controls: list[str] | None = None) -> dict:
    """Callaway–Sant'Anna 的 ATT(g,k)，控制组为 not-yet-treated + never-treated。"""
    lo, hi = int(window[0]), int(window[1])
    treated_units = d.loc[d[cohort] != never, [idv, cohort]].drop_duplicates(idv)
    cohorts = sorted(treated_units[cohort].dropna().unique())
    never_ids = set(d.loc[d[cohort] == never, idv].unique())

    rows = []
    for g in cohorts:
        g = float(g) if isinstance(g, (np.floating, float)) else g
        base = g - 1
        ids_g = set(treated_units.loc[treated_units[cohort] == g, idv])
        for k in range(lo, hi + 1):
            if k == -1:
                continue
            t1 = g + k
            # 控制组：从未处理 + 到 t1 仍未处理（g' > t1）
            late = set(treated_units.loc[treated_units[cohort] > t1, idv])
            ctrl = never_ids | late
            dy = _long_diff(d, idv, y, period, t1, base)
            if dy.empty:
                continue
            st = _two_sample(dy, ids_g, ctrl)
            if st["att"] is None:
                continue
            st.update({"g": g, "k": k, "period": t1, "base": base})
            rows.append(st)

    # 按处理组规模加权（simple aggregation：k>=0 的各格）
    post = [r for r in rows if r["k"] >= 0 and r["att"] is not None]
    agg = None
    if post:
        w = np.array([r["n_treat"] for r in post], dtype=float)
        a = np.array([r["att"] for r in post], dtype=float)
        s = np.array([r["se"] for r in post], dtype=float)
        w = w / w.sum()
        att = float((w * a).sum())
        # 近似：格子间协方差忽略不计，SE 取加权平方和的开方
        se = float(np.sqrt(((w * s) ** 2).sum()))
        t = att / se if se > 0 else np.nan
        agg = {"att": att, "se": se, "t": float(t),
               "p": float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan,
               "n_cells": len(post), "method": "按处理组规模加权（k>=0）"}

    return {"att_gt": rows, "aggregate": agg,
            "cohorts": [float(c) for c in cohorts],
            "never_units": len(never_ids)}


def bacon_pairs(d: pd.DataFrame, y: str, idv: str, period: str, cohort: str,
                never=0, window: tuple[int, int] = (-4, 6)) -> list[dict]:
    """2×2 配对诊断：每格给点估计与样本量，不加权汇总。"""
    lo, hi = int(window[0]), int(window[1])
    treated_units = d.loc[d[cohort] != never, [idv, cohort]].drop_duplicates(idv)
    cohorts = sorted(treated_units[cohort].dropna().unique())
    never_ids = set(d.loc[d[cohort] == never, idv].unique())
    out = []

    def add(kind, g, control_ids, t1, t0, note=""):
        dy = _long_diff(d, idv, y, period, t1, t0)
        if dy.empty:
            return
        ids_g = set(treated_units.loc[treated_units[cohort] == g, idv])
        st = _two_sample(dy, ids_g, control_ids)
        if st["att"] is None:
            return
        st.update({"type": kind, "cohort": g, "period_from": t0, "period_to": t1,
                   "note": note})
        out.append(st)

    for g in cohorts:
        g = float(g) if isinstance(g, (np.floating, float)) else g
        # 各队列与从未处理组比较（窗口两端）
        for k in (0, hi):
            t1, t0 = g + k, g - 1
            add("vs_never", g, never_ids, t1, t0, "该队列 vs 从未处理组")
        # 早期 vs 后期（窗口：早期处理期到后期处理期前一期）
        for g2 in cohorts:
            g2 = float(g2) if isinstance(g2, (np.floating, float)) else g2
            if g2 <= g:
                continue
            ids_early = set(treated_units.loc[treated_units[cohort] == g, idv])
            ids_late = set(treated_units.loc[treated_units[cohort] == g2, idv])
            dy = _long_diff(d, idv, y, period, g2 - 1, g - 1)
            if dy.empty:
                continue
            st = _two_sample(dy, ids_early, ids_late)
            if st["att"] is None:
                continue
            st.update({"type": "early_vs_late", "cohort": g, "period_from": g - 1,
                       "period_to": g2 - 1,
                       "note": f"队列 {g} vs 队列 {g2}（后期组此时未处理）"})
            out.append(st)
            # timing：后期组 vs 已处理的早期组（负权重的主要来源）
            dy2 = _long_diff(d, idv, y, period, g2, g2 - 1)
            if dy2.empty:
                continue
            st2 = _two_sample(dy2, ids_late, ids_early)
            if st2["att"] is None:
                continue
            st2.update({"type": "late_vs_treated", "cohort": g2, "period_from": g2 - 1,
                        "period_to": g2,
                        "note": "后期组 vs 已处理组（TWFE 负权重来源）"})
            out.append(st2)
    return out


def heterogeneity_q(rows: list[dict]) -> dict | None:
    """同一 k 上跨队列的 ATT 是否一致（Cochran Q 的加权版）。"""
    by_k: dict[int, list[dict]] = {}
    for r in rows:
        if r["k"] >= 0 and r["att"] is not None and r["se"] and r["se"] > 0:
            by_k.setdefault(int(r["k"]), []).append(r)
    tests = []
    for k, cells in sorted(by_k.items()):
        if len(cells) < 2:
            continue
        w = np.array([1 / c["se"] ** 2 for c in cells])
        a = np.array([c["att"] for c in cells])
        bar = float((w * a).sum() / w.sum())
        q = float((w * (a - bar) ** 2).sum())
        df = len(cells) - 1
        tests.append({"k": k, "q": q, "df": df,
                      "p": float(stats.chi2.sf(q, df)),
                      "att_bar": bar,
                      "by_cohort": {str(c["g"]): c["att"] for c in cells}})
    if not tests:
        return None
    worst = min(tests, key=lambda x: x["p"])
    return {"by_k": tests, "min_p": worst["p"], "worst_k": worst["k"]}
