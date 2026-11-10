"""DID：事件研究式动态效应 + 预趋势检验。

为什么自己做而不调 pyfixest 的 event_study：后者只回一个汇总 ATT，
取不到逐期系数；而论文要的是 T-4…T+6 那条系数线加上预趋势联合检验。

模型（单期与交错共用）：
    y_it = α_i + γ_t + Σ_{k≠-1} β_k · 1[t - g_i = k] + X'δ + ε_it
g_i 是首次处理期（从未处理填 never，默认 0）。
β_k 即事件时间 k 的处理效应，基准期 k=-1。

交错处理的警告写进 warnings：TWFE 动态效应在多个处理时点时存在负权重，
需要 CS / did2s / Sun-Abraham 复核。工具不替你选估计量，但一定提醒。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..findings import Finding

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def _event_time(df: pd.DataFrame, period: str, cohort: str, never) -> pd.Series:
    g = df[cohort]
    t = df[period]
    et = t - g
    et = et.where(~(g == never) & g.notna())
    return et


def did(df: pd.DataFrame, cfg: dict) -> Finding:
    if pf is None:
        return Finding(cfg.get("id", "did"), "did", "fail", "未安装 pyfixest", {})
    fid = cfg.get("id", "did")
    y, period, cohort, idv = cfg["y"], cfg["period"], cfg["cohort"], cfg["idvar"]
    never = cfg.get("never", 0)
    fe = cfg.get("fe") or [idv, period]
    clusters = cfg.get("cluster") or [idv]
    controls = cfg.get("controls") or []
    lo, hi = cfg.get("window", [-4, 6])
    binned = True

    need = [y, period, cohort, idv] + fe + clusters + controls
    missing = [c for c in need if c not in df.columns]
    if missing:
        near = [c for c in df.columns if any(m.split("_")[0] in c for m in missing)][:8]
        return Finding(fid, "did", "fail", f"列不存在：{missing}；可能是 {near}", {})

    d = df.copy()
    d["_et"] = _event_time(d, period, cohort, never)
    d = d[d["_et"].notna() | (d[cohort] == never)].copy()
    if len(d) == 0:
        return Finding(fid, "did", "fail", "构造事件时间后没有可用观测", {})

    ks = [k for k in range(int(lo), int(hi) + 1) if k != -1]
    # 端点归并：窗口外的处理期必须并入 lo / hi，否则它们没有 dummy，
    # 会通过固定效应把基准期 k=-1 拉偏，制造出假的预趋势。
    d["_etc"] = d["_et"].clip(int(lo), int(hi))
    names = []
    for k in ks:
        col = f"_ev_m{abs(k)}" if k < 0 else f"_ev_{k}"
        d[col] = (d["_etc"] == k).astype(int)
        names.append(col)

    present = [n for n in names if d[n].sum() > 0]
    if not present:
        return Finding(fid, "did", "fail", f"窗口 [{lo},{hi}] 内没有处理期观测", {})

    fml = f"{y} ~ {' + '.join(present + controls)} | {' + '.join(fe)}"
    try:
        m = pf.feols(fml, data=d, vcov={"CRV1": clusters[0]} if len(clusters) == 1
                     else {"CRV1": "+".join(clusters)})
    except Exception as exc:
        return Finding(fid, "did", "fail", f"估计失败：{type(exc).__name__}: {exc}",
                       {"formula": fml})

    tidy = m.tidy()
    rows = []
    # 按 k 重新对齐（present 可能少于 ks）
    label_to_k = {}
    for k in ks:
        label_to_k[f"_ev_m{abs(k)}" if k < 0 else f"_ev_{k}"] = k
    for n in present:
        if n not in tidy.index:
            continue
        k = label_to_k[n]
        rows.append({"k": k, "coef": float(tidy.loc[n, "Estimate"]),
                     "se": float(tidy.loc[n, "Std. Error"]),
                     "t": float(tidy.loc[n, "t value"]),
                     "p": float(tidy.loc[n, "Pr(>|t|)"])})
    rows.sort(key=lambda r: r["k"])
    if not rows:
        return Finding(fid, "did", "fail", "没有提取到任何事件时间系数", {"formula": fml})

    # 预趋势联合检验：k<0 的系数是否同时为 0
    pre = [n for n in present if label_to_k[n] < 0 and n in tidy.index]
    wt = None
    if len(pre) >= 1:
        try:
            idx_all = list(tidy.index)
            R = np.zeros((len(pre), len(idx_all)))
            for i, n in enumerate(pre):
                R[i, idx_all.index(n)] = 1
            res = m.wald_test(R=R, distribution="chi2")
            stat, df1 = float(res.stat), int(res.df1)
            # 聚类稳健 V 按 χ² 查表会低估小样本不确定性（24 聚类实测 10% vs 7%），
            # 与 iv.py 的第一阶段 F 用同一口径：F = W/df1，df2 = G-1
            g = int(min(m._G)) if getattr(m, "_G", None) else None
            if g:
                from scipy import stats as _st

                df2 = max(g - 1, 1)
                p_f = float(_st.f.sf(stat / df1, df1, df2))
            else:
                df2, p_f = None, float(res.pvalue)
            wt = {"stat": stat, "p": p_f, "df": df1, "df2": df2,
                  "distribution": "F" if g else "chi2"}
        except Exception as exc:
            wt = None

    post = [r for r in rows if r["k"] >= 0]
    # 全 pre 窗口时没有 post 格，att 置 None 而不是 nan（nan 会进 why 与「同向」判断）
    att = float(np.mean([r["coef"] for r in post])) if post else None

    cohorts = sorted(set(d.loc[d[cohort] != never, cohort].dropna().unique()))
    staggered = len(cohorts) > 1

    # did2s 交叉核对：TWFE 动态效应在效应较大或交错时，pre 系数会被处理效应污染，
    # 出现假预趋势。用 did2s 的汇总 ATT 做对照——若两种估计量一致，
    # 说明是动态效应的基准期问题，不是识别失败。
    att_did2s = None
    try:
        es = pf.event_study(data=d[[y, idv, period, cohort]].rename(
            columns={y: "_y", idv: "_id", period: "_t", cohort: "_g"}),
            yname="_y", idname="_id", tname="_t", gname="_g", estimator="did2s")
        att_did2s = float(es.tidy().loc["is_treated", "Estimate"])
    except Exception:
        att_did2s = None

    warnings, status = [], "ok"
    if wt and wt["p"] < 0.05:
        status = "warn"
        msg = (f"预趋势未通过：k<0 的系数联合检验 p={wt['p']:.4f}"
               f"（{wt.get('distribution', 'chi2')} 口径）")
        if staggered and att_did2s is not None and att is not None:
            # 只有交错处理才存在负权重导致的基准期污染；单队列 DID 里 two 估计量等价，
            # 预趋势不过就是平行趋势违例，不能拿 did2s 去宽慰
            gap = abs(att_did2s - att) / abs(att_did2s) if att_did2s else float("nan")
            if np.sign(att_did2s) == np.sign(att) and gap < 0.25:
                msg += (f"；did2s 汇总 ATT={att_did2s:+.4f} 与 TWFE {att:+.4f} 相差 {gap:.0%}，"
                        "差异在 25% 以内，更像 TWFE 动态效应的基准期污染而非识别失败——"
                        "正文仍要交代平行趋势存疑并附 did2s 结果")
            else:
                msg += (f"；did2s 汇总 ATT={att_did2s:+.4f} 与 TWFE {att:+.4f} 相差 {gap:.0%}，"
                        "两个估计量不一致，这更像真的识别问题")
        else:
            msg += "；单队列 DID 没有负权重问题，这条就是平行趋势违例，需在正文交代"
        warnings.append(msg)
    stag = None
    if staggered:
        status = "warn"
        # 交错处理：不满足于报警，直接给一个没有负权重的估计量做对照
        try:
            from .staggered import bacon_pairs, cs_att_gt, heterogeneity_q

            stag = cs_att_gt(d, y, idv, period, cohort, never=never, window=(lo, hi))
            stag["pairs"] = bacon_pairs(d, y, idv, period, cohort, never=never,
                                        window=(lo, hi))
            stag["heterogeneity"] = heterogeneity_q(stag["att_gt"])
            cs = stag["aggregate"]
            if cs:
                gap = cs["att"] - att
                rel = gap / att if att else float("nan")
                if (att and np.sign(cs["att"]) != np.sign(att)) or abs(rel) > 0.5:
                    warnings.append(
                        f"CS 汇总 ATT={cs['att']:+.4f} 与 TWFE 动态效应均值 {att:+.4f} 相差 "
                        f"{gap:+.4f}（{abs(rel):.0%}）：TWFE 在交错处理下的负权重已经污染了估计，"
                        "正文应报 CS，TWFE 只作对照")
                else:
                    warnings.append(
                        f"CS 汇总 ATT={cs['att']:+.4f} 与 TWFE {att:+.4f} 接近，"
                        f"负权重影响不大，但仍需在正文交代用了哪个估计量")
            het = stag.get("heterogeneity")
            if het and het["min_p"] < 0.05:
                warnings.append(
                    f"各队列效应不一致（k={het['worst_k']} 的异质性检验 p={het['min_p']:.4f}）："
                    "处理效应随处理时点变化，单一 ATT 的解读要谨慎，建议分组报告")
                status = "warn"
        except Exception as exc:
            warnings.append(f"交错处理（{len(cohorts)} 个处理时点）：TWFE 存在负权重，"
                            f"CS 复核失败（{type(exc).__name__}: {exc}），需自行补做")
    if min(sum(d[n]) for n in present) < 10:
        warnings.append("某些事件时间只有不到 10 个观测，系数不稳")

    why = (f"DID 事件时间 {len(rows)} 期（基准 k=-1）"
           + (f"，处理后平均效应 {att:+.4f}（k>=0 各格等权，未按处理期长度加权）"
              if att is not None else "，窗口内没有 post 期")
           + (f"；预趋势联合检验 p={wt['p']:.3f}" if wt else ""))
    return Finding(fid, "did", status, why,
                   {"y": y, "cohort": cohort, "period": period, "window": [lo, hi],
                    "event_time": rows, "att_post": att, "pre_trend_wald": wt,
                    "endpoint_binned": binned,
                    "cohorts": [str(c) for c in cohorts], "staggered": staggered,
                    "att_did2s": att_did2s, "staggered_check": stag,
                    "formula": fml, "n": int(m._N)}, warnings=warnings)
