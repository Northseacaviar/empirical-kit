"""倾向得分匹配与 PSM-DID。

为什么必须自己写匹配而不用现成包：中文期刊里的 PSM 大多只要三件事——
1:1 最近邻（可加卡尺）、平衡性表（SMD）、共同支撑；PSM-DID 还要在匹配样本上再跑 TWFE。
现成包通常把结果封在对象里、无法与 spec 的样本流对接，也拿不到"哪些个体被丢弃"。

口径：
- 倾向得分用 logit（含常数项），与文献默认一致；
- 标准化偏差 SMD = (mean_t - mean_c) / sqrt((var_t + var_c)/2)，|SMD| > 10% 判不平衡
  （Rosenbaum & Rubin 1985 的经验门槛）；
- ATT 的标准误用 bootstrap（按个体重抽样，默认 199 次），不用 Abadie-Imbens 解析式——
  后者在 1:k 匹配与卡尺下公式很长，出错代价比 bootstrap 的耗时更贵。
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


def _propensity(d: pd.DataFrame, treat: str, covs: list[str]) -> np.ndarray:
    import statsmodels.api as sm

    X = sm.add_constant(d[covs].astype(float))
    m = sm.Logit(d[treat].astype(int), X).fit(disp=0)
    return np.asarray(m.predict(X))


def _smd(t: pd.Series, c: pd.Series) -> float:
    vt, vc = float(t.var(ddof=1)), float(c.var(ddof=1))
    pooled = np.sqrt((vt + vc) / 2)
    if pooled == 0:
        return 0.0
    return float((t.mean() - c.mean()) / pooled)


def _att(d: pd.DataFrame, treat: str, y: str, idx_t: np.ndarray,
         idx_c: np.ndarray) -> float:
    return float(d.iloc[idx_t][y].mean() - d.iloc[idx_c][y].mean())


def psm(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "psm")
    treat, y = cfg["treat"], cfg["y"]
    covs = cfg.get("covariates") or []
    if not covs:
        return Finding(fid, "psm", "fail", "没有给匹配变量 covariates", {})
    need = [treat, y] + list(covs)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "psm", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need).copy()
    d[treat] = d[treat].astype(int)
    if d[treat].nunique() != 2:
        return Finding(fid, "psm", "fail", f"处理变量 {treat} 必须恰好两个取值（0/1）", {})

    ratio = int(cfg.get("ratio", 1))
    caliper = cfg.get("caliper")
    replace = bool(cfg.get("replace", False))
    boot = int(cfg.get("boot", 199))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    try:
        d["_ps"] = _propensity(d, treat, covs)
    except Exception as exc:
        return Finding(fid, "psm", "fail", f"倾向得分估计失败（可能完全分离）：{exc}", {})

    t_idx = np.where(d[treat] == 1)[0]
    c_idx = np.where(d[treat] == 0)[0]
    matched_pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    unmatched = 0
    for i in t_idx:
        p = d["_ps"].values[i]
        dist = np.abs(d["_ps"].values[c_idx] - p)
        order = np.argsort(dist)
        picked = 0
        for j in order:
            if caliper is not None and dist[j] > caliper:
                break
            if not replace and j in used:
                continue
            matched_pairs.append((int(i), int(c_idx[j])))
            used.add(int(j))
            picked += 1
            if picked >= ratio:
                break
        if picked == 0:
            unmatched += 1
    if not matched_pairs:
        return Finding(fid, "psm", "fail",
                       f"没有一个处理组个体匹配成功（卡尺 {caliper} 可能太紧）",
                       {"n_treated": int(len(t_idx)), "n_control": int(len(c_idx))})

    mt = np.array([a for a, _ in matched_pairs])
    mc = np.array([b for _, b in matched_pairs])
    att = _att(d, treat, y, mt, mc)

    # bootstrap：按个体重抽样，重跑倾向得分与匹配（不能只重抽已匹配样本，那会低估不确定性）
    b_att = []
    for _ in range(boot):
        pick = rng.choice(len(d), size=len(d), replace=True)
        db = d.iloc[pick].reset_index(drop=True)
        if db[treat].nunique() != 2:
            continue
        try:
            db["_ps"] = _propensity(db, treat, covs)
        except Exception:
            continue
        tb = np.where(db[treat] == 1)[0]
        cb = np.where(db[treat] == 0)[0]
        uu: set[int] = set()
        pairs_b = []
        for i, p in zip(tb, db["_ps"].values[tb]):
            dist = np.abs(db["_ps"].values[cb] - p)
            order = np.argsort(dist)
            got = 0
            for j in order:
                if caliper is not None and dist[j] > caliper:
                    break
                if not replace and j in uu:
                    continue
                pairs_b.append((int(i), int(cb[j])))
                uu.add(int(j))
                got += 1
                if got >= ratio:
                    break
        if pairs_b:
            mtt = np.array([a for a, _ in pairs_b])
            mcc = np.array([b for _, b in pairs_b])
            b_att.append(_att(db, treat, y, mtt, mcc))
    if len(b_att) >= 20:
        se = float(np.std(b_att, ddof=1))
        lo, hi = np.percentile(b_att, [2.5, 97.5])
    else:
        se = lo = hi = float("nan")

    # 平衡性：匹配前后
    before = {c: _smd(d.loc[d[treat] == 1, c], d.loc[d[treat] == 0, c]) for c in covs}
    after = {c: _smd(d.iloc[mt][c], d.iloc[mc][c]) for c in covs}
    bad = [c for c, v in after.items() if abs(v) > 0.1]

    warnings, status = [], "ok"
    if bad:
        status = "warn"
        warnings.append("匹配后仍有变量 |SMD| > 10%：" + "、".join(bad)
                        + "（应当重新设定协变量或改用核匹配）")
    if unmatched:
        warnings.append(f"{unmatched} 个处理组个体在卡尺内找不到对照，已剔除"
                        "（说明共同支撑不足，正文要交代）")
    if replace:
        warnings.append("有放回匹配：同一个对照个体可能被多次使用，标准误按 bootstrap 计")

    why = (f"PSM（1:{ratio} {'有放回' if replace else '无放回'}，卡尺 {caliper}）："
           f"ATT={att:+.4f}，bootstrap SE={se:.4f}，95% CI [{lo:+.4f}, {hi:+.4f}]，"
           f"匹配 {len(matched_pairs)} 对，N={len(d)}")
    return Finding(fid, "psm", status, why,
                   {"treat": treat, "y": y, "covariates": covs, "att": att,
                    "se": se, "ci_low": float(lo), "ci_high": float(hi),
                    "n_matched": len(matched_pairs), "n_unmatched": unmatched,
                    "balance_before": before, "balance_after": after,
                    "unbalanced": bad, "ratio": ratio, "caliper": caliper,
                    "replace": replace, "n": len(d),
                    "matched_ids": [int(x) for x in mt],
                    "control_ids": [int(x) for x in mc]},
                   warnings=warnings)


def psm_did(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    """PSM-DID：先匹配，再在匹配样本上跑 TWFE（处理×后）。"""
    fid = cfg.get("id", "psm_did")
    if pf is None:
        return Finding(fid, "psm_did", "fail", "未安装 pyfixest", {})
    base = psm(df, {**cfg, "id": fid + "_psm", "y": cfg.get("y_pre") or cfg["y"]})
    if base.status == "fail":
        return Finding(fid, "psm_did", "fail", f"匹配阶段失败：{base.why}", {})

    ids = base.data["matched_ids"] + base.data["control_ids"]
    sub = df.loc[df.index.isin(ids)].copy()
    if len(sub) == 0:
        return Finding(fid, "psm_did", "fail", "匹配样本为空", {})

    entity, time = cfg["entity"], cfg["time"]
    post = cfg.get("post") or cfg.get("after")
    y = cfg["y"]
    if post not in sub.columns:
        return Finding(fid, "psm_did", "fail", f"缺少后处理指示变量 {post}", {})
    sub["_did"] = sub[cfg["treat"]].astype(int) * sub[post].astype(int)

    controls = cfg.get("controls") or []
    fml = f"{y} ~ 1 + _did" + ("".join(f" + {c}" for c in controls))
    fml += f" | {entity} + {time}"
    clusters = cfg.get("cluster") or [entity]
    vcov = {"CRV1": clusters[0]} if len(clusters) == 1 else {"CRV1": "+".join(clusters)}
    try:
        m = pf.feols(fml, data=sub, vcov=vcov)
        td = m.tidy()
        coef = float(td.loc["_did", "Estimate"])
        se = float(td.loc["_did", "Std. Error"])
        t = float(td.loc["_did", "t value"])
        p = float(td.loc["_did", "Pr(>|t|)"])
    except Exception as exc:
        return Finding(fid, "psm_did", "fail",
                       f"匹配样本上的 DID 估计失败：{type(exc).__name__}: {exc}",
                       {"formula": fml})

    warnings = list(base.warnings) + ["PSM-DID 的推断未考虑匹配阶段的不确定性："
                                      "严格做法是按个体 bootstrap 整条链（匹配 + DID）"]
    why = (f"PSM-DID：匹配 {base.data['n_matched']} 对后在匹配样本上估 TWFE，"
           f"DID 系数 {coef:+.4f}（t={t:.2f}，p={p:.4f}，N={int(m._N)}）")
    return Finding(fid, "psm_did", "warn" if base.status == "warn" else "ok", why,
                   {"coef": coef, "se": se, "t": t, "p": p, "n": int(m._N),
                    "matched_pairs": base.data["n_matched"],
                    "att_psm": base.data["att"],
                    "balance_after": base.data["balance_after"],
                    "formula": fml}, warnings=warnings)
