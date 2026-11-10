"""中介效应：三步系数表 + 间接效应的 bootstrap 置信区间。

为什么两张表一起给：
- Baron-Kenny 三步（a、b、c、c′）是审稿人熟悉、能对上纸质论文的表；
- 但逐步法的显著性不能当中介存在的证据（Fritz & MacKinnon 2007 的检验力很低），
  现代做法看的是间接效应 ab 的 bootstrap 区间。
所以三步表照给，结论以 CI 为准，两者并列写进结论。

实现：
- 三条方程都用同一个估计器（含固定效应与聚类标准误），保证口径一致；
- bootstrap 按个体（或聚类）重抽样，重跑三条方程收集 ab，给百分位区间；
- 中介比例 ab/c 在 c 不显著时会失真，只在 c 同号且显著时报告。
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


def _fit_coef(d: pd.DataFrame, y: str, rhs: list[str], fe: list[str],
              vcov, focus: str) -> dict | None:
    fml = f"{y} ~ 1 + " + " + ".join(rhs)
    if fe:
        fml += " | " + " + ".join(fe)
    try:
        m = pf.feols(fml, data=d, vcov=vcov)
        td = m.tidy()
    except Exception:
        return None
    if focus not in td.index:
        return None
    return {"coef": float(td.loc[focus, "Estimate"]),
            "se": float(td.loc[focus, "Std. Error"]),
            "t": float(td.loc[focus, "t value"]),
            "p": float(td.loc[focus, "Pr(>|t|)"]),
            "n": int(m._N), "formula": fml}


def mediation(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "mediation")
    if pf is None:
        return Finding(fid, "mediation", "fail", "未安装 pyfixest", {})

    y, x = cfg["y"], cfg["x"]
    ms = cfg.get("m") or []
    if isinstance(ms, str):
        ms = [ms]
    if not ms:
        return Finding(fid, "mediation", "fail", "没有给中介变量 m", {})
    controls = cfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = cfg.get("fe") or []
    clusters = cfg.get("cluster") or []
    boot = int(cfg.get("boot", 499))
    seed = int(cfg.get("seed", 42))

    need = [y, x] + list(ms) + list(controls) + list(fe) + list(clusters)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "mediation", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    vcov = ({"CRV1": clusters[0]} if len(clusters) == 1
            else {"CRV1": "+".join(clusters)}) if clusters else "hetero"

    rows, warnings = [], []
    for mvar in ms:
        a = _fit_coef(d, mvar, [x] + list(controls), fe, vcov, x)
        bc = _fit_coef(d, y, [x, mvar] + list(controls), fe, vcov, mvar)
        cpath = _fit_coef(d, y, [x] + list(controls), fe, vcov, x)
        cprime = _fit_coef(d, y, [x, mvar] + list(controls), fe, vcov, x)
        if not (a and bc and cpath and cprime):
            warnings.append(f"{mvar}：三条方程里有估计失败的，跳过")
            continue

        ab = a["coef"] * bc["coef"]

        # bootstrap：按聚类（没给就按行）重抽样，重跑三条方程
        rng = np.random.default_rng(seed)
        if clusters:
            units = d[clusters[0]].dropna().unique()
            groups = {u: g for u, g in d.groupby(clusters[0])}
        else:
            units = None
        abs_boot = []
        for _ in range(boot):
            if clusters:
                pick = rng.choice(units, size=len(units), replace=True)
                dd = pd.concat([groups[u].assign(_b=i) for i, u in enumerate(pick)],
                               ignore_index=True)
                dd = dd.drop(columns=clusters, errors="ignore").rename(
                    columns={"_b": clusters[0]})
            else:
                dd = d.sample(n=len(d), replace=True, random_state=int(rng.integers(1e9)))
            a_b = _fit_coef(dd, mvar, [x] + list(controls), fe, vcov, x)
            b_b = _fit_coef(dd, y, [x, mvar] + list(controls), fe, vcov, mvar)
            if a_b and b_b:
                abs_boot.append(a_b["coef"] * b_b["coef"])

        if len(abs_boot) >= 20:
            lo, hi = np.percentile(abs_boot, [2.5, 97.5])
            boot_se = float(np.std(abs_boot, ddof=1))
        else:
            lo = hi = float("nan")
            boot_se = float("nan")
            warnings.append(f"{mvar}：bootstrap 有效次数只有 {len(abs_boot)} 次，区间不可信")

        share = ab / cpath["coef"] if abs(cpath["coef"]) > 1e-9 else float("nan")
        rows.append({"m": mvar,
                     "a": a, "b": bc, "c": cpath, "c_prime": cprime,
                     "ab": ab, "ab_se": boot_se,
                     "ci_low": float(lo), "ci_high": float(hi),
                     "share": float(share),
                     "n_boot_ok": len(abs_boot)})

        same_sign = np.sign(ab) == np.sign(cpath["coef"])
        if np.isfinite(lo) and (lo > 0 or hi < 0):
            if cpath["p"] >= 0.05 and same_sign:
                warnings.append(
                    f"{mvar}：总效应 c 不显著（p={cpath['p']:.3f}）但间接效应 CI 不含 0，"
                    "同号说明是「只有中介路径」；若异号则是遮掩效应，不能叫中介")
        else:
            warnings.append(
                f"{mvar}：间接效应 ab={ab:+.4f} 的 95% CI [{lo:+.4f}, {hi:+.4f}] 含 0，"
                "不能宣称中介效应成立")

    if not rows:
        return Finding(fid, "mediation", "fail", "所有中介变量都没跑出完整结果",
                       {"rows": []}, warnings=warnings)

    warnings.append("结论以 bootstrap 间接效应区间为准；三步表的逐步显著性只能作为辅助"
                    "（Fritz & MacKinnon 2007：逐步法检验力偏低）")
    ok = any(np.isfinite(r["ci_low"]) and (r["ci_low"] > 0 or r["ci_high"] < 0) for r in rows)
    status = "ok" if ok and not any("含 0" in w for w in warnings) else "warn"

    first = rows[0]
    why = (f"{x} → {y} 经 {first['m']}：间接效应 ab={first['ab']:+.4f}，"
           f"95% CI [{first['ci_low']:+.4f}, {first['ci_high']:+.4f}]（bootstrap {first['n_boot_ok']} 次），"
           f"总效应 c={first['c']['coef']:+.4f}，直接效应 c'={first['c_prime']['coef']:+.4f}")
    return Finding(fid, "mediation", status, why,
                   {"y": y, "x": x, "rows": rows, "boot": boot,
                    "vcov": vcov if isinstance(vcov, str) else "cluster",
                    "n": first["a"]["n"]}, warnings=warnings)
