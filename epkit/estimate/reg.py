"""回归层：OLS / 固定效应 / IV，一次给出全部推断口径。

为什么必须一次给全：同一份数据换聚类口径，t 值能从 2.3 掉到 1.6，
结论的严格程度就变了。论文里手写三个 V 矩阵再拼双向，是最容易出错的一段。

用法见 spec 的 tests[].reg；本模块不接触样本，只接收已建好的横截面表。
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


def _formula(y: str, xs: list[str], controls: list[str], fe: list[str],
             iv: dict[str, Any] | None = None) -> str:
    right = " + ".join(xs + controls) if (xs + controls) else "1"
    f = f"{y} ~ {right}"
    if fe:
        f += " | " + " + ".join(fe)
    if iv:
        endog = " + ".join(iv["endog"])
        inst = " + ".join(iv["instruments"])
        f += f" | {endog} ~ {inst}"
    return f


def _vcov_specs(clusters: list[str]) -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = [("iid", "iid"), ("hetero", "hetero")]
    for c in clusters:
        specs.append((f"cluster_{c}", {"CRV1": c}))
    if len(clusters) >= 2:
        specs.append(("cluster_twoway", {"CRV1": "+".join(clusters)}))
    return specs


def _n_clusters(df: pd.DataFrame, vcov: Any) -> int | None:
    if isinstance(vcov, dict) and "CRV1" in vcov:
        vars_ = str(vcov["CRV1"]).split("+")
        return int(min(df[v].nunique() for v in vars_ if v in df.columns))
    return None


def reg(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    if pf is None:
        return Finding(cfg.get("id", "reg"), "reg", "fail", "未安装 pyfixest", {})
    fid = cfg.get("id", "reg")
    y = cfg["y"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    fe = cfg.get("fe", []) or []
    clusters = cfg.get("cluster", []) or []
    ladder = cfg.get("controls", [])
    if ladder and not isinstance(ladder[0], list):
        ladder = [ladder]
    if not ladder:
        ladder = [[]]
    iv = cfg.get("iv")
    focus = cfg.get("focus") or (xs[0] if xs else None)

    vcovs = _vcov_specs(clusters)
    base = cfg.get("main_vcov") or next((n for n, _ in reversed(vcovs)
                                         if n.startswith("cluster")), "hetero")

    steps, warnings, status = [], [], "ok"
    for i, ctrls in enumerate(ladder):
        fml = _formula(y, xs, ctrls, fe, iv)
        label = "+".join(ctrls) if ctrls else "单变量"
        fitted = {}
        try:
            for vname, vspec in vcovs:
                m = pf.feols(fml, data=df, vcov=vspec)
                tidy = m.tidy()
                row = {}
                for xv in xs:
                    if xv not in tidy.index:
                        continue
                    row[xv] = {
                        "coef": float(tidy.loc[xv, "Estimate"]),
                        "se": float(tidy.loc[xv, "Std. Error"]),
                        "t": float(tidy.loc[xv, "t value"]),
                        "p": float(tidy.loc[xv, "Pr(>|t|)"]),
                    }
                g = int(min(m._G)) if getattr(m, "_G", None) else _n_clusters(df, vspec)
                fitted[vname] = {"terms": row, "n": int(m._N),
                                 "r2": float(m._r2) if m._r2 is not None else None,
                                 "g": g}
        except Exception as exc:
            steps.append({"label": label, "formula": fml, "error": f"{type(exc).__name__}: {exc}"})
            status = "fail"
            warnings.append(f"{label} 估计失败：{exc}")
            continue

        main = fitted.get(base, {})
        step = {"label": label, "formula": fml, "main_vcov": base, "focus": focus,
                "coef": main.get("terms", {}).get(focus, {}).get("coef"),
                "n": main.get("n"), "r2": main.get("r2"), "by_vcov": fitted}
        steps.append(step)

        # 口径敏感性：主口径显著而某口径不显著，或符号翻转
        mt = main.get("terms", {}).get(focus, {}).get("t")
        for vname, v in fitted.items():
            t = v["terms"].get(focus, {}).get("t")
            if mt is None or t is None:
                continue
            if np.sign(mt) != np.sign(t):
                warnings.append(f"{label}：{vname} 的系数符号与主口径({base})相反")
                status = "warn"
            elif abs(mt) >= 1.96 > abs(t):
                warnings.append(f"{label}：主口径 t={mt:.2f} 显著，{vname} t={t:.2f} 不显著")
                status = "warn"
        gs = [v["g"] for v in fitted.values() if v["g"]]
        if gs and min(gs) < 30:
            warnings.append(f"{label}：最小聚类数 {min(gs)} < 30，聚类标准误不可靠")
            status = "warn"

    main0 = next((s for s in steps if "by_vcov" in s), None)
    if main0:
        coef = main0.get("coef")
        t_main = main0["by_vcov"][base]["terms"].get(focus, {}).get("t")
        why = (f"{focus} → {y}：β={coef:.4f}，t={t_main:.2f}（{base}，N={main0['n']}）")
    else:
        why = "所有设定均估计失败"
    return Finding(fid, "reg", status, why,
                   {"y": y, "x": xs, "fe": fe, "cluster": clusters, "main_vcov": base,
                    "steps": steps, "focus": focus}, warnings=warnings)
