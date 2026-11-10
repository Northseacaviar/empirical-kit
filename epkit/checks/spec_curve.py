"""规格曲线：把"换一组控制变量、换一套推断口径，结论还成不成立"变成一张分布图。

为什么要做：审稿人问的是"你这个结果稳不稳"，而作者的回答常常是"我试了几种都显著"——
试了哪些、怎么说选的，全凭记忆。规格曲线把每个设定都摊开，中位数和显著比例
自己会说话。

口径：所有设定共用同一个聚焦变量 x，只在 controls / fe / cluster / 样本口径上变。
不做"换个因变量"这种更宽的自由度——那属于另一项检验，混进来就没法解读了。
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def _vcov(name: str):
    if name in ("iid", "hetero"):
        return name
    return {"CRV1": name.replace("cluster:", "")}


def spec_curve(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "spec_curve")
    if pf is None:
        return Finding(fid, "spec_curve", "fail", "未安装 pyfixest", {})
    y, x = cfg["y"], cfg["x"]
    controls = cfg.get("controls", [[]])
    if controls and isinstance(controls[0], str):
        controls = [controls]
    if not controls:
        controls = [[]]
    fes = cfg.get("fe", [[]])
    if fes and isinstance(fes[0], str):
        fes = [fes]
    if not fes:
        fes = [[]]
    vcovs = cfg.get("vcov", ["hetero"])
    if isinstance(vcovs, str):
        vcovs = [vcovs]
    samples = cfg.get("sample", [{}])
    max_specs = int(cfg.get("max_specs", 240))

    combos = list(itertools.product(range(len(controls)), range(len(fes)),
                                    range(len(vcovs)), range(len(samples))))
    if len(combos) > max_specs:
        return Finding(fid, "spec_curve", "fail",
                       f"设定组合 {len(combos)} 个超过上限 {max_specs}："
                       "先砍掉一些控制变量组或口径，否则跑完也没法读", {})

    rows, errors = [], 0
    for ci, fi, vi, si in combos:
        ctrls, fe, vcov_n, samp = controls[ci], fes[fi], vcovs[vi], samples[si]
        sub = df
        if samp.get("drop_where"):
            try:
                sub = df.query(samp["drop_where"])
            except Exception:
                errors += 1
                continue
        rhs = [x] + list(ctrls)
        fml = f"{y} ~ 1 + " + " + ".join(rhs)
        if fe:
            fml += " | " + " + ".join(fe)
        try:
            m = pf.feols(fml, data=sub, vcov=_vcov(vcov_n))
            td = m.tidy()
            if x not in td.index:
                errors += 1
                continue
            rows.append({
                "coef": float(td.loc[x, "Estimate"]),
                "se": float(td.loc[x, "Std. Error"]),
                "t": float(td.loc[x, "t value"]),
                "p": float(td.loc[x, "Pr(>|t|)"]),
                "n": int(m._N),
                "controls": ctrls, "fe": fe, "vcov": vcov_n,
                "sample": samp.get("label", samp.get("drop_where", "")),
                "formula": fml,
            })
        except Exception:
            errors += 1

    if not rows:
        return Finding(fid, "spec_curve", "fail",
                       f"{len(combos)} 个设定全部估计失败（失败 {errors} 个）", {})

    b = np.array([r["coef"] for r in rows])
    alpha = float(cfg.get("alpha", 0.05))
    sig = np.array([r["p"] < alpha for r in rows])
    expect = cfg.get("sign")
    same_sign = None
    if expect in (1, "+", "pos"):
        same_sign = float((b > 0).mean())
    elif expect in (-1, "-", "neg"):
        same_sign = float((b < 0).mean())

    summary = {
        "n_specs": len(rows), "n_failed": errors,
        "median": float(np.median(b)), "mean": float(b.mean()),
        "min": float(b.min()), "max": float(b.max()),
        "p10": float(np.percentile(b, 10)), "p90": float(np.percentile(b, 90)),
        "share_sig": float(sig.mean()), "alpha": alpha,
        "share_positive": float((b > 0).mean()),
    }
    if same_sign is not None:
        summary["share_expected_sign"] = same_sign

    warnings, status = [], "ok"
    if summary["share_sig"] < 0.5:
        status = "warn"
        warnings.append(f"只有 {summary['share_sig']:.0%} 的设定在 {alpha:.0%} 水平上显著："
                        f"结论对设定敏感，正文不能只报显著的那一个")
    if summary["min"] < 0 < summary["max"]:
        status = "warn"
        warnings.append("系数的符号在不同设定间翻转：这是最需要在正文里交代的情形")
    if same_sign is not None and same_sign < 0.9:
        status = "warn"
        warnings.append(f"只有 {same_sign:.0%} 的设定符合预期符号")

    why = (f"规格曲线 {len(rows)} 个设定：中位系数 {summary['median']:+.4f}"
           f"（{summary['p10']:+.4f} ~ {summary['p90']:+.4f}），"
           f"显著比例 {summary['share_sig']:.0%}"
           + (f"，预期符号占比 {same_sign:.0%}" if same_sign is not None else ""))
    return Finding(fid, "spec_curve", status, why,
                   {"y": y, "x": x, "specs": rows, "summary": summary},
                   warnings=warnings)
