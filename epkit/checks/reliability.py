"""双编码信度：两个编码者给同一批文本打标签，一致性够不够高。

为什么要有：内容分析里"我们手工编码了 352 份公告"这句话，审稿人下一步就问
信度是多少。Krippendorff α 是目前引用最多、又能处理缺失与多编码者的系数
（Cohen κ 只适用于两人且无缺失、且对边际分布敏感）。

实现：nominal 口径的 α。可接受下限 0.8（探索性研究 0.667）。
多个编码列时按 Krippendorff 的定义合并计算。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..findings import Finding


def krippendorff_alpha(units: list[list], level: str = "nominal") -> float:
    """units: 每个单元一个列表，含该单元上各编码者的取值（可缺失、可不等长）。"""
    pairs_total, pairs_disagree = 0.0, 0.0
    values: list = []
    for u in units:
        vals = [v for v in u if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) < 2:
            continue
        n = len(vals)
        pairs_total += n * (n - 1)
        dis = sum(1 for a, b in itertools.combinations(vals, 2) if a != b)
        pairs_disagree += dis * 2.0       # 每一对有序两次
        values.extend(vals)
    if pairs_total == 0:
        return float("nan")
    D_o = pairs_disagree / pairs_total

    s = pd.Series(values)
    N = len(s)
    if N < 2:
        return float("nan")
    vc = s.value_counts()
    D_e = 1 - float((vc * (vc - 1)).sum()) / (N * (N - 1))
    if D_e == 0:
        return 1.0 if D_o == 0 else float("nan")
    return float(1 - D_o / D_e)


def reliability(df: pd.DataFrame, cfg: dict) -> Finding:
    fid = cfg.get("id", "reliability")
    cols = cfg.get("coders") or []
    if len(cols) < 2:
        return Finding(fid, "reliability", "fail",
                       "至少需要两个编码列（coders: [a, b]）", {})
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return Finding(fid, "reliability", "fail", f"列不存在：{missing}", {})

    units = [[r[c] for c in cols] for _, r in df[cols].iterrows()]
    alpha = krippendorff_alpha(units, cfg.get("level", "nominal"))
    min_alpha = float(cfg.get("min_alpha", 0.8))
    n_units = len(units)
    n_complete = int(df[cols].notna().all(axis=1).sum())

    agree = None
    if len(cols) == 2:
        both = df[cols].dropna()
        if len(both):
            agree = float((both[cols[0]] == both[cols[1]]).mean())

    warnings, status = [], "ok"
    if not np.isfinite(alpha):
        status = "warn"
        warnings.append("α 无法计算：编码列全缺失或所有单元只有一个编码者")
    elif alpha < min_alpha:
        status = "warn"
        warnings.append(f"α={alpha:.3f} < {min_alpha}：编码一致性不足，"
                        "先统一编码手册、复训后重编码，再报结果")
    if n_complete < n_units:
        warnings.append(f"{n_units - n_complete} 个单元不是所有编码者都给了值，"
                        "α 已按缺失处理，但正文要说明缺失比例")

    why = (f"Krippendorff α={alpha:.3f}（{len(cols)} 个编码者，{n_complete}/{n_units} 单元完整）"
           + (f"，完全一致率 {agree:.1%}" if agree is not None else ""))
    return Finding(fid, "reliability", status, why,
                   {"alpha": alpha, "level": cfg.get("level", "nominal"),
                    "coders": cols, "n_units": n_units, "n_complete": n_complete,
                    "agreement": agree, "min_alpha": min_alpha},
                   warnings=warnings)
