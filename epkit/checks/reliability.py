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


def _clean(u: list) -> list:
    return [v for v in u
            if v is not None and not (isinstance(v, float) and np.isnan(v))]


def krippendorff_alpha(units: list[list], level: str = "nominal") -> float:
    """按 coincidence matrix 算 Krippendorff α（nominal / ordinal / interval）。

    为什么用 coincidence matrix 而不是「池化有序对」：每单元编码者数 m_u 不等时
    （缺失、多编码者），池化写法等价于按 m_u(m_u-1) 加权，会把 α 系统性算高；
    标准定义是每单元先除以 (m_u-1) 再汇总。
    """
    cleaned = [_clean(u) for u in units]
    vals = [v for u in cleaned for v in u]
    if not vals:
        return float("nan")
    try:
        cats = sorted(set(vals))
    except TypeError:
        cats = sorted(set(vals), key=repr)
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)

    o = np.zeros((K, K))
    for u in cleaned:
        m = len(u)
        if m < 2:
            continue
        counts = np.zeros(K)
        for x in u:
            counts[idx[x]] += 1
        for c in range(K):
            if counts[c] == 0:
                continue
            for k in range(K):
                if counts[k] == 0:
                    continue
                if c == k:
                    o[c, c] += counts[c] * (counts[c] - 1) / (m - 1)
                else:
                    o[c, k] += counts[c] * counts[k] / (m - 1)
    n = float(o.sum())
    if n < 2:
        return float("nan")
    nc = o.sum(axis=1)

    if level == "nominal":
        def delta(c: int, k: int) -> float:
            return 0.0 if c == k else 1.0
    elif level == "interval":
        def delta(c: int, k: int) -> float:
            return float((c - k) ** 2)
    elif level == "ordinal":
        def delta(c: int, k: int) -> float:
            lo, hi = min(c, k), max(c, k)
            span = nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2.0
            return float(span ** 2)
    else:
        raise ValueError(f"不支持的 level={level!r}；可选 nominal / ordinal / interval")

    D_o = sum(o[c, k] * delta(c, k) for c in range(K) for k in range(K) if c != k) / n
    D_e = sum(nc[c] * nc[k] * delta(c, k)
              for c in range(K) for k in range(K) if c != k) / (n * (n - 1))
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

    level = cfg.get("level", "nominal")
    if level not in ("nominal", "ordinal", "interval"):
        return Finding(fid, "reliability", "fail",
                       f"不支持的 level={level!r}；当前实现 nominal / ordinal / interval", {})
    units = [[r[c] for c in cols] for _, r in df[cols].iterrows()]
    try:
        alpha = krippendorff_alpha(units, level)
    except ValueError as exc:
        return Finding(fid, "reliability", "fail", str(exc), {})
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
                   {"alpha": alpha, "level": level,
                    "coders": cols, "n_units": n_units, "n_complete": n_complete,
                    "agreement": agree, "min_alpha": min_alpha},
                   warnings=warnings)
