"""分布诊断：偏度、峰度、尾部占比，以及供出图的核密度数据。

为什么要有：均值与标准差在偏态变量上会骗人。补贴金额、市值、交易额这类
右偏变量，不取对数就进回归，等于让几十个极端值决定系数。论文的变量表
通常只给均值/标准差，审稿人看不出这一点，作者自己也常常不看。

判据：|偏度| > 2 或超额峰度 > 7 视为严重偏态，提示对数化或缩尾。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding


def density(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    fid = cfg.get("id", "density")
    variables = cfg.get("vars", [])
    by = cfg.get("by")
    if not variables:
        return Finding(fid, "density", "fail", "没有指定变量", {})
    missing = [v for v in variables if v not in df.columns]
    if missing:
        return Finding(fid, "density", "fail", f"列不存在：{missing}", {})

    items, warnings = [], []
    for v in variables:
        s = df[v].dropna().astype(float)
        if len(s) < 3:
            items.append({"var": v, "n": int(len(s))})
            continue
        skew = float(s.skew())
        kurt = float(s.kurt())          # pandas 的 kurt 已是超额峰度
        q01, q99 = s.quantile(0.01), s.quantile(0.99)
        span = float(s.max() - s.min()) or 1.0
        # 尾部占比按定义恒为 2%，没有信息量；改用「缩尾前后均值变化占标准差的比例」
        sd = float(s.std(ddof=1)) or 1.0
        winsor_gap = float(abs(s.mean() - s.clip(q01, q99).mean()) / sd)
        tail = winsor_gap
        # 出图要用的直方密度：在这里算一次，图模块只负责画
        counts, edges = np.histogram(s, bins=40, density=True)
        hist = {"edges": [float(e) for e in edges], "density": [float(c) for c in counts]}
        items.append({"var": v, "n": int(len(s)), "skew": skew, "kurt": kurt, "hist": hist,
                      "p01": float(q01), "p99": float(q99), "winsor_gap": tail,
                      "mean": float(s.mean()), "median": float(s.median()),
                      "max_min_span": span,
                      "max_over_p99": float(s.max() / q99) if q99 else None})
        if abs(skew) > 2 or kurt > 7:
            warnings.append(f"{v} 偏度 {skew:.2f}、超额峰度 {kurt:.2f}："
                            "严重偏离正态，回归前考虑取对数或缩尾")

    if by and by in df.columns:
        groups = []
        for g, sub in df.groupby(by):
            rec = {"group": str(g), "n": int(len(sub))}
            for v in variables:
                ser = sub[v].dropna().astype(float)
                rec[v] = {"mean": float(ser.mean()) if len(ser) else None,
                          "median": float(ser.median()) if len(ser) else None}
            groups.append(rec)
    else:
        groups = None

    status = "warn" if warnings else "ok"
    why = f"分布诊断 {len(items)} 个变量" + (f"，{len(warnings)} 个严重偏态" if warnings else "，均无明显偏态")
    return Finding(fid, "density", status, why,
                   {"items": items, "by": by, "groups": groups}, warnings=warnings)
