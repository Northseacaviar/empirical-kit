"""时间安慰剂：把事件日换到非事件日，看组间差是否还这么大。

判断逻辑固定：
- 计算事件日组间差在所有锚点组间差分布里的位置
- z = (事件日差 - 安慰剂均值) / 安慰剂标准差
- |z| < 1.96 → warn：事件日的效应用同一套口径在随便哪天也能复制出来
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding
from ..sample import build_sample
from ..spec import StudySpec
from .group import group_diff


def auto_anchors(spec: StudySpec, prices: dict, n: int = 10, gap: int = 20,
                 avoid: int = 10) -> list[str]:
    """事件日前 [260, avoid] 个交易日里等间隔取 n 个锚点。"""
    first = next(iter(prices.values()))
    dates = pd.DatetimeIndex(first.index).sort_values()
    ev = pd.Timestamp(spec.event.date)
    pos = int(dates.searchsorted(ev))
    start = max(0, pos - 260)
    end = max(start + 1, pos - avoid)
    picks = list(dates[start:end:gap])[-n:]
    return [str(p.date()) for p in picks]


def time_placebo(spec: StudySpec, cfg: dict, prices: dict | None = None,
                 xs: pd.DataFrame | None = None, market=None, verbose: bool = False) -> Finding:
    fid = cfg.get("id", "placebo")
    var, by = cfg["var"], cfg["by"]
    if spec.event is None or spec.event.mode != "single":
        return Finding(fid, "placebo", "fail", "安慰剂需要 event.mode=single", {})

    anchors = cfg.get("anchors")
    if not anchors or anchors == "auto":
        if prices is None:
            return Finding(fid, "placebo", "fail", "安慰剂需要价格数据（panel 模式不支持）", {})
        anchors = auto_anchors(spec, prices, cfg.get("n", 10), cfg.get("gap", 20),
                               cfg.get("avoid", 10))

    base_df, _ = build_sample(spec, prices=prices, xs=xs, market=market)
    base = group_diff(base_df, var, by, fid=f"{fid}_event")
    if base.status == "fail":
        return Finding(fid, "placebo", "fail", f"事件日基准检验失败：{base.why}", {**base.data})

    rows = []
    for a in anchors:
        d, _ = build_sample(spec, event_date=a, prices=prices, xs=xs, market=market)
        g = group_diff(d, var, by, fid=f"{fid}_{a}")
        if g.status == "fail" or g.data.get("diff") is None:
            rows.append({"anchor": a, "diff": None, "t": None})
            continue
        rows.append({"anchor": a, "diff": float(g.data["diff"]), "t": float(g.data["t"]),
                     "n": int(sum(x["n"] for x in g.data["groups"]))})
        if verbose:
            print(f"      锚点 {a}: 差 {g.data['diff']:+.3f}（t={g.data['t']:.2f}）")

    ok = [r for r in rows if r["diff"] is not None]
    if len(ok) < 3:
        return Finding(fid, "placebo", "fail", f"有效锚点只有 {len(ok)} 个，不足以做安慰剂",
                       {"anchors": rows})

    diffs = np.array([r["diff"] for r in ok])
    mu, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    event_diff = float(base.data["diff"])
    z = (event_diff - mu) / sd if sd > 0 else float("nan")
    # 经验 p 值：安慰剂里有多少锚点的偏离不小于事件日
    p_emp = float((np.abs(diffs - mu) >= abs(event_diff - mu)).mean())

    status, warnings = "ok", []
    if not np.isfinite(z) or abs(z) < 1.96:
        status = "warn"
        warnings.append(f"事件日差 {event_diff:+.3f} 落在安慰剂分布内（z={z:.2f}），"
                        "同一口径在非事件日也能复制出相近的组间差")
    if len(ok) < 8:
        warnings.append(f"只有 {len(ok)} 个锚点，分布估计不稳")
        status = "warn" if status == "ok" else status

    why = (f"{var} 组间差：事件日 {event_diff:+.3f}；"
           f"{len(ok)} 个安慰剂锚点均值 {mu:+.3f}（sd {sd:.3f}），z={z:.2f}，经验 p={p_emp:.3f}")
    return Finding(fid, "placebo", status, why,
                   {"var": var, "by": by, "event_date": str(spec.event.date),
                    "event_diff": event_diff,
                    "placebo_mean": mu, "placebo_sd": sd, "z": z, "p_empirical": p_emp,
                    "anchors": rows}, warnings=warnings)
