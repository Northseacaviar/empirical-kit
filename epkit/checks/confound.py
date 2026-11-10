"""混杂事件筛查：窗口里还发生了别的什么事。

为什么必须做：事件研究的洁净性假设是"窗口内只有这一件事"。
现实中降准、业绩预告、监管处罚常常挨着公告日，窗口里的异常收益
分不清是谁的。审稿人一问就露馅，所以先把重叠算出来。

口径：主事件日来自 spec.event；每个混杂事件可以只影响某些代码（codes），
不给 codes 就视为全体。重叠 = 主事件窗口内出现该事件日。
"""
from __future__ import annotations

import pandas as pd

from ..findings import Finding


def _parse_date(s) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(str(s))
    except Exception:
        return None


def confound(spec, cfg: dict) -> Finding:
    fid = cfg.get("id", "confound")
    events = cfg.get("other_events") or []
    if not events:
        return Finding(fid, "confound", "fail", "没有给出 other_events", {})

    # 主事件日：single 用 date；column/list 用样本里的日期列（这里只做 single 的精确筛查）
    main = None
    if spec.event is not None and spec.event.mode == "single" and spec.event.date:
        main = _parse_date(spec.event.date)
    if main is None:
        return Finding(fid, "confound", "warn",
                       "主事件不是单一日期（column/list 模式），窗口重叠需逐行判断，本项不适用",
                       {})

    windows = cfg.get("windows") or spec.returns.windows
    all_codes = cfg.get("codes")

    overlaps, rows = [], []
    for ev in events:
        d = _parse_date(ev.get("date") or ev.get("day"))
        hit_windows = []
        if d is not None:
            for w in windows:
                lo, hi = int(w[0]), int(w[1])
                # 用交易日近似日历年偏移：窗口按交易日计，这里给保守的 ±(hi-lo) 自然日估计
                start = main + pd.Timedelta(days=lo * 2)
                end = main + pd.Timedelta(days=hi * 2)
                if start <= d <= end:
                    hit_windows.append(f"[{lo},{hi}]")
        rec = {"date": str(ev.get("date") or ev.get("day")), "label": ev.get("label", ""),
               "codes": ev.get("codes") or all_codes, "hits": hit_windows}
        rows.append(rec)
        if hit_windows:
            overlaps.append(rec)

    n = len(events)
    share = len(overlaps) / n if n else 0.0
    warnings, status = [], "ok"
    if overlaps:
        status = "warn"
        detail = "；".join(f"{o['date']} {o['label']} 落在 {'/'.join(o['hits'])}"
                          for o in overlaps[:5])
        warnings.append(f"{len(overlaps)}/{n} 个混杂事件与主窗口重叠：{detail}")
        warnings.append("重叠样本的 CAR 不能单独归因于主事件；"
                        "要么剔除这些股票-事件，要么在正文交代并做剔除后的稳健性")

    why = (f"混杂事件 {n} 个，与窗口重叠 {len(overlaps)} 个（{share:.0%}）"
           if n else "没有给出混杂事件")
    return Finding(fid, "confound", status, why,
                   {"main_event": str(main.date()), "windows": windows,
                    "events": rows, "n_overlap": len(overlaps)},
                   warnings=warnings)
