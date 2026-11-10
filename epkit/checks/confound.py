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
    empty = {"main_event": None, "windows": [], "events": [], "n_overlap": 0}
    if not events:
        return Finding(fid, "confound", "fail", "没有给出 other_events", empty)

    # 主事件日：single 用 date；column/list 用样本里的日期列（这里只做 single 的精确筛查）
    main = None
    if spec.event is not None and spec.event.mode == "single" and spec.event.date:
        main = _parse_date(spec.event.date)
    if main is None:
        return Finding(fid, "confound", "warn",
                       "主事件不是单一日期（column/list 模式），窗口重叠需逐行判断，本项不适用",
                       empty)

    windows = cfg.get("windows") or spec.returns.windows
    all_codes = cfg.get("codes")
    # 真实交易日历（可选）：给了就用交易日索引映射窗口，没给退回 ×2 自然日的保守近似
    cal = [pd.Timestamp(str(x)) for x in (cfg.get("trading_days") or [])]
    cal = sorted([c for c in cal if pd.notna(c)])

    def window_dates(w) -> tuple[pd.Timestamp, pd.Timestamp]:
        lo, hi = int(w[0]), int(w[1])
        if cal:
            try:
                i0 = cal.index(main)
            except ValueError:
                i0 = None
            if i0 is not None:
                a = cal[max(min(i0 + lo, len(cal) - 1), 0)]
                b = cal[max(min(i0 + hi, len(cal) - 1), 0)]
                return a, b
        return main + pd.Timedelta(days=lo * 2), main + pd.Timedelta(days=hi * 2)

    overlaps, rows = [], []
    for ev in events:
        d = _parse_date(ev.get("date") or ev.get("day"))
        hit_windows = []
        if d is not None:
            for w in windows:
                start, end = window_dates(w)
                if start <= d <= end:
                    hit_windows.append(f"[{int(w[0])},{int(w[1])}]")
        ev_codes = ev.get("codes") or all_codes
        if hit_windows and ev_codes:
            # 事件只影响部分股票时，只有主事件样本里存在这些代码才算污染
            sample_codes = cfg.get("sample_codes")
            if sample_codes is not None:
                shared = set(str(c) for c in ev_codes) & set(str(c) for c in sample_codes)
                if not shared:
                    hit_windows = []
        rec = {"date": str(ev.get("date") or ev.get("day")), "label": ev.get("label", ""),
               "codes": ev_codes, "hits": hit_windows}
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
