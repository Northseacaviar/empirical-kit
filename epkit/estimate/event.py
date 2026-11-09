"""事件研究：异常收益与多窗口 CAR。

三种基准口径（spec.returns.method）：
- raw               : 原始收益
- market_adjusted   : r - m
- market_model      : r - (alpha + beta * m)，估计窗口内 OLS

两种窗口口径（spec.returns.car_mode）不可混用，见 spec.ReturnsSpec 的说明。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..spec import StudySpec


def window_label(a: int, b: int) -> str:
    fa = f"m{abs(a)}" if a < 0 else str(a)
    fb = f"m{abs(b)}" if b < 0 else str(b)
    return f"car_{fa}_{fb}"


@dataclass
class PanelResult:
    panel: pd.DataFrame      # 每行一个 code/事件，含 car_* 列
    flow: list[dict]         # 计算阶段的剔除统计
    notes: list[str]


def _resolve_anchor(dates: pd.DatetimeIndex, event_date: pd.Timestamp, rule: str) -> int | None:
    if len(dates) == 0:
        return None
    if rule == "last_on_or_before":
        pos = dates.searchsorted(event_date, side="right") - 1
        return int(pos) if pos >= 0 else None
    pos = dates.searchsorted(event_date, side="left")
    return int(pos) if pos < len(dates) else None


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.astype(float).pct_change(fill_method=None)


def _market_model_ar(r: pd.Series, m: pd.Series, est: pd.DataFrame | None,
                     min_obs: int) -> pd.Series | None:
    """在估计窗口里估 r = a + b*m，返回全样本的异常收益。估计不足返回 None。"""
    joined = pd.concat([r.rename("r"), m.rename("m")], axis=1).dropna()
    if est is None:
        return None
    est_j = joined.loc[est.index[0]: est.index[-1]]
    if len(est_j) < min_obs:
        return None
    if est_j["m"].std(ddof=0) == 0:
        return None
    X = np.column_stack([np.ones(len(est_j)), est_j["m"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, est_j["r"].to_numpy(), rcond=None)
    return joined["r"] - (coef[0] + coef[1] * joined["m"])


def build_event_panel(spec: StudySpec, prices: dict[str, pd.DataFrame],
                      market: pd.Series | None) -> PanelResult:
    cfg = spec.returns
    dcfg = spec.data
    close_col = dcfg.close_col
    windows = [(int(a), int(b)) for a, b in cfg.windows]
    rows: list[dict] = []
    n_total = 0
    no_anchor = 0
    short = 0
    est_fail = 0

    for code, df in prices.items():
        n_total += 1
        close = pd.to_numeric(df[close_col], errors="coerce") if close_col in df.columns else None
        if close is None or close.dropna().empty:
            short += 1
            continue
        close = close.dropna()
        dates = pd.DatetimeIndex(close.index)
        event_date = pd.Timestamp(spec.event.date)
        i = _resolve_anchor(dates, event_date, spec.event.anchor_rule)
        if i is None:
            no_anchor += 1
            continue

        r = _daily_returns(close)
        ar = r
        if cfg.method != "raw":
            if market is None:
                raise ValueError(f"method={cfg.method} 需要 spec.data.index")
            m_all = market.reindex(r.index)
            if cfg.method == "market_adjusted":
                ar = r - m_all
            else:
                est = None
                if cfg.estimation_start and cfg.estimation_end:
                    est = pd.DataFrame(index=r.loc[cfg.estimation_start: cfg.estimation_end].index)
                else:
                    lo = max(0, i - 260)
                    hi = max(0, i - 10)
                    est = pd.DataFrame(index=dates[lo:hi])
                ar = _market_model_ar(r, m_all, est, cfg.min_est_obs)
                if ar is None:
                    est_fail += 1
                    continue

        row: dict = {"code": code, "event_date": str(dates[i].date())}
        ok_any = False
        for a, b in windows:
            col = window_label(a, b)
            if cfg.car_mode == "close_to_close":
                j0, j1 = i + a, i + b
                if j0 < 0 or j1 >= len(close):
                    row[col] = np.nan
                    continue
                val = (close.iloc[j1] / close.iloc[j0] - 1) * 100
            else:
                j0, j1 = i + a, i + b
                if j0 < 0 or j1 >= len(ar):
                    row[col] = np.nan
                    continue
                seg = ar.iloc[j0: j1 + 1]
                if seg.isna().any() or len(seg) < (b - a + 1):
                    row[col] = np.nan
                    continue
                val = float(seg.sum()) * 100
            row[col] = float(val) if np.isfinite(val) else np.nan
            ok_any = ok_any or np.isfinite(row[col])
        if not ok_any:
            short += 1
            continue
        rows.append(row)

    panel = pd.DataFrame(rows)
    flow = [
        {"step": "compute_car", "reason": "证券价格表总数", "n": n_total},
        {"step": "compute_car", "reason": "事件日不在交易区间", "n": no_anchor},
        {"step": "compute_car", "reason": "价格缺失/窗口越界", "n": short},
        {"step": "compute_car", "reason": "估计窗口不足", "n": est_fail},
        {"step": "compute_car", "reason": "进入面板", "n": len(panel)},
    ]
    notes = []
    if cfg.car_mode == "close_to_close":
        notes.append("[0,h] 表示从事件日收盘持有到 T+h 收盘，与 sum_daily 口径相差一日。")
    return PanelResult(panel, flow, notes)
