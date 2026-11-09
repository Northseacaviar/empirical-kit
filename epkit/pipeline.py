"""流水线：spec → 面板 → 样本 → 检验 → 产物。

顺序固定，任何一步的样本变化都进样本流表；任何检验都不许回头改样本。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .checks.group import group_diff
from .data import SampleFlow, load_cross_section, load_market, load_prices, read_table
from .estimate.event import build_event_panel
from .findings import Finding, Findings
from .spec import StudySpec


def _apply_filters(df: pd.DataFrame, spec: StudySpec, flow: SampleFlow) -> pd.DataFrame:
    for i, f in enumerate(spec.sample.filters, start=1):
        before = len(df)
        df2 = flow.apply(df, f, i)
        if len(df2) == before:
            flow.record(f.name or f"filter_{i}", (f.reason or "无剔除") + "（未命中）", before, len(df2))
        df = df2
    return df


def _apply_groups(df: pd.DataFrame, spec: StudySpec) -> list[str]:
    made = []
    for g in spec.sample.groups:
        if g.expr:
            try:
                df[g.name] = df.eval(g.expr).astype(int)
            except Exception as exc:
                raise ValueError(f"分组 {g.name} 的表达式无法求值：{g.expr}（{exc}）") from exc
            made.append(g.name)
        elif g.values:
            raise NotImplementedError("values 型分组尚未实现，请用 expr")
    return made


def run_spec(spec_path: str | Path, verbose: bool = True) -> Findings:
    t0 = time.time()
    spec = StudySpec.load(spec_path)
    F = Findings(spec.name, spec.hash())
    F.meta["spec_path"] = str(Path(spec_path).resolve())
    F.meta["car_mode"] = spec.returns.car_mode
    F.meta["method"] = spec.returns.method
    F.meta["windows"] = spec.returns.windows

    prices = load_prices(spec)
    if verbose:
        print(f"[1/5] 价格 {len(prices)} 只")
    market = load_market(spec)
    res = build_event_panel(spec, prices, market)
    panel = res.panel
    if verbose:
        print(f"[2/5] 事件面板 {len(panel)} 行，窗口 {len(spec.returns.windows)} 个")

    xs = load_cross_section(spec)
    flow = SampleFlow(initial=len(panel))
    if xs is not None:
        key = spec.data.code_col
        if key not in xs.columns:
            raise KeyError(f"横截面表缺少代码列 {key}")
        panel = panel.rename(columns={"code": key})
        panel[key] = panel[key].astype(str)
        xs[key] = xs[key].astype(str)
        merged = xs.merge(panel, on=key, how="inner", validate="one_to_one")
        flow.record("merge_cross_section", "并入样本框（内连接）", len(panel), len(merged))
        dup = xs[key].duplicated().sum()
        if dup:
            F.add(Finding("merge_dup", "data_audit", "warn",
                          f"样本框有 {int(dup)} 个重复代码，内连接可能丢行", {"dup_codes": int(dup)}))
        df = merged
    else:
        df = panel
        flow.record("no_cross_section", "未提供样本框，直接用价格面板", len(panel), len(panel))

    df = _apply_filters(df, spec, flow)
    _apply_groups(df, spec)
    F.sample_flow = flow.as_list()
    if verbose:
        print(f"[3/5] 样本 {flow.initial} → {len(df)}")

    for t in spec.tests:
        if "group_diff" in t:
            c = t["group_diff"]
            F.add(group_diff(df, c["var"], c["by"], fid=c.get("id"), min_n=c.get("min_n", 30)))
        else:
            F.add(Finding(str(list(t))[:40], "unknown", "fail",
                          f"未知检验类型：{list(t)}；支持的：group_diff"))

    F.meta["elapsed_s"] = round(time.time() - t0, 2)
    F.meta["n_final"] = int(len(df))
    if verbose:
        s = F.summary()
        print(f"[4/5] 检验 {len(F.items)} 条：ok {s.get('ok',0)} / warn {s.get('warn',0)} / fail {s.get('fail',0)}")
    return F
