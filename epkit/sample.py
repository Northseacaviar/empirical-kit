"""建样本：一条路径，事件日可替换（安慰剂要换锚点重算）。

pipeline 与 placebo 共用这里，保证「同一个样本口径」不出现第二份实现。
"""
from __future__ import annotations

import pandas as pd

from .data import SampleFlow, load_cross_section, load_market, load_prices, read_table
from .estimate.event import build_event_panel
from .spec import StudySpec


def load_inputs(spec: StudySpec) -> tuple[dict | None, pd.DataFrame | None, pd.Series | None]:
    """价格（字典）/ 横截面表 / 市场收益。panel 模式下价格为空。"""
    if spec.data.panel:
        return None, None, None
    prices = load_prices(spec)
    xs = load_cross_section(spec)
    market = load_market(spec)
    return prices, xs, market


def build_sample(spec: StudySpec, event_date: str | None = None,
                 prices: dict | None = None, xs: pd.DataFrame | None = None,
                 market: pd.Series | None = None) -> tuple[pd.DataFrame, SampleFlow]:
    """返回 (样本表, 样本流)。

    event_date 覆盖 spec 的事件日；panel 模式忽略该参数。
    """
    flow: SampleFlow

    if spec.data.panel:
        df = read_table(spec.data.panel)
        flow = SampleFlow(initial=len(df))
        flow.record("load_panel", "读入已有分析面板（跳过 CAR 计算）", len(df), len(df))
    else:
        if prices is None:
            prices = load_prices(spec)
        if market is None and spec.returns.method != "raw":
            market = load_market(spec)
        if event_date:
            spec = spec.model_copy(deep=True)
            spec.event.date = event_date
        res = build_event_panel(spec, prices, market)
        df = res.panel
        flow = SampleFlow(initial=len(df))

    if xs is not None:
        key = spec.data.code_col
        if key not in xs.columns:
            raise KeyError(f"横截面表缺少代码列 {key}")
        df = df.rename(columns={"code": key})
        df[key] = df[key].astype(str)
        x = xs.copy()
        x[key] = x[key].astype(str)
        df = x.merge(df, on=key, how="inner", validate="one_to_one")
        flow.record("merge_cross_section", "并入样本框（内连接）", flow.initial, len(df))
        flow.initial = len(df)

    for d in spec.sample.derive:
        try:
            v = df.eval(d.expr)
        except Exception as exc:
            raise ValueError(f"派生变量 {d.name} 无法求值：{d.expr}（{exc}）") from exc
        if getattr(v, "dtype", None) == bool:
            v = v.astype(int)
        df[d.name] = v

    for i, f in enumerate(spec.sample.filters, start=1):
        df = flow.apply(df, f, i)

    for g in spec.sample.groups:
        if g.expr:
            try:
                df[g.name] = df.eval(g.expr).astype(int)
            except Exception as exc:
                raise ValueError(f"分组 {g.name} 的表达式无法求值：{g.expr}（{exc}）") from exc
        elif g.values:
            raise NotImplementedError("values 型分组尚未实现，请用 expr")

    return df, flow
