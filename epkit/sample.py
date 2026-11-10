"""建样本：一条路径，事件日可替换（安慰剂要换锚点重算）。

pipeline 与 placebo 共用这里，保证「同一个样本口径」不出现第二份实现。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import SampleFlow, load_cross_section, load_market, load_prices, read_table
from .vars import evaluate
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


def _norm_code(s: pd.Series) -> pd.Series:
    """代码列一律当字符串并按 6 位补零。

    pandas 读 CSV 会把 "000001" 解析成 1，前导零丢失后与行情/财务表 merge 一律落空，
    而下游只会报「没有观测」——真正的原因（代码格式）不会出现在任何提示里。
    """
    out = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    is_num = out.str.fullmatch(r"\d+")
    return out.where(~is_num, out.str.zfill(6))


def _apply_merges(df: pd.DataFrame, spec: StudySpec, flow: SampleFlow) -> pd.DataFrame:
    """按 spec.data.merge 依次并入外部表，每一步都进样本流。"""
    for m in spec.data.merge:
        path = Path(m.source)
        if not path.exists():
            raise FileNotFoundError(f"合并表不存在：{path}")
        right = read_table(path, encoding=spec.data.encoding, column_map=spec.data.column_map)
        for k in m.on:
            if k in right.columns and k in df.columns:
                right[k] = _norm_code(right[k])
                df[k] = _norm_code(df[k])
        missing_key = [c for c in m.on if c not in right.columns or c not in df.columns]
        if missing_key:
            raise KeyError(f"{path.name} 与主表的连接键对不上：{missing_key}"
                           f"（右表列：{list(right.columns)[:10]}；主表列：{list(df.columns)[:10]}）")

        keep = list(m.keep) or [c for c in right.columns if c not in m.on and c not in df.columns]
        lacking = [c for c in keep if c not in right.columns]
        if lacking:
            raise KeyError(f"{path.name} 缺列 {lacking}；实际列：{list(right.columns)[:12]}")
        right = right[list(m.on) + keep].copy()

        clash = [c for c in keep if c in df.columns and c not in m.on]
        if clash and not m.rename and not m.suffix:
            raise ValueError(f"并入的列与主表同名：{clash}；用 rename 改名或给 suffix")
        if m.rename:
            right = right.rename(columns={k: v for k, v in m.rename.items() if k in right.columns})

        n_before = len(df)
        try:
            merged = df.merge(right, on=m.on, how=m.how, suffixes=("", m.suffix),
                              validate=m.cardinality)
        except Exception as exc:
            raise ValueError(
                f"合并 {path.name} 失败（validate={m.cardinality}）：{exc}。"
                f"右表连接键重复行数："
                f"{int(right.duplicated(subset=m.on).sum())}；"
                f"主表重复行数：{int(df.duplicated(subset=m.on).sum())}") from exc

        if len(merged) == 0:
            left_keys = df[m.on[0]].astype(str).head(3).tolist()
            right_keys = right[m.on[0]].astype(str).head(3).tolist()
            raise ValueError(
                f"合并 {path.name} 后 0 行：连接键格式可能不一致"
                f"（主表示例 {left_keys} vs 右表示例 {right_keys}）。"
                "A 股常见原因是代码没补零或带交易所前缀")

        probe = [c for c in keep if c in merged.columns]
        hit = 1.0
        if m.how in ("left", "outer") and probe:
            hit = float(merged[probe[0]].notna().mean())
        elif m.how == "inner" and n_before:
            hit = len(merged) / n_before

        reason = f"{path.name}（{m.how}，匹配 {hit:.1%}）"
        if hit < m.min_match:
            reason += f"，低于 {m.min_match:.0%}：检查代码格式与年份口径"
        flow.record(f"merge_{path.stem}", reason, n_before, len(merged))
        flow.initial = len(merged)
        df = merged

    return df


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
        if spec.data.code_col in df.columns:
            df[spec.data.code_col] = _norm_code(df[spec.data.code_col])
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

    if spec.data.merge:
        df = _apply_merges(df, spec, flow)

    for d in spec.sample.derive:
        try:
            v = evaluate(d.expr, df)
        except Exception as exc:
            raise ValueError(f"派生变量 {d.name} 无法求值：{d.expr}（{exc}）") from exc
        if not isinstance(v, pd.Series):
            v = pd.Series(v, index=df.index)
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
            raise ValueError("values 型分组尚未实现，请用 expr")

    return df, flow
