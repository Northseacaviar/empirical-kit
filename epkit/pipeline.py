"""流水线：spec → 样本 → 检验 → 产物。

顺序固定，任何一步的样本变化都进样本流表；任何检验都不许回头改样本。
"""
from __future__ import annotations

import time
from pathlib import Path

from .checks.group import group_diff
from .checks.desc import corr, describe
from .checks.leave_one import leave_one_out
from .checks.mean import mean_test
from .checks.vif import vif
from .checks.placebo import time_placebo
from .estimate.reg import reg
from .findings import Finding, Findings
from .sample import build_sample, load_inputs
from .spec import StudySpec


def run_spec(spec_path: str | Path, verbose: bool = True) -> Findings:
    t0 = time.time()
    spec = StudySpec.load(spec_path)
    F = Findings(spec.name, spec.hash())
    F.meta["spec_path"] = str(Path(spec_path).resolve())
    F.meta["car_mode"] = spec.returns.car_mode
    F.meta["method"] = spec.returns.method
    F.meta["windows"] = spec.returns.windows
    F.meta["scale"] = spec.data.scale
    F.meta["unit"] = spec.data.unit

    prices, xs, market = load_inputs(spec)
    df, flow = build_sample(spec, prices=prices, xs=xs, market=market)
    if verbose:
        n_price = len(prices) if prices is not None else None
        if n_price:
            print(f"[1/4] 价格 {n_price} 只，样本 {len(df)} 行")
        else:
            print(f"[1/4] 面板 {len(df)} 行（panel 模式）")

    F.sample_flow = flow.as_list()
    if verbose:
        print(f"[2/4] 样本 {flow.initial} → {len(df)}")

    for t in spec.tests:
        if "group_diff" in t:
            c = t["group_diff"]
            F.add(group_diff(df, c["var"], c["by"], fid=c.get("id"),
                             min_n=c.get("min_n", 30), winsor=c.get("winsor"),
                             drop_where=c.get("drop_where"),
                             scale=spec.scale_for(c["var"]), unit=spec.data.unit))
        elif "reg" in t:
            F.add(reg(df, t["reg"]))
        elif "describe" in t:
            c = t["describe"]
            F.add(describe(df, c["vars"], fid=c.get("id"), by=c.get("by"),
                           scale=spec.data.scale, scale_fn=spec.scale_for))
        elif "corr" in t:
            c = t["corr"]
            F.add(corr(df, c["vars"], fid=c.get("id"), method=c.get("method", "pearson"),
                       threshold=c.get("threshold", 0.8)))
        elif "vif" in t:
            c = t["vif"]
            F.add(vif(df, c["vars"], fid=c.get("id"), fe=c.get("fe"),
                      warn_at=c.get("warn_at", 10.0)))
        elif "mean_test" in t:
            c = t["mean_test"]
            F.add(mean_test(df, c["var"], fid=c.get("id"), min_n=c.get("min_n", 30),
                            by=c.get("by"), scale=spec.scale_for(c["var"]),
                            unit=spec.data.unit))
        elif "leave_one_out" in t:
            c = t["leave_one_out"]
            F.add(leave_one_out(df, c["var"], c["by"], fid=c.get("id")))
        elif "placebo" in t:
            F.add(time_placebo(spec, t["placebo"], prices=prices, xs=xs, market=market,
                               verbose=verbose))
        else:
            F.add(Finding(str(list(t))[:40], "unknown", "fail",
                          f"未知检验类型：{list(t)}；支持的：group_diff / reg / placebo / leave_one_out / mean_test / describe / corr / vif"))

    F.meta["elapsed_s"] = round(time.time() - t0, 2)
    F.meta["n_final"] = int(len(df))
    if verbose:
        s = F.summary()
        print(f"[3/4] 检验 {len(F.items)} 条：ok {s.get('ok',0)} / warn {s.get('warn',0)} / fail {s.get('fail',0)}")
    return F
