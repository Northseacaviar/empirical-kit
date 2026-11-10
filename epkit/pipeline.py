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
from .checks.multitest import collect_p, multitest
from .checks.confound import confound
from .checks.density import density
from .checks.pileup import pileup
from .checks.reliability import reliability
from .data_sources.csmar import audit_units
from .checks.quantile import quantile
from .checks.spec_curve import spec_curve
from .checks.subsample import subsample
from .checks.vif import vif
from .checks.placebo import time_placebo
from .estimate.did import did
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

    # 单位审计：小数/百分点混用差 100 倍且不报错，这一步就是防它
    unit_notes = audit_units(df, spec.data.scale, spec.data.unit)
    if unit_notes:
        F.add(Finding("unit_audit", "unit", "warn",
                      "收益率单位与 spec 的换算系数可能不一致", {"notes": unit_notes},
                      warnings=unit_notes))
    if verbose:
        print(f"[2/4] 样本 {flow.initial} → {len(df)}")

    # 多重检验校正依赖其他检验的结果，所以放在整轮之后跑，spec 里的书写顺序不影响。
    pending_multitest = []

    for t in spec.tests:
        if "multitest" in t:
            pending_multitest.append(t["multitest"])
            continue
        if "group_diff" in t:
            c = t["group_diff"]
            F.add(group_diff(df, c["var"], c["by"], fid=c.get("id"),
                             min_n=c.get("min_n", 30), winsor=c.get("winsor"),
                             drop_where=c.get("drop_where"),
                             scale=spec.scale_for(c["var"]), unit=spec.data.unit))
        elif "reg" in t:
            F.add(reg(df, t["reg"]))
        elif "did" in t:
            F.add(did(df, t["did"]))
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
        elif "confound" in t:
            cfg_c = dict(t["confound"])
            if "sample_codes" not in cfg_c and "code" in df.columns:
                cfg_c["sample_codes"] = df["code"].dropna().unique().tolist()
            F.add(confound(spec, cfg_c))
        elif "pileup" in t:
            F.add(pileup(df, t["pileup"]))
        elif "reliability" in t:
            F.add(reliability(df, t["reliability"]))
        elif "density" in t:
            F.add(density(df, t["density"]))
        elif "spec_curve" in t:
            F.add(spec_curve(df, t["spec_curve"]))
        elif "subsample" in t:
            F.add(subsample(df, t["subsample"]))
        elif "quantile" in t:
            F.add(quantile(df, t["quantile"]))
        elif "placebo" in t:
            F.add(time_placebo(spec, t["placebo"], prices=prices, xs=xs, market=market,
                               verbose=verbose))
        else:
            F.add(Finding(str(list(t))[:40], "unknown", "fail",
                          f"未知检验类型：{list(t)}；支持的：group_diff / reg / did / placebo / leave_one_out / mean_test / describe / corr / vif"))

    for cfg in pending_multitest:
        F.add(multitest(F, cfg))

    # 族规模提示：一跑十几次检验却没人提校正，是论文最常见的一处疏漏
    if not pending_multitest:
        n_p = len(collect_p(F))
        if n_p >= 5:
            F.add(Finding("multitest_hint", "multitest", "warn",
                          f"本轮共 {n_p} 个可校正的 p 值，没有任何 multitest 项",
                          {"n_tests": n_p},
                          warnings=["同一族里跑了多次检验：要么预注册主检验，"
                                    "要么加一条 multitest 给出校正后的结论"]))

    F.meta["elapsed_s"] = round(time.time() - t0, 2)
    F.meta["n_final"] = int(len(df))
    if verbose:
        s = F.summary()
        print(f"[3/4] 检验 {len(F.items)} 条：ok {s.get('ok',0)} / warn {s.get('warn',0)} / fail {s.get('fail',0)}")
    return F
