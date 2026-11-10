"""出表：Markdown 与纯文本。数字一律从 findings 里取，不手抄。"""
from __future__ import annotations

from ..findings import Findings


def md_table(headers: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def sample_flow_md(flow: list[dict]) -> str:
    rows = [[f["step"], f["reason"], f["n_before"], f["dropped"], f["n_after"]] for f in flow]
    return md_table(["步骤", "口径", "样本", "剔除", "剩余"], rows)


def group_diff_md(F: Findings, fid_prefix: str = "") -> str:
    items = [f for f in F.items if f.type == "group_diff" and f.id.startswith(fid_prefix)]
    if not items:
        return "_没有 group_diff 结果_"
    rows = []
    for f in items:
        d = f.data
        g0, g1 = d["groups"]
        rows.append([d["var"], f"{g0['mean']:.3f} [{g0['n']}]", f"{g1['mean']:.3f} [{g1['n']}]",
                     f"{d['diff']:.3f}", f"{d['t_welch']:.2f}", f"{d['p_welch']:.4f}",
                     f"{d['t_equal']:.2f}", f.status])
    return md_table(["变量", "组0 均值 [n]", "组1 均值 [n]", "差(组1-组0)", "t(Welch)", "p", "t(等方差)", "状态"], rows)


def reg_md(F: Findings, fid_prefix: str = "") -> str:
    """回归阶梯 + 聚类口径敏感性两张表。"""
    items = [f for f in F.items if f.type == "reg" and f.id.startswith(fid_prefix)]
    if not items:
        return "_没有 reg 结果_"
    out = []
    for f in items:
        focus = f.data.get("focus")
        steps = [s for s in f.data["steps"] if "by_vcov" in s]
        if not steps:
            out.append(f"_{f.id}：全部设定估计失败_")
            continue
        base = f.data["main_vcov"]
        out.append(f"**{f.id}**：{focus} → {f.data['y']}，主口径 `{base}`")
        out.append("")
        rows = []
        for st in steps:
            r = st["by_vcov"][base]["terms"].get(focus, {})
            rows.append([st["label"], f"{r.get('coef',float('nan')):.4f}",
                         f"({r.get('se',float('nan')):.4f})", f"{r.get('t',float('nan')):.2f}",
                         st["n"], f"{st['r2']:.3f}" if st.get("r2") else "-"])
        out.append(md_table(["设定", "系数", "标准误", "t", "N", "R²"], rows))
        out.append("")
        # 口径敏感性：每个口径的 t 值
        vnames = list(steps[0]["by_vcov"].keys())
        rows2 = []
        for st in steps:
            row = [st["label"]]
            for vn in vnames:
                t = st["by_vcov"][vn]["terms"].get(focus, {}).get("t")
                row.append(f"{t:.2f}" if t is not None else "-")
            rows2.append(row)
        out.append(md_table(["设定"] + vnames, rows2))
        out.append("")
    return "\n".join(out)


def describe_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "describe"]
    out = []
    for f in items:
        if f.data.get("by"):
            rows = []
            for it in f.data["vars"]:
                for g in it["groups"]:
                    rows.append([it["var"], g["label"], g["n"], f"{g['mean']:.3f}",
                                 f"{g['sd']:.3f}", f"{g['median']:.3f}",
                                 f"{g['p25']:.3f}", f"{g['p75']:.3f}"])
            out.append(md_table(["变量", "组", "N", "均值", "标准差", "中位数", "P25", "P75"], rows))
        else:
            rows = [[it["var"], it["n"], f"{it['mean']:.3f}", f"{it['sd']:.3f}",
                     f"{it['median']:.3f}", f"{it['p25']:.3f}", f"{it['p75']:.3f}",
                     f"{it['min']:.3f}", f"{it['max']:.3f}"] for it in f.data["vars"] if it.get("n", 0) > 1]
            out.append(md_table(["变量", "N", "均值", "标准差", "中位数", "P25", "P75", "最小", "最大"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有描述统计_"


def corr_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "corr"]
    out = []
    for f in items:
        vs = f.data["vars"]
        m = f.data["matrix"]
        rows = []
        for i, v in enumerate(vs):
            rows.append([v] + ["" if i == j else f"{m[i][j]:.2f}" if m[i][j] is not None else "-"
                               for j in range(len(vs))])
        out.append(f"**{f.id}**（{f.data['method']}，{f.data['n_complete']} 行完整）")
        out.append("")
        out.append(md_table([""] + vs, rows))
        out.append("")
    return "\n".join(out) if out else "_没有相关矩阵_"


def vif_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "vif"]
    out = []
    for f in items:
        rows = [[i["var"], f"{i['vif']:.2f}" if i.get("vif") is not None else "-",
                 f"{i['r2']:.3f}" if i.get("r2") is not None else "-"] for i in f.data["items"]]
        out.append(f"**{f.id}**（{'组内去均值：' + '/'.join(f.data['fe']) if f.data['fe'] else '未吸收固定效应'}，N={f.data['n']}）")
        out.append("")
        out.append(md_table(["变量", "VIF", "R²"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有共线性诊断_"


def did_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "did"]
    out = []
    for f in items:
        d = f.data
        rows = [[("基准" if r["k"] == -1 else f"{r['k']:+d}"), f"{r['coef']:+.4f}",
                 f"({r['se']:.4f})", f"{r['t']:.2f}", f"{r['p']:.3f}"] for r in d["event_time"]]
        out.append(f"**{f.id}**：{f.data['y']}，窗口 [{d['window'][0]},{d['window'][1]}]"
                   f"（端点归并），基准 k=-1，N={d['n']}")
        out.append("")
        out.append(md_table(["事件时间", "系数", "标准误", "t", "p"], rows))
        out.append("")
        if d.get("pre_trend_wald"):
            wt = d["pre_trend_wald"]
            out.append(f"预趋势联合检验：χ²={wt['stat']:.2f}，df={wt['df']}，p={wt['p']:.4f}"
                       + ("（未通过）" if wt["p"] < 0.05 else "（通过）"))
            out.append("")
        if d.get("staggered"):
            out.append("> 交错处理：TWFE 动态效应存在负权重，需 CS / did2s / Sun-Abraham 复核。")
            out.append("")
    return "\n".join(out) if out else "_没有 DID 结果_"


def robustness_md(F: Findings) -> str:
    """安慰剂与留一法。"""
    out = []
    for f in F.items:
        if f.type == "placebo":
            rows = [[r["anchor"], f"{r['diff']:+.3f}" if r["diff"] is not None else "-",
                     f"{r['t']:.2f}" if r["t"] is not None else "-"] for r in f.data["anchors"]]
            out.append(f"**{f.id}**：{f.data['var']}，事件日 {f.data['event_diff']:+.3f}，"
                       f"安慰剂均值 {f.data['placebo_mean']:+.3f}（sd {f.data['placebo_sd']:.3f}），"
                       f"z={f.data['z']:.2f}，经验 p={f.data['p_empirical']:.3f}")
            out.append("")
            out.append(md_table(["锚点日期", "组间差", "t"], rows))
            out.append("")
        elif f.type == "leave_one_out":
            out.append(f"**{f.id}**：原差 {f.data['diff']:+.3f}；最敏感单点使差变为 "
                       f"{f.data['diff_without_extreme']:+.3f}（移动 {f.data['max_shift']:+.3f}）")
            out.append("")
    return "\n".join(out) if out else "_没有稳健性检验_"


def mediation_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "mediation"]
    out = []
    for f in items:
        for r in f.data.get("rows", []):
            out.append(f"**{f.id}**：{f.data['x']} → {f.data['y']}，中介变量 {r['m']}"
                       f"（bootstrap {r['n_boot_ok']} 次）")
            out.append("")
            rows = [
                ["a：x → m", f"{r['a']['coef']:+.4f}", f"({r['a']['se']:.4f})", f"{r['a']['p']:.3f}"],
                ["b：m → y|x", f"{r['b']['coef']:+.4f}", f"({r['b']['se']:.4f})", f"{r['b']['p']:.3f}"],
                ["c：总效应", f"{r['c']['coef']:+.4f}", f"({r['c']['se']:.4f})", f"{r['c']['p']:.3f}"],
                ["c′：直接效应", f"{r['c_prime']['coef']:+.4f}", f"({r['c_prime']['se']:.4f})",
                 f"{r['c_prime']['p']:.3f}"],
                ["ab：间接效应", f"{r['ab']:+.4f}", f"({r['ab_se']:.4f})",
                 f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"],
            ]
            out.append(md_table(["路径（Baron-Kenny 三步）", "系数", "标准误", "p / 区间"], rows))
            out.append("")
            out.append(f"中介比例 ab/c = {r['share']:.1%}")
            out.append("")
    return "\n".join(out) if out else "_没有中介效应结果_"


def group_test_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "group_test"]
    out = []
    for f in items:
        d = f.data
        if not d.get("groups") or d.get("coef_low") is None:
            out.append(f"_{f.id}：{f.why}_")
            out.append("")
            continue
        rows = [[f"{d['by']}={d['groups'][0]}", f"{d['coef_low']:+.4f}", "-", "-"],
                [f"{d['by']}={d['groups'][1]}", f"{d['coef_high']:+.4f}", "-", "-"],
                ["差（交互项）", f"{d['diff']:+.4f}", f"({d['se_diff']:.4f})",
                 f"F={d['f']:.2f}, p={d['p_f']:.4f}"]]
        out.append(f"**{f.id}**：{d['x']} → {d['y']} 按 {d['by']} 分组（N={d['n']}）")
        out.append("")
        out.append(md_table(["分组", "系数", "标准误", "差异检验"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有跨组系数检验_"


def tobit_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "tobit"]
    out = []
    for f in items:
        d = f.data
        if not d.get("rows"):
            out.append(f"_{f.id}：{f.why}_")
            out.append("")
            continue
        rows = [[r["var"], f"{r['coef']:+.4f}", f"({r['se']:.4f})",
                 f"{r['z']:.2f}", f"{r['p']:.3f}"] for r in d["rows"]]
        out.append(f"**{f.id}**：{d['y']} 的 Tobit，下界 {d['lower']}，上界 {d['upper']}，"
                   f"删失 {d['n_left']} 左 / {d['n_right']} 右，N={d['n']}，σ={d['sigma']:.4f}")
        out.append("")
        out.append(md_table(["变量", "系数", "标准误", "z", "p"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有 Tobit 结果_"


def spec_curve_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "spec_curve"]
    out = []
    for f in items:
        d = f.data
        s = d["summary"]
        out.append(f"**{f.id}**：{s['n_specs']} 个设定（失败 {s['n_failed']}），"
                   f"中位系数 {s['median']:+.4f}，P10~P90 {s['p10']:+.4f} ~ {s['p90']:+.4f}，"
                   f"显著比例 {s['share_sig']:.0%}")
        out.append("")
        specs = sorted(d["specs"], key=lambda r: r["coef"])
        rows = []
        for r in specs:
            label = ("+".join(r["controls"]) or "无控制") + " | " + \
                    ("+".join(r["fe"]) or "无FE") + " | " + str(r["vcov"])
            rows.append([label, f"{r['coef']:+.4f}", f"({r['se']:.4f})",
                         f"{r['t']:.2f}", f"{r['p']:.3f}", r["n"]])
        out.append(md_table(["控制变量 | 固定效应 | 推断", "系数", "标准误", "t", "p", "N"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有规格曲线_"


def subsample_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "subsample"]
    out = []
    for f in items:
        rows = []
        for r in f.data["rows"]:
            if r.get("coef") is None:
                rows.append([r["group"], r["n"], "-", "-", "-", "-"])
            else:
                rows.append([r["group"], r["n"], f"{r['coef']:+.4f}", f"({r['se']:.4f})",
                             f"{r['t']:.2f}", f"{r['p']:.3f}"])
        out.append(f"**{f.id}**：按 {f.data['by']} 分组，{f.data['x']} → {f.data['y']}")
        out.append("")
        out.append(md_table(["分组", "N", "系数", "标准误", "t", "p"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有子样本网格_"


def quantile_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "quantile"]
    out = []
    for f in items:
        rows = [[f"{r['q']:.2f}", f"{r['coef']:+.4f}", f"({r['se']:.4f})",
                 f"{r['t']:.2f}", f"{r['p']:.3f}"] for r in f.data["rows"]]
        note = "（固定效应为组内去均值近似）" if f.data.get("fe") else ""
        out.append(f"**{f.id}**：{f.data['x']} → {f.data['y']} 的分位回归{note}")
        out.append("")
        out.append(md_table(["分位", "系数", "标准误", "t", "p"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有分位回归_"


def density_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "density"]
    out = []
    for f in items:
        rows = [[i["var"], i["n"], f"{i['skew']:.2f}", f"{i['kurt']:.2f}",
                 f"{i['p01']:.3f}", f"{i['p99']:.3f}", f"{i.get('winsor_gap', 0):.3f}"]
                for i in f.data["items"] if i.get("skew") is not None]
        if not rows:
            continue
        out.append(f"**{f.id}**：分布诊断（|偏度|>2 或超额峰度>7 判严重偏态）")
        out.append("")
        out.append(md_table(["变量", "N", "偏度", "超额峰度", "P1", "P99", "缩尾前后均值差/SD"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有分布诊断_"


def confound_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "confound"]
    out = []
    for f in items:
        rows = [[e.get("date", "-"), e.get("label", ""),
                 "、".join(e.get("hits") or []) or "无重叠"]
                for e in (f.data.get("events") or [])]
        if not rows:
            out.append(f"_{f.id}：{f.why}_")
            out.append("")
            continue
        out.append(f"**{f.id}**：主事件 {f.data.get('main_event')}，窗口 "
                   f"{f.data.get('windows')}")
        out.append("")
        out.append(md_table(["日期", "混杂事件", "落在哪几个窗口"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有混杂事件筛查_"


def pileup_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "pileup"]
    out = []
    for f in items:
        d = f.data
        rows = [[f"{b['center']:.4f}", b["count"]] for b in (d.get("bins") or [])]
        if d.get("theta") is None:
            out.append(f"_{f.id}：{f.why}_")
            out.append("")
            continue
        out.append(f"**{f.id}**：{d.get('var')} 在阈值 {d.get('cut')} 附近的密度"
                   f"（带宽 ±{d.get('bw')}，N={d.get('n_near')}）")
        out.append("")
        out.append(f"θ={d['theta']:+.3f}，z={d['z']:.2f}，p={d['p']:.3f}"
                   f"（左侧 {d['f_left']:.1f}/箱，右侧 {d['f_right']:.1f}/箱）")
        out.append("")
        out.append(md_table(["箱中心", "频数"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有阈值堆积检验_"


def reliability_md(F: Findings) -> str:
    items = [f for f in F.items if f.type == "reliability"]
    rows = []
    for f in items:
        d = f.data
        rows.append([f.id, f"{d['alpha']:.3f}" if d["alpha"] == d["alpha"] else "-",
                     f"{d['agreement']:.1%}" if d.get("agreement") is not None else "-",
                     "/".join(d["coders"]), f"{d['n_complete']}/{d['n_units']}", f.status])
    return (md_table(["编码项", "Krippendorff α", "完全一致率", "编码者", "完整单元", "状态"], rows)
            if rows else "_没有双编码信度_")


def multitest_md(F: Findings) -> str:
    """多重检验校正：只列显著性与校正后变化明显的行。"""
    items = [f for f in F.items if f.type == "multitest" and f.id != "multitest_hint"]
    out = []
    for f in items:
        d = f.data
        tests = d.get("tests") or []
        if not tests:
            out.append(f"_{f.id}：{f.why}_")
            out.append("")
            continue
        rows = [[t["label"], f"{t['p_raw']:.4f}", f"{t['p_adj']:.4f}",
                 "显著" if t["sig_raw"] else "", "显著" if t["sig_adj"] else ""]
                for t in tests]
        out.append(f"**{f.id}**：{d['method']} 校正，α={d['alpha']}，"
                   f"未校正显著 {d['n_sig_raw']} / 校正后显著 {d['n_sig_adj']}（共 {len(tests)} 次检验）")
        out.append("")
        out.append(md_table(["检验", "p（未校正）", "p（校正后）", "未校正", "校正后"], rows))
        out.append("")
    return "\n".join(out) if out else "_没有多重检验校正_"


def summary_text(F: Findings) -> str:
    lines = []
    s = F.summary()
    lines.append(f"spec={F.spec_name} hash={F.spec_hash} 耗时={F.meta.get('elapsed_s')}s")
    lines.append(f"样本 {F.meta.get('n_final')}；检验 ok={s.get('ok',0)} warn={s.get('warn',0)} fail={s.get('fail',0)}")
    for f in F.items:
        lines.append(f"  [{f.status:4s}] {f.id}: {f.why}")
        for w in f.warnings:
            lines.append(f"         ! {w}")
    return "\n".join(lines)


def report_md(F: Findings) -> str:
    parts = [f"# {F.spec_name}", "", f"spec hash `{F.spec_hash}`；方法 `{F.meta.get('method')}`；"
             f"窗口口径 `{F.meta.get('car_mode')}`", "", "## 样本流", "", sample_flow_md(F.sample_flow),
             "", "## 描述统计", "", describe_md(F),
             "", "## 相关矩阵", "", corr_md(F),
             "", "## 共线性", "", vif_md(F),
             "", "## 组间差异", "", group_diff_md(F),
             "", "## 回归", "", reg_md(F),
             "", "## DID", "", did_md(F),
             "", "## 分布诊断", "", density_md(F),
             "", "## 混杂事件", "", confound_md(F),
             "", "## 阈值堆积", "", pileup_md(F),
             "", "## 编码信度", "", reliability_md(F),
             "", "## 规格曲线", "", spec_curve_md(F),
             "", "## 子样本", "", subsample_md(F),
             "", "## 分位回归", "", quantile_md(F),
             "", "## Tobit", "", tobit_md(F),
             "", "## 中介效应", "", mediation_md(F),
             "", "## 跨组系数差异", "", group_test_md(F),
             "", "## 多重检验校正", "", multitest_md(F),
             "", "## 稳健性", "", robustness_md(F), ""]
    warns = [f for f in F.items if f.status != "ok"]
    if warns:
        parts += ["## 需要处理", ""]
        for f in warns:
            parts.append(f"- **{f.id}**（{f.status}）：{f.why}")
            for w in f.warnings:
                parts.append(f"  - {w}")
        parts.append("")
    return "\n".join(parts)
