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
             "", "## 组间差异", "", group_diff_md(F),
             "", "## 回归", "", reg_md(F), ""]
    warns = [f for f in F.items if f.status != "ok"]
    if warns:
        parts += ["## 需要处理", ""]
        for f in warns:
            parts.append(f"- **{f.id}**（{f.status}）：{f.why}")
            for w in f.warnings:
                parts.append(f"  - {w}")
        parts.append("")
    return "\n".join(parts)
