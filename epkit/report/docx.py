"""期刊三线表（docx）：把 findings 直接排成能贴进稿子的表。

为什么单独做：markdown 表贴进 Word 还要手工重排，而排版这一步最容易出错
（表头漏线、标准误错行、表注编号对不上）。这里一次生成三线表，
数字仍然从 findings 取，不手抄。

版式口径：
- 三线表：顶线与底线粗（1.5pt），表头下细线（0.75pt），无竖线与内部横线；
- 系数单元格里系数在上、括号标准误在下（同一格两段）；
- 表注写在表下方、小一号字，编号与正文的"表 N"对应由作者自己控制。
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from ..findings import Findings

CN_FONT = "宋体"
EN_FONT = "Times New Roman"


def _style_run(run, size: float = 9, bold: bool = False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = EN_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    run.font.color.rgb = RGBColor(0, 0, 0)


def _border_edge(tag: str, sz: int, val: str = "single"):
    el = OxmlElement(f"w:{tag}")
    el.set(qn("w:val"), val)
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")
    return el


def _three_line_borders(table):
    """整表只留顶线/底线，内部线全关；表头下那条线由单元格负责。"""
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    borders.append(_border_edge("top", 12))
    borders.append(_border_edge("bottom", 12))
    borders.append(_border_edge("left", 0, "none"))
    borders.append(_border_edge("right", 0, "none"))
    borders.append(_border_edge("insideH", 0, "none"))
    borders.append(_border_edge("insideV", 0, "none"))
    tblPr.append(borders)


def _header_underline(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    borders.append(_border_edge("bottom", 6))
    tcPr.append(borders)


def _vis(t: str) -> int:
    """视觉宽度：中文按两格算。Word 的 autofit 会把长文本挤成三行，只能自己算。"""
    return sum(1 if ord(c) < 128 else 2 for c in t)


def _column_widths(doc, headers: list[str], rows: list[list], min_cm: float = 1.6):
    """每列先给最小宽（保证 3 位数不被折断），剩下的按内容长度分配。"""
    sec = doc.sections[0]
    total_cm = (sec.page_width - sec.left_margin - sec.right_margin) / 360000
    n = len(headers)
    need = []
    for i, h in enumerate(headers):
        cells = [str(r[i]).split("\n") for r in rows if i < len(r)]
        vis = max([_vis(str(h))] + [_vis(x) for c in cells for x in c] + [1])
        need.append(max(vis - 6, 1))
    base = min(min_cm, total_cm / n)
    rest = max(total_cm - base * n, 0.0)
    ssum = sum(need)
    return [Cm(base + rest * (x / ssum)) for x in need]


def three_line_table(doc, headers: list[str], rows: list[list], note: str | None = None,
                     size: float = 9):
    n_col = max([len(headers)] + [len(r) for r in rows]) if rows else len(headers)
    headers = list(headers) + [""] * (n_col - len(headers))
    rows = [list(r) + [""] * (n_col - len(r)) for r in rows]
    table = doc.add_table(rows=1, cols=n_col)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _three_line_borders(table)

    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _style_run(p.add_run(str(h)), size=size, bold=True)
        _header_underline(hdr[i])

    widths = _column_widths(doc, headers, rows)
    # 同时写 w:gridCol：OOXML 的固定布局以 grid 为准，只写 tcW 时 WPS/LibreOffice
    # 会按等宽重排，长文本被挤成多行
    grid = table._tbl.tblGrid
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w.twips)))
    tblPr = table._tbl.tblPr
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:type"), "dxa")
    tw.set(qn("w:w"), str(int(sum(w.twips for w in widths))))
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblPr.append(tw)
    for row in table.rows:
        for i, c in enumerate(row.cells):
            c.width = widths[i]

    for r in rows:
        cells = table.add_row().cells
        for i, c in enumerate(cells):
            c.width = widths[i]
        for i, v in enumerate(r):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            parts = str(v).split("\n")
            _style_run(p.add_run(parts[0]), size=size)
            for extra in parts[1:]:
                p.add_run().add_break()
                _style_run(p.add_run(extra), size=size - 1)

    if note:
        p = doc.add_paragraph()
        _style_run(p.add_run(note), size=size - 1)
    else:
        doc.add_paragraph()
    return table


def _cell_coef(coef, se) -> str:
    if coef is None:
        return "-"
    return f"{coef:.4f}\n({se:.4f})" if se is not None else f"{coef:.4f}"


# —— 各节的数据提取（结构与 report.md 一致，但给 docx 用结构化数据）——

def _sec_sample_flow(F: Findings):
    if not F.sample_flow:
        return None
    rows = [[f["step"], f["reason"], f["n_before"], f["dropped"], f["n_after"]]
            for f in F.sample_flow]
    return ["步骤", "口径", "样本", "剔除", "剩余"], rows


def _sec_describe(F: Findings):
    items = [f for f in F.items if f.type == "describe"]
    if not items:
        return None
    rows = []
    for f in items:
        for it in f.data["vars"]:
            if it.get("n", 0) <= 1:
                continue
            rows.append([it["var"], it["n"], f"{it['mean']:.3f}", f"{it['sd']:.3f}",
                         f"{it['median']:.3f}", f"{it['p25']:.3f}", f"{it['p75']:.3f}",
                         f"{it['min']:.3f}", f"{it['max']:.3f}"])
    return ["变量", "N", "均值", "标准差", "中位数", "P25", "P75", "最小", "最大"], rows


def _sec_corr(F: Findings):
    items = [f for f in F.items if f.type == "corr"]
    if not items:
        return None
    f = items[0]
    vs, m = f.data["vars"], f.data["matrix"]
    rows = []
    for i, v in enumerate(vs):
        rows.append([v] + ["" if i == j else (f"{m[i][j]:.2f}" if m[i][j] is not None else "-")
                           for j in range(len(vs))])
    return [""] + vs, rows


def _sec_group_diff(F: Findings):
    items = [f for f in F.items if f.type == "group_diff"]
    if not items:
        return None
    rows = []
    for f in items:
        d = f.data
        g0, g1 = d["groups"]
        rows.append([d["var"], f"{g0['mean']:.3f}\n[{g0['n']}]", f"{g1['mean']:.3f}\n[{g1['n']}]",
                     f"{d['diff']:.3f}", f"{d['t_welch']:.2f}", f"{d['p_welch']:.4f}"])
    return ["变量", "组 0 均值\n[样本]", "组 1 均值\n[样本]", "差(1−0)", "t(Welch)", "p"], rows


def _sec_reg(F: Findings):
    out = []
    for f in F.items:
        if f.type != "reg":
            continue
        base = f.data["main_vcov"]
        focus = f.data["focus"]
        steps = [s for s in f.data["steps"] if "by_vcov" in s]
        if not steps:
            continue
        headers = ["设定", f"{focus} → {f.data['y']}", "N", "R²"]
        def _short(lb: str) -> str:
            return lb if _vis(lb) <= 46 else lb[:22] + "…"

        rows = [[_short(s["label"]),
                 _cell_coef(s["by_vcov"][base]["terms"].get(focus, {}).get("coef"),
                            s["by_vcov"][base]["terms"].get(focus, {}).get("se")),
                 s["n"], f"{s['r2']:.3f}" if s.get("r2") else "-"] for s in steps]
        out.append((f.id, headers, rows))
    return out


def _sec_did(F: Findings):
    out = []
    for f in F.items:
        if f.type != "did":
            continue
        d = f.data
        rows = [[("基准" if r["k"] == -1 else f"{r['k']:+d}"),
                 _cell_coef(r["coef"], r["se"]), f"{r['t']:.2f}", f"{r['p']:.3f}"]
                for r in d["event_time"]]
        note = f"窗口 [{d['window'][0]},{d['window'][1]}]，端点归并；基准期 k=−1；N={d['n']}。"
        sc = d.get("staggered_check")
        if sc and sc.get("aggregate"):
            note += (f" CS 汇总 ATT={sc['aggregate']['att']:.4f}"
                     f"（se={sc['aggregate']['se']:.4f}）。")
        out.append((f.id, ["事件时间", "系数\n(标准误)", "t", "p"], rows, note))
    return out


def _sec_multitest(F: Findings):
    items = [f for f in F.items if f.type == "multitest" and f.data.get("tests")]
    if not items:
        return None
    if len(items) > 1:
        rows = []
        for f in items:
            rows.extend([[t["label"], f"{t['p_raw']:.4f}", f"{t['p_adj']:.4f}",
                          "是" if t["sig_raw"] else "", "是" if t["sig_adj"] else ""]
                         for t in f.data["tests"]])
        return (["检验", "p（未校正）", "p（校正后）", "未校正显著", "校正后显著"], rows,
                "；".join(f"{f.id} 用 {f.data['method']}" for f in items))
    f = items[0]
    rows = [[t["label"], f"{t['p_raw']:.4f}", f"{t['p_adj']:.4f}",
             "是" if t["sig_raw"] else "", "是" if t["sig_adj"] else ""]
            for t in f.data["tests"]]
    note = f"{f.data['method']} 校正，α={f.data['alpha']}；族内 {len(rows)} 次检验。"
    return ["检验", "p（未校正）", "p（校正后）", "未校正显著", "校正后显著"], rows, note


def report_docx(F: Findings, path: str | Path) -> Path:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = EN_FONT
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    h = doc.add_paragraph()
    _style_run(h.add_run(F.spec_name), size=14, bold=True)
    p = doc.add_paragraph()
    _style_run(p.add_run(f"spec hash {F.spec_hash}；方法 {F.meta.get('method')}；"
                         f"窗口口径 {F.meta.get('car_mode')}；样本 {F.meta.get('n_final')}"), size=9)

    sec = _sec_sample_flow(F)
    if sec:
        _style_run(doc.add_paragraph().add_run("表 A1  样本筛选"), size=10, bold=True)
        three_line_table(doc, *sec, note="逐步剔除后的样本。")

    sec = _sec_describe(F)
    if sec:
        _style_run(doc.add_paragraph().add_run("表 A2  描述统计"), size=10, bold=True)
        three_line_table(doc, *sec, note="连续变量给均值、标准差与分位数。")

    sec = _sec_corr(F)
    if sec:
        _style_run(doc.add_paragraph().add_run("表 A3  相关系数"), size=10, bold=True)
        three_line_table(doc, *sec, note="Pearson 相关，下三角显示。")

    sec = _sec_group_diff(F)
    if sec:
        _style_run(doc.add_paragraph().add_run("表 1  组间差异"), size=10, bold=True)
        three_line_table(doc, *sec, note="括号内为样本量；t 用 Welch 口径。")

    regs = _sec_reg(F)
    for i, (fid, headers, rows) in enumerate(regs, start=1):
        base = next((x.data["main_vcov"] for x in F.items if x.id == fid), None)
        label = {"iid": "同方差", "hetero": "异方差稳健"}.get(base or "", "聚类稳健")
        _style_run(doc.add_paragraph().add_run(f"表 2{'' if len(regs) == 1 else i}  回归（{fid}）"),
                   size=10, bold=True)
        three_line_table(doc, headers, rows, note=f"括号内为{label}标准误。")

    dids = _sec_did(F)
    for i, (fid, headers, rows, note) in enumerate(dids, start=1):
        _style_run(doc.add_paragraph().add_run(f"表 3{'' if len(dids) == 1 else i}  事件研究（{fid}）"),
                   size=10, bold=True)
        three_line_table(doc, headers, rows, note=note)

    mt = _sec_multitest(F)
    if mt:
        _style_run(doc.add_paragraph().add_run("表 4  多重检验校正"), size=10, bold=True)
        three_line_table(doc, mt[0], mt[1], note=mt[2])

    warns = [f for f in F.items if f.status != "ok"]
    if warns:
        _style_run(doc.add_paragraph().add_run("需要处理"), size=10, bold=True)
        for f in warns:
            q = doc.add_paragraph()
            _style_run(q.add_run(f"[{f.status}] {f.id}：{f.why}"), size=9)
            for w in f.warnings:
                r = doc.add_paragraph()
                _style_run(r.add_run(f"· {w}"), size=9)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out
