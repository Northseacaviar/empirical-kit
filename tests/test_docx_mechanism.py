"""三线表 docx 的机制测试：线在不在、数字对不对、SE 有没有错行。

Word 版式没法截图断言，但三线表的三条线是 XML 属性，可以精确检查：
顶线/底线 12（1.5pt）、表头下 6（0.75pt）、无竖线与内部横线。
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

from epkit.findings import Finding, Findings
from epkit.report.docx import report_docx


def toy_findings() -> Findings:
    """一份最小的结果集：组间差异 + DID + 多重检验，够覆盖三类表。"""
    F = Findings("toy", "hash123")
    F.meta.update({"method": "raw", "car_mode": "sum_daily", "n_final": 352})
    F.sample_flow = [{"step": "初始", "reason": "", "n_before": 400, "dropped": 48, "n_after": 352}]
    F.add(Finding("g1", "group_diff", "ok", "", {
        "var": "car_0_5", "diff": 0.1234, "t_welch": 2.1, "p_welch": 0.036,
        "t_equal": 2.0, "groups": [{"mean": -0.05, "n": 176}, {"mean": 0.07, "n": 176}]}))
    F.add(Finding("d1", "did", "ok", "", {
        "y": "y", "window": [-3, 3], "n": 352,
        "event_time": [{"k": -3, "coef": -0.786, "se": 0.336, "t": -2.34, "p": 0.02},
                       {"k": 0, "coef": 0.4, "se": 0.2, "t": 2.0, "p": 0.046}],
        "staggered_check": None}))
    F.add(Finding("mt", "multitest", "ok", "", {
        "method": "bh", "alpha": 0.05,
        "tests": [{"label": "g1", "p_raw": 0.036, "p_adj": 0.036,
                   "sig_raw": True, "sig_adj": True}],
        "n_sig_raw": 1, "n_sig_adj": 1, "flipped": []}))
    return F


def _borders(table):
    tblPr = table._tbl.tblPr
    el = tblPr.find(qn("w:tblBorders"))
    out = {}
    if el is not None:
        for edge in el:
            out[edge.tag.split("}")[1]] = edge.get(qn("w:val")), edge.get(qn("w:sz"))
    return out


def test_docx_has_three_line_tables(tmp_path):
    F = toy_findings()
    p = report_docx(F, tmp_path / "report.docx")
    assert p.exists() and p.stat().st_size > 5000
    doc = Document(str(p))
    assert doc.tables, "没有任何表格"
    b = _borders(doc.tables[0])
    assert b["top"] == ("single", "12"), b
    assert b["bottom"] == ("single", "12"), b
    assert b["insideV"] == ("none", "0"), b
    assert b["insideH"] == ("none", "0"), b
    # 表头下细线挂在单元格上
    hdr_tc = doc.tables[0].rows[0].cells[0]._tc
    tcPr = hdr_tc.find(qn("w:tcPr"))
    tb = tcPr.find(qn("w:tcBorders"))
    assert tb is not None and tb.find(qn("w:bottom")).get(qn("w:sz")) == "6"


def test_docx_numbers_come_from_findings(tmp_path):
    F = toy_findings()
    p = report_docx(F, tmp_path / "report.docx")
    doc = Document(str(p))
    texts = []
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                texts.append(c.text)
    flat = "\n".join(texts)
    assert "-0.7860" in flat, flat[:400]     # DID k=-3 的系数
    assert "0.123" in flat                   # 组间差（3 位小数口径）
    assert "352" in flat                     # 样本量


def test_docx_survives_missing_sections(tmp_path):
    """只有一条结果的 findings 也要能出表，不能因为缺节报错。"""
    F = Findings("bare", "h")
    F.add(Finding("only", "group_diff", "ok", "", {
        "var": "x", "diff": 0.5, "t_welch": 2.0, "p_welch": 0.04, "t_equal": 1.9,
        "groups": [{"mean": 0.0, "n": 40}, {"mean": 0.5, "n": 45}]}))
    p = report_docx(F, tmp_path / "bare.docx")
    assert p.exists()
    doc = Document(str(p))
    assert len(doc.tables) == 1
