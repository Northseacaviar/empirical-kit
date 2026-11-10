"""命令行：给 agent 用的唯一入口。

    epkit run <spec.yaml> [--out DIR] [--quiet]
    epkit validate <spec.yaml>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .pipeline import run_spec
from .report.docx import report_docx
from .report.figures import make_all
from .report.tables import report_md, summary_text
from .spec import StudySpec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="epkit", description="spec 驱动的实证研究工具")
    ap.add_argument("--version", action="version", version=f"epkit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="跑一个 spec")
    p_run.add_argument("spec")
    p_run.add_argument("--out", default=None, help="输出目录，默认取 spec.output_dir")
    p_run.add_argument("--quiet", action="store_true")
    p_run.add_argument("--no-docx", action="store_true", help="不生成三线表 docx")

    p_val = sub.add_parser("validate", help="只校验 spec 与数据路径")
    p_val.add_argument("spec")

    args = ap.parse_args(argv)

    if args.cmd == "validate":
        spec = StudySpec.load(args.spec)
        checks = []
        for label, p in [("prices", spec.data.prices), ("index", spec.data.index),
                         ("cross_section", spec.data.cross_section),
                         ("panel", spec.data.panel)]:
            if p:
                checks.append((label, p, Path(p).exists()))
        print(json.dumps({"name": spec.name, "hash": spec.hash(),
                          "paths": [{"what": a, "path": b, "exists": c} for a, b, c in checks]},
                         ensure_ascii=False, indent=2))
        return 0 if all(c for _, _, c in checks) else 1

    spec = StudySpec.load(args.spec)
    F = run_spec(args.spec, verbose=not args.quiet)
    out = Path(args.out or spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    F.save(out / "findings.json")
    (out / "report.md").write_text(report_md(F), encoding="utf-8")
    figs = make_all(F, out / "figures")
    docx_path = None
    if not args.no_docx:
        try:
            docx_path = report_docx(F, out / "report.docx")
        except Exception as exc:      # docx 失败不影响数字产出
            print(f"  ! 三线表 docx 生成失败：{type(exc).__name__}: {exc}")
    if not args.quiet:
        print(summary_text(F))
        print(f"\n产物：{out/'findings.json'}  {out/'report.md'}" + (f"  {docx_path}" if docx_path else ""))
        for p in figs:
            print(f"      图：{p}")
    return 0 if not any(f.status == "fail" for f in F.items) else 2


if __name__ == "__main__":
    sys.exit(main())
