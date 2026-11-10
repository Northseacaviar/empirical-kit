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

    p_fetch = sub.add_parser("fetch", help="用 baostock 拉日线/指数到本地 csv")
    p_fetch.add_argument("--codes", required=True, help="逗号分隔，如 600000,000001")
    p_fetch.add_argument("--start", required=True)
    p_fetch.add_argument("--end", required=True)
    p_fetch.add_argument("--out", required=True, help="股票目录或指数 csv 路径")
    p_fetch.add_argument("--adjust", default="2", choices=["1", "2", "3"],
                         help="1 后复权 / 2 前复权（默认） / 3 不复权")
    p_fetch.add_argument("--index", action="store_true", help="按指数抓，写成一张长表")
    p_fetch.add_argument("--force", action="store_true", help="已存在的文件也重拉")

    p_map = sub.add_parser("map-columns", help="把导出的表按内置映射改成标准列名")
    p_map.add_argument("--in", dest="src", required=True)
    p_map.add_argument("--out", dest="dst", required=True)
    p_map.add_argument("--mapping", default="csmar")

    p_val = sub.add_parser("validate", help="只校验 spec 与数据路径")
    p_val.add_argument("spec")

    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        from .data_sources import baostock as bsf

        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        if args.index:
            out = bsf.fetch_index(codes, args.start, args.end, args.out)
            print(f"指数长表：{out}")
        else:
            res = bsf.fetch(codes, args.start, args.end, args.out,
                            adjustflag=args.adjust, force=args.force)
            print(f"写入 {len(res['written'])} 只，跳过 {len(res['skipped'])} 只，"
                  f"失败 {len(res['failed'])} 只 → {res['out_dir']}")
            for c, why in res["failed"]:
                print(f"  ! {c}: {why}")
        return 0

    if args.cmd == "map-columns":
        from .data import read_table

        df = read_table(args.src, column_map=args.mapping)
        out = Path(args.dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in (".xlsx", ".xls"):
            df.to_excel(out, index=False)
        else:
            df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{out}：{len(df)} 行，{len(df.columns)} 列")
        return 0

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
