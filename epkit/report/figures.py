"""出图：期限结构图与安慰剂散点图。

只画 findings 里已有结论，不重新估计——图上的数必须能在 report.md 里找到同源。
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..findings import Findings

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_WIN = re.compile(r"car_m(\d+)_(\d+)|car_(\d+)_m(\d+)|car_(\d+)_(\d+)")


def parse_window(var: str) -> tuple[int, int] | None:
    m = _WIN.fullmatch(var)
    if not m:
        return None
    g = m.groups()
    if g[0] is not None:
        return (-int(g[0]), int(g[1]))
    if g[2] is not None:
        return (int(g[2]), -int(g[3]))
    return (int(g[4]), int(g[5]))


def term_structure(F: Findings, path: Path, prefix: str = "", title: str | None = None) -> Path | None:
    """组间差的期限结构：横轴窗口右端，纵轴差，误差棒 95% CI（Welch）。"""
    pts = []
    for f in F.items:
        if f.type != "group_diff" or not f.id.startswith(prefix):
            continue
        if f.data.get("winsor") or f.data.get("drop_where"):
            continue                      # 稳健性变体不进主图
        w = parse_window(f.data["var"])
        if w is None:
            continue
        lo, hi = f.data["ci95_welch"]
        pts.append((w, f.data["diff"], lo, hi))
    if len(pts) < 2:
        return None
    pts.sort(key=lambda x: (x[0][1], x[0][0]))
    x = np.arange(len(pts))
    labels = [f"[{a},{b}]" for (a, b), *_ in pts]
    y = np.array([p[1] for p in pts])
    err = np.array([[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]])

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#1f4e79", ecolor="#9db9d0",
                capsize=3, lw=1.4, ms=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_xlabel("事件窗口")
    ax.set_ylabel(f"组间差（{F.meta.get('unit') or '原单位'}）")
    ax.set_title(title or f"{F.spec_name}：组间差期限结构", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def placebo_scatter(F: Findings, path: Path, title: str | None = None) -> Path | None:
    """安慰剂散点：锚点日期 vs 组间差，标出事件日与 ±2sd 区间。"""
    f = next((x for x in F.items if x.type == "placebo"), None)
    if f is None:
        return None
    rows = [r for r in f.data["anchors"] if r["diff"] is not None]
    if len(rows) < 3:
        return None
    d = f.data
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.axhspan(d["placebo_mean"] - 2 * d["placebo_sd"], d["placebo_mean"] + 2 * d["placebo_sd"],
               color="#d9e2ec", alpha=0.6, label="安慰剂均值 ±2sd")
    ax.axhline(d["placebo_mean"], color="#888", lw=0.8, ls="--")
    import pandas as pd

    xs = pd.to_datetime([r["anchor"] for r in rows])
    ys = [r["diff"] for r in rows]
    ev = pd.Timestamp(d["event_date"])
    ax.plot(xs, ys, "o", color="#888", ms=5, label="非事件锚点")
    ax.plot([ev], [d["event_diff"]], "D", color="#c0392b", ms=7, label="事件日")
    ax.annotate(f"事件日 {d['event_diff']:+.2f}（z={d['z']:.1f}）",
                xy=(ev, d["event_diff"]), xytext=(-10, 18), textcoords="offset points",
                color="#c0392b", fontsize=9, ha="right")
    ax.axvline(ev, color="#c0392b", lw=0.7, ls=":")
    ax.set_ylabel(f"组间差（{F.meta.get('unit') or '原单位'}）")
    ax.set_title(title or f"{F.spec_name}：安慰剂检验", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def event_study(F: Findings, path: Path, title: str | None = None) -> Path | None:
    """DID 事件研究图：各事件时间的系数与 95% 置信区间。"""
    f = next((x for x in F.items if x.type == "did"), None)
    if f is None:
        return None
    rows = f.data["event_time"]
    if len(rows) < 2:
        return None
    ks = [r["k"] for r in rows]
    coef = np.array([r["coef"] for r in rows])
    se = np.array([r["se"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.axvline(-0.5, color="#c0392b", lw=0.8, ls=":")   # 处理时点（基准 -1 与 0 之间）
    ax.errorbar(ks, coef, yerr=1.96 * se, fmt="o", color="#1f4e79", ecolor="#9db9d0",
                capsize=3, ms=5)
    ax.set_xticks(ks)
    ax.set_xlabel("相对处理时点（期）")
    ax.set_ylabel(f"处理效应（{F.meta.get('unit') or '原单位'}）")
    ax.set_title(title or f"{f.id}：DID 事件研究", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(F: Findings, out_dir: Path) -> list[Path]:
    made = []
    for fn, name in [(lambda p: term_structure(F, p), "term_structure.png"),
                     (lambda p: placebo_scatter(F, p), "placebo.png"),
                     (lambda p: event_study(F, p), "event_study.png")]:
        try:
            p = fn(out_dir / name)
            if p:
                made.append(p)
        except Exception as exc:      # 图失败不影响主流程
            print(f"  ! 出图 {name} 失败：{type(exc).__name__}: {exc}")
    return made
