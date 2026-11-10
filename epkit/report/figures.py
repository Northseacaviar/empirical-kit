"""出图：期限结构图与安慰剂散点图。

只画 findings 里已有结论，不重新估计——图上的数必须能在 report.md 里找到同源。
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def spec_curve(F: Findings, path: Path, title: str | None = None) -> Path | None:
    """规格曲线：每个设定一个点，按系数排序，画出中位数与 95% 区间。"""
    f = next((x for x in F.items if x.type == "spec_curve"), None)
    if f is None or not f.data.get("specs"):
        return None
    specs = sorted(f.data["specs"], key=lambda r: r["coef"])
    coef = np.array([r["coef"] for r in specs])
    se = np.array([r["se"] for r in specs])
    p = np.array([r["p"] for r in specs])
    xs = np.arange(len(specs))
    med = float(f.data["summary"]["median"])

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.axhline(med, color="#c0392b", lw=1.0, ls="-", label=f"中位数 {med:+.3f}")
    ax.errorbar(xs[p >= 0.05], coef[p >= 0.05], yerr=1.96 * se[p >= 0.05],
                fmt="o", ms=4, color="#9db9d0", ecolor="#d5e2ec", capsize=2,
                label="不显著")
    ax.errorbar(xs[p < 0.05], coef[p < 0.05], yerr=1.96 * se[p < 0.05],
                fmt="o", ms=4.5, color="#1f4e79", ecolor="#7fa6c8", capsize=2,
                label="5% 显著")
    ax.set_xlabel("设定（按系数排序）")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("系数")
    ax.set_title(title or f"{f.id}：规格曲线（{len(specs)} 个设定）", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2, lw=0.6, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def density_plot(F: Findings, path: Path, max_vars: int = 4) -> Path | None:
    """分布图：每个变量一张直方密度，标出均值与中位数。"""
    f = next((x for x in F.items if x.type == "density"), None)
    if f is None:
        return None
    items = [i for i in f.data["items"] if i.get("hist")][:max_vars]
    if not items:
        return None
    ncol = 2 if len(items) > 1 else 1
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.6 * nrow), dpi=150, squeeze=False)
    for k, it in enumerate(items):
        ax = axes[k // ncol][k % ncol]
        edges = np.array(it["hist"]["edges"])
        dens = np.array(it["hist"]["density"])
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, dens, width=np.diff(edges), color="#9db9d0",
               edgecolor="#1f4e79", lw=0.4, alpha=0.85)
        ax.axvline(it["mean"], color="#c0392b", lw=0.9, ls="-")
        ax.axvline(it["median"], color="#1f4e79", lw=0.9, ls=":")
        ax.set_title(f"{it['var']}（偏度 {it['skew']:.2f}，峰度 {it['kurt']:.1f}）", fontsize=9)
        ax.grid(alpha=0.2, lw=0.5, axis="y")
        ax.tick_params(labelsize=7)
    for k in range(len(items), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("分布：实线均值，点线中位数", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _palette(n: int, bw: bool = False):
    """配色。bw=True 走灰度 + 线型区分，供期刊黑白印刷。"""
    if not bw:
        base = ["#1f4e79", "#c0392b", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f"]
        return [base[i % len(base)] for i in range(n)], [None] * n
    grays = ["#000000", "#555555", "#888888", "#aaaaaa", "#cccccc", "#e6e6e6"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    return [grays[i % len(grays)] for i in range(n)], [styles[i % len(styles)] for i in range(n)]


def trend_plot(F: Findings, path: Path, bw: bool = False, title: str | None = None) -> Path | None:
    """时间趋势：每组一条线，带 95% 区间。"""
    f = next((x for x in F.items if x.type == "trend"), None)
    if f is None or not f.data.get("points"):
        return None
    pts = f.data["points"]
    groups = sorted({p["group"] for p in pts})
    times = sorted({p["time"] for p in pts}, key=lambda s: (len(s), s))
    xi = {t: i for i, t in enumerate(times)}
    colors, styles = _palette(max(len(groups), 1), bw)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    for gi, g in enumerate(groups):
        sel = sorted([p for p in pts if p["group"] == g], key=lambda p: xi[p["time"]])
        if not sel:
            continue
        xs = [xi[p["time"]] for p in sel]
        ys = [p["mean"] for p in sel]
        lo = [p["ci_low"] for p in sel]
        hi = [p["ci_high"] for p in sel]
        label = g or f.data["var"]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=colors[gi],
                ls=styles[gi] or "-", label=label)
        if all(v is not None for v in lo + hi):
            ax.fill_between(xs, lo, hi, color=colors[gi], alpha=0.15 if not bw else 0.08,
                            lw=0)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(f.data["by"])
    ax.set_ylabel(f.data["var"])
    ax.set_title(title or f"{f.id}：{f.data['var']} 的时间趋势（95% 区间）", fontsize=10)
    if len(groups) > 1:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2, lw=0.6, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def box_plot_fig(F: Findings, path: Path, bw: bool = False, title: str | None = None) -> Path | None:
    """分组箱线：用 ax.bxp 从统计量直接画（图上的数必须与报告同源）。

    手搓矩形在 IQR 窄时会画成一条竖线；bxp 会按坐标轴比例给箱体宽度，可读性好得多。
    """
    f = next((x for x in F.items if x.type == "box"), None)
    if f is None or not f.data.get("groups"):
        return None
    gs = [g for g in f.data["groups"] if g.get("median") is not None]
    if not gs:
        return None
    stats = [{"med": g["median"], "q1": g["q1"], "q3": g["q3"],
              "whislo": g["min"], "whishi": g["max"], "fliers": [],
              "label": str(g["group"])} for g in gs]
    colors, _ = _palette(len(gs), bw)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    bp = ax.bxp(stats, patch_artist=True, widths=0.45,
                medianprops={"color": "black", "lw": 2},
                whiskerprops={"color": "black", "lw": 0.9},
                capprops={"color": "black", "lw": 1.2},
                boxprops={"lw": 0.9, "edgecolor": "black"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75 if not bw else 0.55)
    means = [g["mean"] for g in gs if g.get("mean") is not None]
    if len(means) == len(gs):
        ax.plot(range(1, len(gs) + 1), means, ls="none", marker="D", ms=4,
                mfc="white", mec="black", mew=1.0, label="均值")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlabel(f.data["by"])
    ax.set_ylabel(f.data["var"])
    ax.set_title(title or f"{f.id}：{f.data['var']} 的分组分布", fontsize=10)
    ax.grid(alpha=0.2, lw=0.6, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def corr_heatmap(F: Findings, path: Path, bw: bool = False) -> Path | None:
    """相关系数热图：数据来自已有的 corr 结果，不重算。"""
    f = next((x for x in F.items if x.type == "corr"), None)
    if f is None or not f.data.get("matrix"):
        return None
    vs, mat = f.data["vars"], f.data["matrix"]
    n = len(vs)
    arr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if mat[i][j] is not None:
                arr[i, j] = mat[i][j]
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    cmap = "Greys" if bw else "RdBu_r"
    im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(vs, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(vs, fontsize=7)
    for i in range(n):
        for j in range(n):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(arr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{f.id}：{f.data.get('method', 'pearson')} 相关系数", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(F: Findings, out_dir: Path) -> list[Path]:
    made = []
    for fn, name in [(lambda p: term_structure(F, p), "term_structure.png"),
                     (lambda p: placebo_scatter(F, p), "placebo.png"),
                     (lambda p: event_study(F, p), "event_study.png"),
                     (lambda p: spec_curve(F, p), "spec_curve.png"),
                     (lambda p: density_plot(F, p), "density.png"),
                     (lambda p: trend_plot(F, p), "trend.png"),
                     (lambda p: box_plot_fig(F, p), "box.png"),
                     (lambda p: corr_heatmap(F, p), "corr_heatmap.png"),
                     (lambda p: trend_plot(F, p, bw=True), "trend_bw.png"),
                     (lambda p: box_plot_fig(F, p, bw=True), "box_bw.png"),
                     (lambda p: corr_heatmap(F, p, bw=True), "corr_heatmap_bw.png")]:
        try:
            p = fn(out_dir / name)
            if p:
                made.append(p)
        except Exception as exc:      # 图失败不影响主流程
            print(f"  ! 出图 {name} 失败：{type(exc).__name__}: {exc}")
    return made
