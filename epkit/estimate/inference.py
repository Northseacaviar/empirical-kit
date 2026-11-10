"""bootstrap 与置换推断：当解析标准误不可信时，直接给抽样分布。

为什么 wild cluster bootstrap 调 pyfixest 而不是自己写：
它的 wildboottest 用的就是 py-econometrics/wildboottest 那套实现，
与主回归是同一个模型对象、同一套固定效应与聚类；自己重写最容易在
impose_null（原假设下重抽）与聚类加权的细节上出错，而这两处正是它的价值所在。

pairs bootstrap 与置换检验 pyfixest 没有现成接口，这里自己实现：
- pairs：按聚类整块重抽，重估；
- permutation：打乱解释变量（行级或按聚类整体），得到 |β*| ≥ |β| 的比例。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding
from .reg import _formula, _vcov_specs

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def _main_vcov(clusters: list[str], cfg: dict) -> Any:
    if cfg.get("main_vcov"):
        for name, spec in _vcov_specs(clusters):
            if name == cfg["main_vcov"]:
                return spec
    for name, spec in reversed(_vcov_specs(clusters)):
        if name.startswith("cluster"):
            return spec
    return "hetero"


def inference(df: pd.DataFrame, cfg: dict[str, Any], registry: dict) -> Finding:
    fid = cfg.get("id", "inference")
    if pf is None:
        return Finding(fid, "inference", "fail", "未安装 pyfixest", {})
    src = cfg.get("from")
    rcfg = registry.get(src)
    if not rcfg:
        return Finding(fid, "inference", "fail",
                       f"找不到要重抽的回归 {src!r}；先在同一 spec 里写一条 reg，并在 from 里引用它的 id",
                       {"available": sorted(registry)})

    y = rcfg["y"]
    xs = rcfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    controls = rcfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = rcfg.get("fe") or []
    clusters = rcfg.get("cluster") or []
    focus = rcfg.get("focus") or (xs[0] if xs else None)
    if not focus:
        return Finding(fid, "inference", "fail", f"{src} 没有可检验的聚焦系数", {})

    fml = _formula(y, xs, controls, fe)
    vcov = _main_vcov(clusters, rcfg)
    n = int(cfg.get("n", 499))
    seed = int(cfg.get("seed", 42))
    kind = cfg.get("type", "wild_cluster")

    try:
        m = pf.feols(fml, data=df, vcov=vcov)
        td = m.tidy()
        analytic = {"coef": float(td.loc[focus, "Estimate"]),
                    "se": float(td.loc[focus, "Std. Error"]),
                    "t": float(td.loc[focus, "t value"]),
                    "p": float(td.loc[focus, "Pr(>|t|)"])}
    except Exception as exc:
        return Finding(fid, "inference", "fail",
                       f"重建 {src} 的模型失败：{type(exc).__name__}: {exc}", {"formula": fml})

    warnings = []
    out: dict[str, Any] = {"from": src, "type": kind, "n": n, "focus": focus,
                           "formula": fml, "analytic": analytic}

    if kind in ("wild_cluster", "iid"):
        if kind == "wild_cluster" and not clusters:
            return Finding(fid, "inference", "fail",
                           "wild_cluster 需要聚类变量：在 reg 里写 cluster: [...]", {})
        # numba 编译的 wildboottest 只吃数值聚类标签：字符串代码（000001）会报
        # "non-precise type array(pyobject)"。先整数编码，聚类结构不变。
        d_est, cl_int = df, None
        if kind == "wild_cluster":
            cl_int = "_cl_int"
            d_est = df.copy()
            d_est[cl_int] = pd.factorize(d_est[clusters[0]])[0]
        try:
            if kind == "wild_cluster":
                m = pf.feols(fml, data=d_est, vcov={"CRV1": cl_int})
            r = m.wildboottest(reps=n, param=focus,
                               cluster=cl_int if kind == "wild_cluster" else None,
                               seed=seed)
        except Exception as exc:
            return Finding(fid, "inference", "fail",
                           f"wild bootstrap 失败：{type(exc).__name__}: {exc}"
                           "（需要安装 wildboottest 包）", out)
        out["boot"] = {"t": float(r.get("t value", np.nan)),
                       "p": float(r.get("Pr(>|t|)", np.nan)),
                       "bootstrap_type": str(r.get("bootstrap_type", "")),
                       "impose_null": bool(r.get("impose_null", True))}
        gap = (abs(abs(analytic["t"]) - abs(out["boot"]["t"])) / abs(analytic["t"])
               if analytic["t"] else 0.0)
        if gap > 0.3:
            warnings.append(f"bootstrap t={out['boot']['t']:.2f} 与解析 t={analytic['t']:.2f} 相差 "
                            f"{gap:.0%}：聚类口径不稳（聚类数少或组内相关强），正文报 bootstrap")

    elif kind == "pairs":
        if not clusters:
            return Finding(fid, "inference", "fail", "pairs bootstrap 需要聚类变量", {})
        rng = np.random.default_rng(seed)
        groups = {u: g for u, g in df.groupby(clusters[0])}
        units = list(groups)
        betas = []
        for _ in range(n):
            pick = rng.choice(units, size=len(units), replace=True)
            dd = pd.concat([groups[u].assign(_b=i) for i, u in enumerate(pick)],
                           ignore_index=True).drop(columns=clusters, errors="ignore")
            dd = dd.rename(columns={"_b": clusters[0]})
            try:
                mm = pf.feols(fml, data=dd, vcov=vcov)
                betas.append(float(mm.tidy().loc[focus, "Estimate"]))
            except Exception:
                continue
        if len(betas) < 20:
            return Finding(fid, "inference", "fail",
                           f"pairs bootstrap 只有 {len(betas)} 次成功，结果不可用", out)
        betas = np.array(betas)
        lo, hi = np.percentile(betas, [2.5, 97.5])
        out["pairs"] = {"se": float(betas.std(ddof=1)), "ci_low": float(lo),
                        "ci_high": float(hi), "n_ok": int(len(betas)),
                        "share_wrong_sign": float((np.sign(betas) != np.sign(analytic["coef"])).mean())}
        gap = abs(out["pairs"]["se"] - analytic["se"]) / analytic["se"] if analytic["se"] else 0
        if gap > 0.3:
            warnings.append(f"pairs bootstrap SE={out['pairs']['se']:.4f} 与解析 SE={analytic['se']:.4f} "
                            f"相差 {gap:.0%}")

    else:
        return Finding(fid, "inference", "fail",
                       f"未知 type={kind!r}；支持 wild_cluster / iid / pairs", out)

    why = (f"{kind}（{n} 次）：t={analytic['t']:.2f} → "
           + (f"bootstrap t={out['boot']['t']:.2f}，p={out['boot']['p']:.4f}"
              if "boot" in out else
              f"bootstrap SE={out['pairs']['se']:.4f}，95% CI "
              f"[{out['pairs']['ci_low']:+.4f}, {out['pairs']['ci_high']:+.4f}]"))
    return Finding(fid, "inference", "warn" if warnings else "ok", why, out, warnings=warnings)


def permutation(df: pd.DataFrame, cfg: dict[str, Any], registry: dict) -> Finding:
    fid = cfg.get("id", "permutation")
    if pf is None:
        return Finding(fid, "permutation", "fail", "未安装 pyfixest", {})
    src = cfg.get("from")
    rcfg = registry.get(src)
    if not rcfg:
        return Finding(fid, "permutation", "fail",
                       f"找不到要置换的回归 {src!r}", {"available": sorted(registry)})

    y = rcfg["y"]
    xs = rcfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    controls = rcfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = rcfg.get("fe") or []
    clusters = rcfg.get("cluster") or []
    focus = rcfg.get("focus") or (xs[0] if xs else None)
    if not focus:
        return Finding(fid, "permutation", "fail", f"{src} 没有聚焦系数", {})

    fml = _formula(y, xs, controls, fe)
    vcov = _main_vcov(clusters, rcfg)
    n = int(cfg.get("n", 499))
    seed = int(cfg.get("seed", 42))
    scheme = cfg.get("scheme", "shuffle_x")
    rng = np.random.default_rng(seed)

    try:
        base = pf.feols(fml, data=df, vcov=vcov)
        b0 = float(base.tidy().loc[focus, "Estimate"])
    except Exception as exc:
        return Finding(fid, "permutation", "fail",
                       f"原模型估计失败：{type(exc).__name__}: {exc}", {"formula": fml})

    betas = []
    for _ in range(n):
        dd = df.copy()
        if scheme == "shuffle_cluster" and clusters:
            # 按个体整体打乱：保留个体内的时间结构（面板处理分配的置换应这样）
            units = dd[clusters[0]].dropna().unique()
            order = rng.permutation(len(units))
            mapping = dict(zip(units, units[order]))
            dd[focus] = dd[clusters[0]].map(mapping).map(
                dd.groupby(clusters[0])[focus].mean())
        else:
            dd[focus] = rng.permutation(dd[focus].values)
        try:
            mm = pf.feols(fml, data=dd, vcov=vcov)
            if focus in mm.tidy().index:
                betas.append(float(mm.tidy().loc[focus, "Estimate"]))
        except Exception:
            continue

    if len(betas) < 20:
        return Finding(fid, "permutation", "fail",
                       f"置换只有 {len(betas)} 次成功，结果不可用", {})
    betas = np.asarray(betas)
    p_emp = float((np.abs(betas) >= abs(b0)).mean())
    warnings, status = [], "ok"
    if p_emp > 0.1:
        status = "warn"
        warnings.append(f"置换检验 p={p_emp:.3f}：把处理随机重排后同样大小的系数很常见，"
                        "说明原效应可能来自抽样噪声而非处理")
    why = (f"{scheme} 置换 {len(betas)} 次：原系数 {b0:+.4f}，"
           f"置换分布 95% 区间 [{np.percentile(betas, 2.5):+.4f}, "
           f"{np.percentile(betas, 97.5):+.4f}]，经验 p={p_emp:.3f}")
    return Finding(fid, "permutation", status, why,
                   {"from": src, "scheme": scheme, "n_ok": int(len(betas)),
                    "coef": b0, "p_empirical": p_emp,
                    "perm_mean": float(betas.mean()), "perm_sd": float(betas.std(ddof=1))},
                   warnings=warnings)
