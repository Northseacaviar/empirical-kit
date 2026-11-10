"""回归层：OLS / 固定效应 / IV，一次给出全部推断口径。

为什么必须一次给全：同一份数据换聚类口径，t 值能从 2.3 掉到 1.6，
结论的严格程度就变了。论文里手写三个 V 矩阵再拼双向，是最容易出错的一段。

用法见 spec 的 tests[].reg；本模块不接触样本，只接收已建好的横截面表。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..findings import Finding
from .iv import iv_diagnostics, iv_warnings

try:
    import pyfixest as pf
except ImportError:  # pragma: no cover
    pf = None


def _formula(y: str, xs: list[str], controls: list[str], fe: list[str],
             iv: dict[str, Any] | None = None) -> str:
    # pyfixest 0.70 起 IV 要用方括号内联写法，旧的 `y ~ x | fe | x ~ z` 已弃用。
    parts = list(xs) + list(controls)
    if iv:
        endog = " + ".join(iv["endog"])
        inst = " + ".join(iv["instruments"])
        parts.append(f"[{endog} ~ {inst}]")
    right = " + ".join(parts)
    f = f"{y} ~ 1 + {right}" if right else f"{y} ~ 1"
    if fe:
        f += " | " + " + ".join(fe)
    return f


GLM_FAMILIES = {"logit", "probit", "poisson"}
OLS_MODELS = {"ols", "gaussian", ""}


def _fit(fml: str, df, vspec, model: str):
    """按 model 选估计器。离散选择走 pyfixest 的 feglm/fepois，
    这样高维固定效应与聚类标准误跟线性模型是同一套代码路径。"""
    if model in OLS_MODELS:
        return pf.feols(fml, data=df, vcov=vspec)
    if model == "poisson":
        return pf.fepois(fml, data=df, vcov=vspec)
    if model in ("logit", "probit"):
        return pf.feglm(fml, data=df, family=model, vcov=vspec)
    raise ValueError(f"未知 model={model!r}；支持 ols / logit / probit / poisson（nb 与 tobit 走单独路径）")


def _avg_marginal(model: str, coef: dict, m) -> dict:
    """平均边际效应：离散模型里系数不是边际效应，论文要报 dy/dx。

    口径：样本平均（APE）。logit 用 mean(p(1-p))、probit 用 mean(φ(xb))、
    poisson 用 mean(μ)。标准误按 delta 法近似（把权重当常数），
    与 statsmodels get_margeff 的差别只在小样本明显。

    注意：pyfixest 的 predict() 返回的是**线性预测**（xb 或 log μ），不是概率，
    直接当概率用会把边际效应算成负数。
    """
    from scipy import stats as sps

    try:
        eta = np.asarray(m.predict(), dtype=float)
    except Exception:
        return {}
    if model == "logit":
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = float(np.mean(mu * (1 - mu)))
    elif model == "probit":
        w = float(np.mean(sps.norm.pdf(eta)))
    elif model == "poisson":
        w = float(np.mean(np.exp(eta)))
    else:
        return {}
    return {k: {"dy_dx": v["coef"] * w, "se_approx": abs(v["se"] * w),
                "method": "APE（样本平均），SE 为 delta 近似"}
            for k, v in coef.items()}


def _pseudo_r2(m, y: np.ndarray, model: str, n: int) -> float | None:
    """McFadden 伪 R²。pyfixest 的 fitstat.loglik 在 0.70 里是 nan，只能自己算。"""
    from scipy import stats as sps

    try:
        eta = np.asarray(m.predict(), dtype=float)
        yy = np.asarray(y, dtype=float)
        if len(eta) != len(yy):
            return None
        if model == "logit":
            mu = 1.0 / (1.0 + np.exp(-eta))
        elif model == "probit":
            mu = sps.norm.cdf(eta)
        else:
            mu = np.exp(eta)
        eps = 1e-12
        if model == "poisson":
            llf = float(np.sum(yy * np.log(mu + eps) - mu))
            p0 = max(float(yy.mean()), eps)
            llnull = float(np.sum(yy * np.log(p0) - p0))
        else:
            mu = np.clip(mu, eps, 1 - eps)
            llf = float(np.sum(yy * np.log(mu) + (1 - yy) * np.log(1 - mu)))
            p0 = np.clip(float(yy.mean()), eps, 1 - eps)
            llnull = float(np.sum(yy * np.log(p0) + (1 - yy) * np.log(1 - p0)))
        if llnull == 0 or not np.isfinite(llf) or not np.isfinite(llnull):
            return None
        return float(1 - llf / llnull)
    except Exception:
        return None


def _vcov_specs(clusters: list[str]) -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = [("iid", "iid"), ("hetero", "hetero")]
    for c in clusters:
        specs.append((f"cluster_{c}", {"CRV1": c}))
    if len(clusters) >= 2:
        specs.append(("cluster_twoway", {"CRV1": "+".join(clusters)}))
    return specs


def _n_clusters(df: pd.DataFrame, vcov: Any) -> int | None:
    if isinstance(vcov, dict) and "CRV1" in vcov:
        vars_ = str(vcov["CRV1"]).split("+")
        return int(min(df[v].nunique() for v in vars_ if v in df.columns))
    return None


def _reg_nb(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    """负二项（NB2）。pyfixest 不支持，走 statsmodels。

    边界：不做高维固定效应——NB 的个体 FE 同样有附带参数问题；
    给了 fe 就报错并建议改用 poisson（PPML 在高维 FE 下的标准做法）。
    """
    import statsmodels.api as sm

    fid = cfg.get("id", "nb")
    y = cfg["y"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    controls = cfg.get("controls") or []
    if controls and isinstance(controls[0], list):
        controls = controls[-1]
    fe = cfg.get("fe") or []
    if fe:
        return Finding(fid, "reg", "fail",
                       f"负二项不支持固定效应 {fe}：面板计数模型请用 model: poisson"
                       "（PPML + 高维 FE 是标准做法），或把行业/年份做成哑变量后去掉 fe", {})
    clusters = cfg.get("cluster") or []
    need = [y] + list(xs) + list(controls) + list(clusters)
    missing = [c for c in need if c not in df.columns]
    if missing:
        return Finding(fid, "reg", "fail", f"列不存在：{missing}", {})

    d = df.dropna(subset=need)
    X = sm.add_constant(d[list(xs) + list(controls)].astype(float))
    yy = d[y].astype(float)
    focus = cfg.get("focus") or (xs[0] if xs else None)

    specs = [("nonrobust", {}), ("hetero", {"cov_type": "HC1"})]
    for c in clusters:
        specs.append((f"cluster_{c}", {"cov_type": "cluster", "cov_kwds": {"groups": d[c]}}))

    fitted, warnings = {}, []
    for vname, kw in specs:
        try:
            m = sm.NegativeBinomial(yy, X).fit(disp=0, **kw)
            terms = {v: {"coef": float(m.params[v]), "se": float(m.bse[v]),
                         "t": float(m.tvalues[v]), "p": float(m.pvalues[v])}
                     for v in [*xs, *controls] if v in m.params.index}
            fitted[vname] = {"terms": terms, "n": int(m.nobs),
                             "alpha": float(m.params.get("alpha", float("nan")))}
        except Exception as exc:
            warnings.append(f"{vname} 口径失败：{type(exc).__name__}: {exc}")

    if not fitted:
        return Finding(fid, "reg", "fail", "所有推断口径都失败（检查过散布与样本量）", {})
    base = next((n for n, _ in reversed(specs) if n in fitted), "hetero")
    main = fitted[base]
    coef = main["terms"].get(focus, {}).get("coef")
    why = (f"NB2：{focus} → {y}，β={coef:+.4f}，"
           f"alpha={main['alpha']:.4f}（过散布参数），N={main['n']}（{base}）"
           if coef is not None else "NB2 估计完成，但聚焦系数未提取")
    if main["alpha"] < 0.05:
        warnings.append(f"alpha={main['alpha']:.3f} 很小：数据接近等散布，poisson 可能就够用")
    steps = [{"label": "单设定", "formula": f"{y} ~ {'+'.join(xs + controls)}",
              "main_vcov": base, "focus": focus, "model": "nb",
              "coef": coef, "n": main["n"], "r2": None,
              "by_vcov": {k: {"terms": v["terms"], "n": v["n"], "r2": None, "g": None}
                          for k, v in fitted.items()}}]
    return Finding(fid, "reg", "warn" if warnings else "ok", why,
                   {"y": y, "x": xs, "fe": [], "cluster": clusters, "main_vcov": base,
                    "steps": steps, "focus": focus, "iv": None, "model": "nb",
                    "alpha": main["alpha"],
                    "note": "NB2 的 alpha 是过散布参数；与 poisson 对照可判断是否需要它"},
                   warnings=warnings)


def reg(df: pd.DataFrame, cfg: dict[str, Any]) -> Finding:
    if pf is None:
        return Finding(cfg.get("id", "reg"), "reg", "fail", "未安装 pyfixest", {})
    fid = cfg.get("id", "reg")
    y = cfg["y"]
    xs = cfg.get("x", [])
    if isinstance(xs, str):
        xs = [xs]
    fe = cfg.get("fe", []) or []
    clusters = cfg.get("cluster", []) or []
    ladder = cfg.get("controls", [])
    if ladder and not isinstance(ladder[0], list):
        ladder = [ladder]
    if not ladder:
        ladder = [[]]
    iv = cfg.get("iv")
    model = (cfg.get("model") or "ols").lower()
    if model == "nb":
        return _reg_nb(df, cfg)
    if model not in OLS_MODELS | GLM_FAMILIES | {"nb"}:
        return Finding(cfg.get("id", "reg"), "reg", "fail",
                       f"model={model!r} 不在支持范围；ols / logit / probit / poisson 走回归层，"
                       "tobit 与 nb 需单独写（见 README）", {})
    if iv and model not in OLS_MODELS:
        return Finding(cfg.get("id", "reg"), "reg", "fail",
                       "IV 只支持线性模型（ols）；离散选择的 IV 请用控制函数法分两步写", {})
    focus = cfg.get("focus") or (xs[0] if xs else None) or (iv["endog"][0] if iv else None)

    vcovs = _vcov_specs(clusters)
    base = cfg.get("main_vcov") or next((n for n, _ in reversed(vcovs)
                                         if n.startswith("cluster")), "hetero")

    steps, warnings, status = [], [], "ok"
    _last_model = None
    for i, ctrls in enumerate(ladder):
        fml = _formula(y, xs, ctrls, fe, iv)
        label = "+".join(ctrls) if ctrls else "单变量"
        fitted = {}
        try:
            for vname, vspec in vcovs:
                m = _fit(fml, df, vspec, model)
                if vname == base:
                    _last_model = m
                tidy = m.tidy()
                row = {}
                wanted = list(xs) + (list(iv["endog"]) if iv else [])
                for xv in wanted:
                    if xv not in tidy.index:
                        continue
                    row[xv] = {
                        "coef": float(tidy.loc[xv, "Estimate"]),
                        "se": float(tidy.loc[xv, "Std. Error"]),
                        "t": float(tidy.loc[xv, "t value"]),
                        "p": float(tidy.loc[xv, "Pr(>|t|)"]),
                    }
                g = int(min(m._G)) if getattr(m, "_G", None) else _n_clusters(df, vspec)
                r2 = float(m._r2) if m._r2 is not None and np.isfinite(m._r2) else None
                fitted[vname] = {"terms": row, "n": int(m._N), "r2": r2, "g": g}
        except Exception as exc:
            steps.append({"label": label, "formula": fml, "error": f"{type(exc).__name__}: {exc}"})
            status = "fail"
            warnings.append(f"{label} 估计失败：{exc}")
            continue

        main = fitted.get(base, {})
        step = {"label": label, "formula": fml, "main_vcov": base, "focus": focus,
                "model": model,
                "coef": main.get("terms", {}).get(focus, {}).get("coef"),
                "n": main.get("n"), "r2": main.get("r2"), "by_vcov": fitted}
        steps.append(step)

        # 口径敏感性：主口径显著而某口径不显著，或符号翻转
        mt = main.get("terms", {}).get(focus, {}).get("t")
        for vname, v in fitted.items():
            t = v["terms"].get(focus, {}).get("t")
            if mt is None or t is None:
                continue
            if np.sign(mt) != np.sign(t):
                warnings.append(f"{label}：{vname} 的系数符号与主口径({base})相反")
                status = "warn"
            elif abs(mt) >= 1.96 > abs(t):
                warnings.append(f"{label}：主口径 t={mt:.2f} 显著，{vname} t={t:.2f} 不显著")
                status = "warn"
        gs = [v["g"] for v in fitted.values() if v["g"]]
        if gs and min(gs) < 30:
            warnings.append(f"{label}：最小聚类数 {min(gs)} < 30，聚类标准误不可靠")
            status = "warn"

    main0 = next((s for s in steps if "by_vcov" in s), None)
    if main0:
        coef = main0.get("coef")
        t_main = main0["by_vcov"][base]["terms"].get(focus, {}).get("t")
        if coef is None:
            why = f"{focus} → {y}：系数未能提取（检查变量名是否与 tidy 一致）"
        else:
            why = (f"{focus} → {y}：β={coef:.4f}，t={t_main:.2f}（{base}，N={main0['n']}）")
    else:
        why = "所有设定均估计失败"
    iv_diag = None
    if iv:
        # IV 诊断按主回归同一套固定效应与聚类口径算；用最全的一组控制变量。
        try:
            iv_diag = iv_diagnostics(df, {"y": y, "x": xs, "controls": ladder[-1],
                                          "fe": fe, "cluster": clusters, "iv": iv})
            w_lines, iv_bad = iv_warnings(iv_diag)
            warnings.extend(w_lines)
            if iv_bad:
                status = "warn" if status != "fail" else status
        except Exception as exc:
            warnings.append(f"IV 诊断失败：{type(exc).__name__}: {exc}")
            status = "warn" if status != "fail" else status

    extra: dict[str, Any] = {"model": model}
    if model in GLM_FAMILIES:
        # 系数不是边际效应：离散模型要额外给 dy/dx，否则论文里会把 0.8 读成「概率上升 80 个百分点」
        try:
            terms = next((st["by_vcov"][base]["terms"] for st in reversed(steps)
                          if "by_vcov" in st), {})
            me = _avg_marginal(model, terms, _last_model)
            if me:
                extra["marginals"] = me
        except Exception as exc:
            warnings.append(f"平均边际效应没算出来：{type(exc).__name__}: {exc}")
        pr2 = _pseudo_r2(_last_model, df[y].values, model, len(df)) if _last_model else None
        if pr2 is not None:
            extra["pseudo_r2"] = pr2
        extra["note"] = "系数是潜在指数的斜率，不是边际效应；报结论用 marginals 里的 dy/dx"

    return Finding(fid, "reg", status, why,
                   {"y": y, "x": xs, "fe": fe, "cluster": clusters, "main_vcov": base,
                    "steps": steps, "focus": focus, "iv": iv_diag, **extra},
                   warnings=warnings)
