"""多重检验校正：把同一族的检验放在一起看。

为什么必须有：一次跑 8 个事件窗口 + 3 个分组就是 11 次检验，
按单次 5% 判断，族错误率 1−0.95^11 ≈ 43%。论文要么预注册主检验，
要么给出校正后的结论；这两种做法都需要先知道"这一族里有几次检验"。

方法：
- bh    Benjamini-Hochberg（控 FDR，实证论文最常用）
- holm  Holm（控 FWER，比 Bonferroni 有力）
- bonferroni 最保守，只在审稿人点名时用
"""
from __future__ import annotations

import numpy as np

from ..findings import Finding, Findings

METHODS = {"bh": "fdr_bh", "holm": "holm", "bonferroni": "bonferroni"}


def collect_p(F: Findings, fids: list[str] | None = None) -> list[tuple[str, float]]:
    """从已有结果里抽出可校正的 p 值。

    from 为空表示整族：所有带 p 值的检验都进同一个族。
    一个结果可以贡献多个 p（DID 的每个事件时间各算一次），
    因为它们本来就是同一族的多次检验。
    """
    out: list[tuple[str, float]] = []
    for f in F.items:
        if fids is not None and f.id not in fids:
            continue
        d = f.data or {}
        if f.type == "group_diff" and d.get("p_welch") is not None:
            out.append((f.id, float(d["p_welch"])))
        elif f.type == "mean_test" and d.get("p") is not None:
            out.append((f.id, float(d["p"])))
        elif f.type == "reg":
            focus = d.get("focus")
            base = d.get("main_vcov")
            for st in d.get("steps", []):
                if "by_vcov" not in st:
                    continue
                term = st["by_vcov"].get(base, {}).get("terms", {}).get(focus, {})
                if term.get("p") is not None and np.isfinite(term["p"]):
                    out.append((f"{f.id}:{focus}[{st['label']}]", float(term["p"])))
        elif f.type == "did":
            for r in d.get("event_time", []):
                if r.get("p") is not None and np.isfinite(r["p"]):
                    out.append((f"{f.id}:k={r['k']:+d}", float(r["p"])))
    return out


def multitest(F: Findings, cfg: dict) -> Finding:
    fid = cfg.get("id", "multitest")
    method = cfg.get("method", "bh")
    alpha = float(cfg.get("alpha", 0.05))
    if method not in METHODS:
        return Finding(fid, "multitest", "fail",
                       f"未知校正方法 {method}；支持 {sorted(METHODS)}", {})

    pairs = collect_p(F, cfg.get("from"))
    if len(pairs) < 2:
        return Finding(fid, "multitest", "warn",
                       f"可校正的 p 值只有 {len(pairs)} 个，族太小，校正没有意义",
                       {"method": method, "tests": []})

    labels = [a for a, _ in pairs]
    p = np.array([b for _, b in pairs], dtype=float)
    try:
        from statsmodels.stats.multitest import multipletests
        rej, p_adj, _, _ = multipletests(p, alpha=alpha, method=METHODS[method])
    except Exception:  # 退化为手写 BH，不因为缺依赖交白卷
        order = np.argsort(p)
        m = len(p)
        p_adj = np.empty_like(p)
        prev = 1.0
        for rank_pos in range(m - 1, -1, -1):
            i = order[rank_pos]
            val = p[i] * m / (rank_pos + 1)
            prev = min(prev, val)
            p_adj[i] = min(prev, 1.0)
        rej = p_adj <= alpha

    tests = [{"label": lb, "p_raw": float(a), "p_adj": float(b),
              "sig_raw": bool(a <= alpha), "sig_adj": bool(c)}
             for lb, a, b, c in zip(labels, p, p_adj, rej)]
    n_raw = int(sum(t["sig_raw"] for t in tests))
    n_adj = int(sum(t["sig_adj"] for t in tests))
    flipped = [t["label"] for t in tests if t["sig_raw"] and not t["sig_adj"]]

    status = "ok"
    warnings = []
    if flipped:
        status = "warn"
        warnings.append(f"校正后有 {len(flipped)} 条结果由显著转为不显著："
                        + "、".join(flipped[:6]) + ("…" if len(flipped) > 6 else ""))
        warnings.append(f"族内 {len(tests)} 次检验，按 {method} 控 FDR 后显著 {n_adj} 条"
                        f"（未校正 {n_raw} 条）——正文不能只报未校正的结论")
    why = (f"{method} 校正：族内 {len(tests)} 次检验，未校正显著 {n_raw} 条，"
           f"校正后显著 {n_adj} 条")
    return Finding(fid, "multitest", status, why,
                   {"method": method, "alpha": alpha, "tests": tests,
                    "n_sig_raw": n_raw, "n_sig_adj": n_adj, "flipped": flipped},
                   warnings=warnings)
