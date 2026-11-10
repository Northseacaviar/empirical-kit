"""造一份覆盖 v2 新能力的面板：合并、离散、删失、中介、跨组、bootstrap。

真结构：
- treat 是「偶数号公司在 2020 年后被处理」，在年内有变异（否则被年份固定效应吸收）；
- roa 随 treat 上升，且 roa 影响 y → treat 经 roa 的中介路径真实存在；
- soe 两组的 treat 斜率不同（0.5 vs 1.2）；
- violate 是 0/1，y_cens 在 0 处左删失。
财务表是公司-年面板（长表），所以合并键是 [code, year]。
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2025)
n_firm, years = 400, 6
rows, fin = [], []
for i in range(n_firm):
    code = f"{i:06d}"
    size = rng.normal(8, 1.2)
    soe = int(i % 3 == 0)
    treated_firm = int(i % 2 == 0)
    lev = rng.uniform(0.2, 0.8)
    for t in range(2018, 2018 + years):
        treat = int(treated_firm and t >= 2020)
        roa = 0.05 + 0.10 * treat + rng.normal(0, 0.02)
        fin.append({"code": code, "year": t, "roa": roa, "lev": lev})
        slope = 0.5 if soe else 1.2
        y = slope * treat + 0.8 * roa + 0.3 * size + rng.normal(0, 0.4)
        y_cens = max(0.0, 0.4 + 1.0 * treat - 0.2 * size + rng.normal(0, 0.6))
        violate = int(rng.random() < 1 / (1 + np.exp(-(-1.2 + 0.8 * treat + 0.2 * size))))
        rows.append({"code": code, "year": t, "treat": treat, "soe": soe,
                     "size": size, "y": y, "y_cens": y_cens, "violate": violate})

pd.DataFrame(rows).to_csv(Path(__file__).parent / "main.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(fin).to_csv(Path(__file__).parent / "fin.csv", index=False, encoding="utf-8-sig")
print(f"{len(rows)} 行面板，{n_firm} 家公司；财务表 {len(fin)} 行（公司-年）")
