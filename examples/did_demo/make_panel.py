"""造一份 DID 演示面板：200 家公司 × 2010-2022，2015 年一半被处理。

真处理效应 0.8，无预趋势。用来验证整条 DID 链路（含事件研究图）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "panel.csv"

n_firm, years = 200, list(range(2010, 2023))
firm_fe = rng.normal(0, 1.0, n_firm)
year_fe = rng.normal(0, 0.4, len(years))
rows = []
for i in range(n_firm):
    treated = i % 2 == 0
    g = 2015 if treated else 0
    for ti, yr in enumerate(years):
        post = 1 if (treated and yr >= 2015) else 0
        y = (firm_fe[i] + year_fe[ti] + 0.8 * post + 0.1 * ti
             + rng.normal(0, 0.5))
        rows.append({"firm": i, "year": yr, "cohort": g, "size": firm_fe[i] + 5,
                     "y": y})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"{OUT}：{len(df)} 行，{df.firm.nunique()} 家公司，处理组 {int((df.cohort>0).sum())} 行")
