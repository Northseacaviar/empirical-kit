"""造一份覆盖演示用的横截面：用来验证描述/分布/规格曲线/子样本/分位回归。

真效应：ind==0 的公司效应 1.2，其余 0.3；x 右偏，故意让分布诊断能抓到偏态。
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
n = 1200
x = np.exp(rng.normal(0, 1.0, n))                 # 右偏，未取对数
ind = rng.integers(0, 6, n)
year = rng.integers(2016, 2023, n)
c1 = rng.normal(size=n)
c2 = rng.normal(size=n)
eff = np.where(ind == 0, 1.2, 0.3)
y = eff * np.log1p(x) + 0.4 * c1 + rng.normal(size=n)
df = pd.DataFrame({"y": y, "lnx": np.log1p(x), "x": x, "c1": c1, "c2": c2,
                   "ind": ind, "year": year})
out = Path(__file__).parent / "cross.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{out}: {len(df)} rows")
