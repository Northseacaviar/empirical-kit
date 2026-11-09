"""把 v2 样本的分红 JSON 整理成 epkit 可吃的横截面表。

这是数据准备，不属于 epkit 内核：工具只吃干净表。
"""
import json
from pathlib import Path

import pandas as pd

SRC = Path("D:/guojiutiao-paper/data")
OUT = Path(__file__).parent / "cs_guojiutiao.csv"

codes = pd.read_csv(SRC / "sample_v2_codes.csv", dtype={"code": str})
rows, missing = [], 0
for c in codes["code"]:
    f = SRC / "div_v2" / f"{c.replace('.', '_')}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"code": c, "div2023": d.get("2023"), "div2022": d.get("2022")})
    else:
        missing += 1
        rows.append({"code": c, "div2023": None, "div2022": None})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"写出 {OUT}：{len(df)} 行，缺分红文件 {missing} 个")
print(df["div2023"].describe())
print("零分红：", int((df.div2023 == 0).sum()), " 有分红：", int((df.div2023 > 0).sum()))
