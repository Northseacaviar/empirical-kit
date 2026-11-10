"""CSMAR 导出表的列名映射与收益率单位识别。

两个真实踩过的坑：
1. 列名对不上——CSMAR 的 Stkcd/Trddt/Clsprc/Dretwd，spec 要的是 code/date/close/ret，
   每次换课题都手写一遍 rename，写错列名报错还报得晚。
2. 单位不一致——CSMAR 的 Dretwd 是小数（0.008），自建面板常常是百分点（0.8）。
   两者混用差 100 倍，而且不报错。这里在读数时就判出来，和 spec 声明的 scale 对一遍。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 只收"跨课题反复用到、且语义无歧义"的字段；公司治理那类一表一义的列不往里塞。
CSMAR_MAP = {
    # 标识
    "Stkcd": "code", "Symbol": "code", "证券代码": "code",
    "Trddt": "date", "Tradedate": "date", "交易日期": "date",
    "Accper": "accper", "Typrep": "typrep",
    # 行情
    "Clsprc": "close", "收盘价": "close",
    "Oppr": "open", "Hipr": "high", "Lopr": "low",
    "Prcrt": "pct_chg", "涨跌幅": "pct_chg",
    "Adjprcwd": "close_adj", "Adjprcnd": "close_adj_nd",
    "Trdsta": "tradestatus",
    # 收益率（D=日，Y=年，wd=考虑现金红利再投资，nd=不考虑）
    "Dretwd": "ret", "Dretnd": "ret_nd", "Yretwd": "yret", "Yretnd": "yret_nd",
    # 规模与行业
    "Msmvttl": "mv_total_raw", "Mv": "mv", "Dsmvtll": "mv_float",
    "Nindcd": "industry", "Nindnme": "industry_name",
    "Indcd": "industry_old", "Indnme": "industry_name_old",
    # 常见公司文件字段
    "ShortName": "name", "Stknme": "name", "Ipodate": "ipo_date", "Listdt": "list_date",
    "Annodt": "announce_date", "Annodt1": "announce_date",
}

RETURN_COLS = ("ret", "ret_nd", "yret", "yret_nd", "pct_chg", "car_0_1")


def normalize(df: pd.DataFrame, mapping: str | dict = "csmar") -> tuple[pd.DataFrame, dict]:
    """按映射表重命名列。返回 (df, info)，info 记录改了哪些列。"""
    table = CSMAR_MAP if isinstance(mapping, str) and mapping == "csmar" else mapping
    if not isinstance(table, dict):
        raise ValueError(f"未知的列名映射 {mapping!r}；支持 'csmar' 或自定义 dict")
    if mapping == "csmar":
        table = CSMAR_MAP
    rename = {k: v for k, v in table.items() if k in df.columns and k != v}
    dropped, conflicts = [], {}
    # 多个原始列映射到同一个标准名（CSMAR 合并表常同时给 Stkcd 与 Symbol）：
    # 直接 rename 会得到两列同名 code，后面 out["code"].str 变成 DataFrame 崩掉。
    from collections import Counter

    for tgt, cnt in Counter(rename.values()).items():
        if cnt <= 1:
            continue
        srcs = [k for k, v in rename.items() if v == tgt]
        keep = srcs[0]
        for k in srcs[1:]:
            rename.pop(k, None)
            dropped.append(k)
        conflicts[tgt] = {"kept": keep, "dropped": srcs[1:]}
    out = df.rename(columns=rename)
    # 代码列去前缀/补零：600000 vs sh.600000 vs 600000.0
    if "code" in out.columns:
        code = (out["code"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .str.replace(r"^(sh|sz|bj)\.", "", regex=True, case=False))
        # Excel 会把 000001 读成 1；不补零就和 baostock 的 6 位代码 merge 成 0 行
        is_num = code.str.fullmatch(r"\d+")
        out["code"] = code.where(~is_num, code.str.zfill(6))
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    info = {"renamed": rename, "mapping": "csmar" if isinstance(mapping, str) else "custom"}
    if dropped:
        info["dropped_duplicate_sources"] = dropped
        info["conflicts"] = conflicts
    return out, info


def detect_return_units(s: pd.Series) -> str | None:
    """判收益率的单位：'decimal'（0.01）还是 'percent'（1.0）。"""
    v = pd.to_numeric(s, errors="coerce").dropna().abs()
    if len(v) < 20:
        return None
    med, p99 = float(v.median()), float(v.quantile(0.99))
    # 用 P99 而不是中位数判：日收益率与 CAR 的量级差两个数量级，中位数的界没法通用。
    # 小数口径下日收益率 P99 约 0.05、CAR 约 0.1；百分点口径下分别是 5 与 2~3。
    if p99 > 1.5:
        return "percent"
    if p99 < 0.5:
        return "decimal"
    return None          # 落在中间（年收益率、特殊构造变量）就不判，宁可沉默也不误报


def audit_units(df: pd.DataFrame, scale: float = 1.0,
                unit: str = "") -> list[str]:
    """把单位判定与 spec 声明的 scale 对一遍，返回给人读的告警。"""
    notes = []
    cols = [c for c in df.columns if c in RETURN_COLS or str(c).startswith("car_")]
    for c in cols:
        u = detect_return_units(df[c])
        if u is None:
            continue
        med = float(pd.to_numeric(df[c], errors="coerce").abs().median())
        if u == "decimal" and scale == 1.0:
            notes.append(
                f"{c} 看起来是小数口径（中位 |{c}|={med:.4f}），而 spec 没有换算系数："
                f"若后续按百分点解读会差 100 倍，需要时写 data.scale: 100")
        elif u == "percent" and scale not in (1.0, None):
            notes.append(
                f"{c} 看起来已经是百分点口径（中位 |{c}|={med:.4f}），"
                f"但 spec 写了 scale={scale}：会再乘一次，量级被放大 {scale:g} 倍")
    if unit and unit != "%" and cols:
        notes.append(f"spec 的单位写成 {unit!r}，但表里有收益率列：确认单位标注是否写错")
    return notes
