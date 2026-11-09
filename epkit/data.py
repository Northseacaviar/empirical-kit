"""数据层：读表、加载价格、记样本流。

样本流是这里唯一不可省的产物：每一步剔除了多少、为什么，直接进论文的筛选表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .spec import FilterSpec, StudySpec


def read_table(path: str | Path, encoding: str = "utf-8-sig", **kw) -> pd.DataFrame:
    """按后缀读表。UTF-8 BOM 是 CSMAR/巨潮导出的常态，默认用 utf-8-sig。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在：{p}")
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(p, **kw)
    if suf == ".parquet":
        return pd.read_parquet(p, **kw)
    if suf in {".dta"}:
        return pd.read_stata(p, **kw)
    return pd.read_csv(p, encoding=encoding, **kw)


@dataclass
class Step:
    step: str
    reason: str
    n_before: int
    n_after: int

    @property
    def dropped(self) -> int:
        return self.n_before - self.n_after

    def as_dict(self) -> dict:
        return {"step": self.step, "reason": self.reason, "n_before": self.n_before,
                "n_after": self.n_after, "dropped": self.dropped}


@dataclass
class SampleFlow:
    """样本流。初始 n 由调用方给定，之后每步追加一行。"""

    initial: int = 0
    steps: list[Step] = field(default_factory=list)

    @property
    def final(self) -> int:
        return self.steps[-1].n_after if self.steps else self.initial

    def record(self, step: str, reason: str, n_before: int, n_after: int) -> None:
        self.steps.append(Step(step, reason, n_before, n_after))

    def apply(self, df: pd.DataFrame, f: FilterSpec, idx: int) -> pd.DataFrame:
        """应用一条 FilterSpec，并记录样本流。"""
        n0 = len(df)
        if f.drop_if:
            df = df[~df.eval(f.drop_if).fillna(False).astype(bool)].copy()
        if f.keep_if:
            df = df[df.eval(f.keep_if).fillna(False).astype(bool)].copy()
        for col, val in f.drop_flags.items():
            if col in df.columns:
                df = df[df[col] != val].copy()
        if f.require_nonnull:
            cols = [c for c in f.require_nonnull if c in df.columns]
            if cols:
                df = df.dropna(subset=cols).copy()
        label = f.name or f"filter_{idx}"
        self.record(label, f.reason or f.drop_if or f.keep_if or "剔除标记行", n0, len(df))
        return df

    def as_list(self) -> list[dict]:
        return [s.as_dict() for s in self.steps]


def load_prices(spec: StudySpec) -> dict[str, pd.DataFrame]:
    """读价格。返回 {code: DataFrame(index=日期, 列为原表列)}。

    支持两种布局：
    - data.prices 是目录：每个文件一只证券，代码取自文件名（去掉前缀分隔符）
    - data.prices 是文件：长表，按 data.code_col 拆分
    """
    cfg = spec.data
    if not cfg.prices:
        raise ValueError("spec.data.prices 未设置")
    p = Path(cfg.prices)
    date_col, close_col = cfg.date_col, cfg.close_col

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if date_col not in df.columns:
            raise KeyError(f"价格表缺少日期列 {date_col}；实际列：{list(df.columns)[:8]}")
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        for c in df.columns:
            if c != date_col and df[c].dtype == object:
                conv = pd.to_numeric(df[c], errors="coerce")
                # 只有大比例可转数值时才转换，避免把字符串标签列毁掉
                if conv.notna().mean() > 0.9:
                    df[c] = conv
        if close_col in df.columns:
            df = df[df[close_col].notna() & (df[close_col] > 0)]
        return df.set_index(date_col).sort_index()

    out: dict[str, pd.DataFrame] = {}
    if p.is_dir():
        for f in sorted(p.glob(cfg.prices_glob)):
            code = f.stem.replace("_", ".", 1)
            out[code] = _norm(read_table(f, encoding=cfg.encoding))
    elif p.is_file():
        long = read_table(p, encoding=cfg.encoding)
        if cfg.code_col not in long.columns:
            raise KeyError(f"长表缺少代码列 {cfg.code_col}")
        for code, g in long.groupby(cfg.code_col):
            out[str(code)] = _norm(g.drop(columns=[cfg.code_col]))
    else:
        raise FileNotFoundError(f"prices 路径不存在：{p}")
    if not out:
        raise ValueError(f"{p} 下没有读到任何价格文件")
    return out


def load_market(spec: StudySpec) -> pd.Series | None:
    """读市场指数，返回日收益序列。"""
    cfg = spec.data
    if not cfg.index:
        return None
    df = read_table(cfg.index, encoding=cfg.encoding)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    s = df.dropna(subset=[cfg.date_col]).set_index(cfg.date_col)[cfg.close_col]
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    return s.pct_change(fill_method=None)


def load_cross_section(spec: StudySpec) -> pd.DataFrame | None:
    cfg = spec.data
    if not cfg.cross_section:
        return None
    df = read_table(cfg.cross_section, encoding=cfg.encoding)
    if cfg.code_col in df.columns:
        df[cfg.code_col] = df[cfg.code_col].astype(str)
    return df
