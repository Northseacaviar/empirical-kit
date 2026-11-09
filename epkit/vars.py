"""变量构造：受限表达式 + 常用变换函数。

为什么不用 df.eval：它不接受自定义函数（lag、winsor 都用不了），
而派生变量的可读性恰恰是 agent 写 spec 时最需要的东西。

安全性：AST 白名单，不允许属性访问与下划线名字，禁止内建函数。
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Compare, ast.BoolOp, ast.IfExp, ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BitAnd, ast.BitOr, ast.Invert, ast.keyword,
)


def _check(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED):
            raise ValueError(f"表达式里不允许 {type(n).__name__}（只支持算术、比较、函数调用）")
        if isinstance(n, ast.Name) and n.id.startswith("_"):
            raise ValueError(f"不允许使用以下划线开头的名字：{n.id}")
        if isinstance(n, ast.Call) and not isinstance(n.func, ast.Name):
            raise ValueError("函数调用必须直接写函数名")


# —— 变换函数库 ——
def lag(s, k: int = 1, by=None):
    """滞后 k 期。by 给定则按个体分组。"""
    if by is None:
        return s.shift(k)
    return s.groupby(by).shift(k)


def lead(s, k: int = 1, by=None):
    return lag(s, -k, by)


def diff(s, k: int = 1, by=None):
    """k 阶差分（一阶即 s - lag(s)）。"""
    return s - lag(s, k, by)


def growth(s, by=None):
    """增长率，极端值截到 [-10, 10]，避免分母接近 0 时爆炸。"""
    prev = lag(s, 1, by)
    g = (s - prev) / prev.abs().clip(lower=1e-12)
    return g.clip(-10, 10)


def log(s):
    """自然对数。非正值置 NaN，不静默改成 0。"""
    out = s.astype(float).where(s > 0)
    return np.log(out)


def lg(s):
    return log1p(s)


def log1p(s):
    return np.log1p(s.astype(float))


def ratio(a, b):
    """比率。分母为 0 置 NaN。"""
    return a / b.replace(0, np.nan)


def z(s):
    """标准化（按全样本）。"""
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd else s * np.nan


def demean(s, by=None):
    """组内去均值。"""
    if by is None:
        return s - s.mean()
    return s - s.groupby(by).transform("mean")


def group_mean(s, by):
    return s.groupby(by).transform("mean")


def winsor(s, lo: float = 0.01, hi: float = 0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def cut_q(s, q: int = 5, labels: bool = False):
    """按分位切组，返回 1..q 的整数。"""
    r = s.rank(method="first")
    codes = pd.qcut(r, q, labels=False) + 1
    return codes.astype(int)


def cut_v(s, breaks: list):
    """按给定断点切组。"""
    return pd.cut(s, bins=[-np.inf] + list(breaks) + [np.inf], labels=False).astype(int) + 1


def as_int(s):
    return s.astype(int)


def as_float(s):
    return s.astype(float)


def where(cond, a, b):
    return np.where(cond, a, b)


FUNCS = {
    "lag": lag, "lead": lead, "diff": diff, "growth": growth,
    "log": log, "ln": log, "log1p": log1p, "exp": np.exp,
    "ratio": ratio, "z": z, "demean": demean, "group_mean": group_mean,
    "winsor": winsor, "cut_q": cut_q, "cut_v": cut_v,
    "as_int": as_int, "as_float": as_float, "where": where,
    "abs": np.abs, "sqrt": np.sqrt, "clip": np.clip, "min": np.minimum, "max": np.maximum,
}


def evaluate(expr: str, df: pd.DataFrame) -> pd.Series | np.ndarray:
    """在 df 的列上求值。列名直接写，函数从 FUNCS 取。"""
    tree = ast.parse(expr, mode="eval")
    _check(tree)
    scope: dict = {}
    scope.update(FUNCS)
    for c in df.columns:
        scope[c] = df[c]
    # 允许在表达式里用 by="col" 之外的字符串字面量，不需要额外名字
    env = {"__builtins__": {}}
    return eval(compile(tree, "<derive>", "eval"), env, scope)
