"""机制测试：变量构造 DSL。"""
import numpy as np
import pandas as pd
import pytest

from epkit.vars import evaluate


def df():
    return pd.DataFrame({"x": [1.0, 2, 4, 8], "pid": [1, 1, 2, 2],
                         "a": [10.0, 20, 30, 40], "b": [2.0, 4, 5, 8]})


def test_lag_respects_group():
    out = evaluate("lag(x, by=pid)", df())
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[2])
    assert out.iloc[1] == 1 and out.iloc[3] == 4


def test_log_keeps_nonpositive_nan():
    d = pd.DataFrame({"x": [0.0, -1, 2]})
    out = evaluate("log(x)", d)
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(np.log(2))


def test_ratio_zero_denominator_is_nan():
    d = pd.DataFrame({"a": [1.0, 2], "b": [0.0, 2]})
    out = evaluate("ratio(a, b)", d)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_cut_q_returns_1_to_q():
    d = pd.DataFrame({"x": np.arange(100.0)})
    out = evaluate("cut_q(x, 5)", d)
    assert set(out.unique()) == {1, 2, 3, 4, 5}


def test_arithmetic_and_compare():
    d = df()
    out = evaluate("(a > 15) & (b < 8)", d)
    assert list(out) == [False, True, True, False]


def test_rejects_unsafe_expr():
    with pytest.raises(ValueError):
        evaluate("__import__('os').system('echo hi')", df())
    with pytest.raises(ValueError):
        evaluate("x.__class__", df())
