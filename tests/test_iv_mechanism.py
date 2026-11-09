"""机制测试：IV 必须纠正 OLS 的内生性偏差。"""
import numpy as np
import pandas as pd
import pytest

from epkit.estimate.reg import reg


def make(seed=1, n=3000, beta=1.5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)            # 工具变量：只通过 x 影响 y
    u = rng.normal(size=n)            # 内生误差
    x = 0.6 * z + u                   # x 与 u 相关 → OLS 有偏
    y = beta * x + u + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_ols_is_biased_iv_is_not():
    df = make()
    ols = reg(df, {"id": "ols", "y": "y", "x": ["x"]}).data["steps"][0]
    iv = reg(df, {"id": "iv", "y": "y", "x": [], "iv": {"endog": ["x"], "instruments": ["z"]}}
             ).data["steps"][0]
    b_ols = ols["by_vcov"]["hetero"]["terms"]["x"]["coef"]
    b_iv = iv["by_vcov"]["hetero"]["terms"]["x"]["coef"]
    assert b_ols > 1.9, f"OLS 应当偏高，实际 {b_ols:.3f}"
    assert b_iv == pytest.approx(1.5, abs=0.12), f"IV 未恢复真值：{b_iv:.3f}"
