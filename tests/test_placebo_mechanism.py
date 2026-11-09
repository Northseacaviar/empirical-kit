"""机制测试：安慰剂必须能分辨「真事件」与「噪声」。

合成：20 只股票、一半为处理组；只有事件日窗口内处理组有 +5% 超额收益。
期望：事件日 z 明显 < -1.96（效应远超安慰剂分布）；无效应版本则 warn。
"""
import numpy as np
import pandas as pd

from epkit.checks.placebo import time_placebo
from epkit.spec import DataSpec, EventSpec, ReturnsSpec, StudySpec


def make(seed=11, effect=0.08, n_days=160):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    event_pos = 140
    prices, rows = {}, []
    for i in range(20):
        r = rng.normal(0, 0.006, size=n_days)
        if i % 2 == 1:                       # 处理组
            r[event_pos:event_pos + 5] += effect / 5
        close = pd.Series(100 * np.cumprod(1 + r), index=dates)
        code = f"sh.{600000 + i}"
        prices[code] = pd.DataFrame({"close": close})
        rows.append({"code": code, "treat": i % 2})
    xs = pd.DataFrame(rows)
    spec = StudySpec(name="p", data=DataSpec(prices="x", close_col="close"),
                     event=EventSpec(mode="single", date=str(dates[event_pos].date())),
                     returns=ReturnsSpec(method="raw", car_mode="close_to_close",
                                         windows=[[0, 5]]))
    return spec, prices, xs


def test_real_effect_beats_placebo():
    spec, prices, xs = make(effect=0.08)
    f = time_placebo(spec, {"id": "p", "var": "car_0_5", "by": "treat", "n": 8, "gap": 12},
                     prices=prices, xs=xs)
    assert f.data["z"] > 1.96, f"真效应未被识别：z={f.data['z']:.2f}"
    assert f.status == "ok"


def test_null_effect_is_flagged():
    spec, prices, xs = make(effect=0.0)
    f = time_placebo(spec, {"id": "p", "var": "car_0_5", "by": "treat", "n": 8, "gap": 12},
                     prices=prices, xs=xs)
    assert f.status == "warn", f"无效应却判 ok：z={f.data['z']:.2f}"
    assert abs(f.data["z"]) < 2.5
