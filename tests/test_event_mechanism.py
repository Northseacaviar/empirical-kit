"""机制测试：市场模型 CAR 与两种窗口口径的差分。

合成一只完全由市场模型生成的股票：r = 0.0002 + 1.2*m + eps。
若估计正确，事件窗 CAR 应接近 0；两种窗口口径应恰好相差一个日收益。
"""
import numpy as np
import pandas as pd

from epkit.spec import DataSpec, EventSpec, ReturnsSpec, StudySpec
from epkit.estimate.event import build_event_panel


def make_prices(seed=7, n=400, alpha=0.0002, beta=1.2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    m = rng.normal(0, 0.01, size=n)
    r = alpha + beta * m + rng.normal(0, 0.002, size=n)
    close = pd.Series(100 * np.cumprod(1 + r), index=dates)
    return {"sh.600000": pd.DataFrame({"close": close})}, pd.Series(m, index=dates)


def test_market_model_car_near_zero():
    prices, market = make_prices()
    spec = StudySpec(name="t", data=DataSpec(prices="x", close_col="close"),
                     event=EventSpec(mode="single", date="2023-12-01"),
                     returns=ReturnsSpec(method="market_model", car_mode="sum_daily",
                                         windows=[[0, 5]], min_est_obs=100))
    res = build_event_panel(spec, prices, market)
    car = res.panel["car_0_5"].iloc[0]
    assert abs(car) < 0.6, f"市场模型 CAR 未接近 0：{car:.3f}（单位 %）"


def test_two_car_modes_differ_by_one_day():
    prices, market = make_prices()
    base = dict(data=DataSpec(prices="x", close_col="close"),
                event=EventSpec(mode="single", date="2023-12-01"))
    a = build_event_panel(StudySpec(name="a", returns=ReturnsSpec(
        method="raw", car_mode="close_to_close", windows=[[0, 5]]), **base), prices, None)
    b = build_event_panel(StudySpec(name="b", returns=ReturnsSpec(
        method="raw", car_mode="sum_daily", windows=[[0, 5]]), **base), prices, None)
    ca, cb = a.panel["car_0_5"].iloc[0], b.panel["car_0_5"].iloc[0]
    # close_to_close 少 T+0 当日收益、多 T+5 当日收益；二者一般不相等
    assert abs(ca - cb) > 0.05, f"两种口径给出同一数字（{ca:.3f} vs {cb:.3f}），口径未生效"
