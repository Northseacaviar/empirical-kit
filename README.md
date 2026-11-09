# epkit

Spec-driven empirical toolkit for research agents.

一份 spec 定义样本、事件、窗口与检验；一条命令跑完，产出表、图与结构化结果。
每个数字带 status（ok / warn / fail）、why 和 spec hash，agent 可直接判读。

## Install

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe pandas numpy scipy statsmodels pyfixest pydantic pyyaml jinja2 matplotlib openpyxl
```

## Use

```
python -m epkit validate studies/your.yaml
python -m epkit run studies/your.yaml
```

输出：`findings.json`（结构化结果 + 样本流 + provenance）、`report.md`（表）。

## Spec

```yaml
name: my-study
data:
  prices: data/px            # 目录（每票一个 csv）或单个长表
  cross_section: data/xs.csv # 样本框 / 横截面变量
  index: data/index.csv      # 市场基准（method != raw 时必需）
event:
  mode: single               # single | column | list
  date: "2024-04-12"
returns:
  method: raw                # raw | market_adjusted | market_model
  car_mode: sum_daily        # sum_daily | close_to_close，二者相差一日
  windows: [[-1, 1], [0, 5]]
sample:
  filters:
    - require_nonnull: [div2023]
      reason: 缺分红数据
  groups:
    - name: zero_div
      expr: "div2023 == 0"
tests:
  - group_diff: {var: car_0_5, by: zero_div}
```

`car_mode` 与主检验口径（Welch / 等方差）都会写进每条结果，换口径不会静默。

## Test

```
python -m pytest tests/ -q
```

`tests/test_repro_guojiutiao.py` 用论文表 2A 的 8 个窗口作回归基线。

## Status

事件研究主线可用：CAR（三种基准、两种窗口口径）、样本流、组间差异检验、结构化输出。
条件：pandas 3.x / Python 3.12。

作者：[@Northseacaviar](https://github.com/Northseacaviar)
