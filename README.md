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

## Robustness

```yaml
- leave_one_out: {var: car_0_5, by: zero_div}
- placebo: {var: car_0_5, by: zero_div, n: 10, gap: 20}   # 自动取事件日前的非事件锚点
```

安慰剂输出 z =（事件日差 − 锚点均值）/ 锚点标准差，|z| < 1.96 判 `warn`：同一套口径在随便哪天也能复制出相近的组间差。留一法给最敏感单点的位移占原差的比例，超过 20% 判 `warn`。

## Regression

```yaml
tests:
  - reg:
      id: h2
      y: car_m3_3
      x: [disclosed_asset, log_amount, "log_amount:lev_w"]  # 支持交互项
      controls: [size, lev_w, roa_w]
      fe: [industry1, year]
      cluster: [sec_code, announcement_date]   # 自动出 iid/hetero/个体/时间/双向
      focus: "log_amount:lev_w"                # 结论看哪个系数
```

五套推断口径一次跑全；任一模式与其他模式在 5% 水平上结论不一致，结果标 `warn` 并写明哪两个口径、各是什么 t 值。聚类数 < 30 也标 `warn`。

已算好面板的课题跳过 CAR 计算：把 `data.panel` 指向 CSV（含因变量、解释变量、固定效应列、聚类列），`event` 可省。

IV：`x: [exog]` + `iv: {endog: [x], instruments: [z]}`。

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

组间差异检验支持就地做稳健性，不必另写脚本：

```yaml
- group_diff: {var: car_0_5, by: zero_div, id: base}
- group_diff: {var: car_0_5, by: zero_div, id: winsor, winsor: [0.01, 0.99]}
- group_diff: {var: car_0_5, by: zero_div, id: ex_big, drop_where: "lncap > 25"}
```

`car_mode` 与主检验口径（Welch / 等方差）、缩尾区间、剔除条件都会写进每条结果，换口径不会静默。

## Test

```
python -m pytest tests/ -q
```

`tests/test_repro_guojiutiao.py` 用论文表 2A 的 8 个窗口作回归基线。

## Status

事件研究 + 横截面回归可用：CAR（三种基准、两种窗口口径）、样本流、组间差异、OLS/FE/IV + 五套推断口径、结构化输出。
稳健性：缩尾、剔除大类、留一法、时间安慰剂。
条件：pandas 3.x / Python 3.12。

作者：[@Northseacaviar](https://github.com/Northseacaviar)
