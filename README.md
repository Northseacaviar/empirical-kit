# epkit

Spec-driven empirical toolkit for research agents.

一份 spec 定义样本、事件、窗口与检验；一条命令跑完，产出表、图与结构化结果。
每个数字带 status（ok / warn / fail）、why 和 spec hash，agent 可直接判读。

## Install

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe pandas numpy scipy statsmodels pyfixest[bootstrap] pydantic pyyaml jinja2 matplotlib openpyxl python-docx baostock
# wild cluster bootstrap 需要：uv pip install wildboottest
```

## Use

```
python -m epkit validate studies/your.yaml
python -m epkit run studies/your.yaml [--out DIR] [--quiet] [--no-docx]

# 取数（baostock，前复权默认）
python -m epkit fetch --codes 600000,000001 --start 2023-01-01 --end 2024-06-30 --out data/prices
python -m epkit fetch --codes sh.000300 --start 2023-01-01 --end 2024-06-30 --out data/index.csv --index

# 导出表列名对齐（CSMAR）
python -m epkit map-columns --in raw.xlsx --out mapped.csv --mapping csmar
```

输出：

```
output/
  findings.json     结构化结果：每条带 status(ok/warn/fail)、why、spec hash、样本 n
  report.md         全部表
  report.docx       三线表：系数在上、标准误在下
  figures/          期限结构、安慰剂、事件研究、规格曲线、分布、趋势、箱线、相关热图
                    每张图另出一份 *_bw.png（灰度 + 线型，供黑白印刷）
```

安装成命令（可选）：`uv pip install -e .`，之后用 `epkit run ...`。

## Tests supported

| 类型 | 作用 |
|---|---|
| `group_diff` | 两组均值差 + Welch / 等方差 t，可就地缩尾、剔除 |
| `mean_test` | 单样本均值检验、中位数、正收益占比 |
| `reg` | OLS / FE / IV，五套推断口径一次给全；IV 附第一阶段聚类 F 与过度识别检验 |
| `reg: model` | `logit` / `probit` / `poisson`（高维 FE 与聚类口径不变）、`nb`（NB2，无 FE）；离散模型自动给平均边际效应与伪 R² |
| `tobit` | 删失正态 MLE，支持左右删失；只做低维 FE |
| `mediation` | Baron-Kenny 三步表 + 间接效应的 bootstrap 区间 |
| `group_test` | 两组系数差异（交互项，Chow 等价），带 FE 与聚类 |
| `inference` | wild cluster bootstrap（pyfixest/wildboottest）与 pairs bootstrap |
| `permutation` | 置换检验：打乱解释变量的经验 p |
| `trend` / `box` | 时间趋势（分组、95% 区间）与分组箱线；另出黑白印刷版 |
| `psm` / `psm_did` | 倾向得分匹配（1:k、卡尺、有放回）、平衡性 SMD、bootstrap ATT；匹配后再跑 TWFE |
| `rdd` | 断点回归（CCT 带宽、偏误校正、robust CI；报表用 robust p） |
| `synth` | 合成控制（自定义 QP 解权重、placebo 排名 p） |
| `heckman` | 两步法（probit 选择方程 + 逆米尔斯比），判断是否存在选择偏差 |
| `gmm` | 差分 GMM（Arellano-Bond 两步、Hansen J、AR(1)/AR(2)） |
| `threshold` | 面板门槛（网格搜索 + Hansen 自助法检验门槛显著性） |
| `spatial` | 空间滞后/误差模型（SAR / SEM，需提供权重矩阵 csv） |
| `mixed` | 多层/混合效应模型（随机截距/斜率、ICC） |
| `sem` | 结构方程（lavaan 风格模型串，CFI/RMSEA/SRMR） |
| `did` | 动态效应、预趋势 Wald；交错处理自动给 CS 估计量与 2x2 配对诊断 |
| `describe` / `corr` / `vif` | 描述统计（含分位）、相关矩阵（自动出热图）、共线性 |
| `density` | 偏度、峰度、尾部占比 + 分布图 |
| `spec_curve` | 同一系数在多组控制变量 / 固定效应 / 推断口径下的分布 |
| `subsample` | 按年、按行业或任意分组重跑主回归 |
| `quantile` | 分位数回归（固定效应用组内去均值近似） |
| `multitest` | BH / Holm / Bonferroni，报校正前后的显著条数 |
| `leave_one_out` / `placebo` | 留一法、时间安慰剂 |
| `confound` | 窗口内混杂事件筛查 |
| `pileup` | 阈值处密度不连续 |
| `reliability` | 双编码信度（Krippendorff alpha） |

## Robustness

```yaml
- leave_one_out: {var: car_0_5, by: zero_div}
- placebo: {var: car_0_5, by: zero_div, n: 10, gap: 20}   # 自动取事件日前的非事件锚点
```

安慰剂输出 z =（事件日差 - 锚点均值）/ 锚点标准差，|z| < 1.96 判 `warn`。留一法给最敏感单点的位移占原差的比例，超过 20% 判 `warn`。

## Regression

```yaml
tests:
  - reg:
      id: h2
      y: car_m3_3
      x: [disclosed_asset, log_amount, "log_amount:lev_w"]  # 支持交互项
      controls: [[size], [size, lev_w, roa_w]]              # 多层阶梯
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
  column_map: csmar          # 可选：CSMAR 导出表列名映射
  scale: 100                 # 可选：数据是小数时换算成百分点
  merge:                     # 可选：依次并入外部表（财务、治理、行业）
    - with: data/fin.csv
      on: [code, year]
      how: left
      keep: [roa, lev]
      validate: many_to_one
event:
  mode: single               # single | column | list
  date: "2024-04-12"
returns:
  method: market_model       # raw | market_adjusted | market_model
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
  - did: {id: policy, y: y, cohort: cohort, period: year, idvar: firm, window: [-4, 6]}
  - spec_curve: {id: sc, y: y, x: lnx, controls: [[], [size]], fe: [[], [year]], vcov: [hetero]}
  - multitest: {id: mt, method: bh}
```

组间差异检验支持就地做稳健性，不必另写脚本：

```yaml
- group_diff: {var: car_0_5, by: zero_div, id: base}
- group_diff: {var: car_0_5, by: zero_div, id: winsor, winsor: [0.01, 0.99]}
- group_diff: {var: car_0_5, by: zero_div, id: ex_big, drop_where: "lncap > 25"}
```

`car_mode` 与主检验口径（Welch / 等方差）、缩尾区间、剔除条件、推断口径都会写进每条结果，换口径不会静默。

收益率单位在读数时就判（小数 0.008 vs 百分点 0.8，差 100 倍但不报错）：判定结果与 `data.scale` 对不上会进 `unit_audit` 告警。

## Test

```
python -m pytest tests/ -q
```

`tests/test_*.py` 覆盖各方法的已知答案检验（真参数能否恢复、诊断能否报警）。

论文级的复现基线（`tests/test_repro_*.py` 与对应的 studies spec）依赖本地私有的论文数据，
不随仓库分发；本机设好数据路径后照常运行。

## Status

条件：pandas 3.x / Python 3.12。

作者：[@Northseacaviar](https://github.com/Northseacaviar)
