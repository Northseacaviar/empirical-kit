"""StudySpec：把口径写成对象。

设计约束：
- 样本、事件、窗口、基准只在这里定义一次，任何检验不得私自改口径。
- 所有字段都有可用的默认值；写 yaml 的人只需要写关心的部分。
- spec 的内容会被 hash 进每条结果，用来回答"这个数字是哪套口径跑出来的"。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class MergeSpec(BaseModel):
    """一条合并规则。语义对齐 Stata 的 merge：连接键 + 保留列 + 基数校验。

    为什么必须有 validate：左边是公司-年度面板、右边是年度行业表时，
    写成 many_to_one 才能发现「右表键重复」这种错——否则 pandas 会静默扇出，
    样本量翻倍而结果看似正常。
    """

    model_config = {"populate_by_name": True}

    source: str = Field(alias="with")
    on: list[str]
    how: Literal["left", "inner", "right", "outer"] = "left"
    keep: list[str] = Field(default_factory=list)
    rename: dict[str, str] = Field(default_factory=dict)
    # 内部叫 cardinality：pydantic 的 BaseModel 自带 validate 方法，同名会遮蔽并告警；
    # 对外仍写 validate，与 Stata/pandas 的习惯一致
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"] = \
        Field(default="many_to_one", alias="validate")
    suffix: str = "_y"
    min_match: float = 0.9



def _fix_yaml_keys(obj):
    """YAML 1.1 会把裸写的 on / off / yes / no 解析成布尔。

    merge 的连接键恰恰叫 `on: [code, year]`，不修的话键会变成 True，
    pydantic 报「on 字段缺失」而作者看不出为什么。
    """
    if isinstance(obj, dict):
        fixed = {}
        for k, v in obj.items():
            if k is True:
                k = "on"
            elif k is False:
                k = "off"
            fixed[k] = _fix_yaml_keys(v)
        return fixed
    if isinstance(obj, list):
        return [_fix_yaml_keys(x) for x in obj]
    return obj


class DataSpec(BaseModel):
    """数据来源。prices 可以是目录（每只证券一个 csv）或单个长表。"""

    prices: str | None = None
    prices_glob: str = "*.csv"
    # 已算好的分析面板：设置后跳过 CAR 计算，直接吃这张表
    panel: str | None = None
    index: str | None = None
    cross_section: str | None = None
    code_col: str = "code"
    date_col: str = "date"
    close_col: str = "close"
    return_col: str | None = None
    # 读入时保留的列（过滤、分组用），默认全留
    keep_cols: list[str] = Field(default_factory=list)
    encoding: str = "utf-8-sig"
    # 列名映射：'csmar' 走内置映射表；也可写自定义 {原名: 标准名} 的 dict
    column_map: str | None = None
    # 依次并入的外部表（财务、治理、行业等）
    merge: list[MergeSpec] = Field(default_factory=list)
    # 变量单位换算：CSMAR 的收益率是小数(-0.008)，自建面板常是百分点(-0.8)。
    # scale=100 表示「数据是小数、报告成百分点」。t 值不受影响。
    scale: float = 1.0
    # 只有这些前缀的变量按 scale 换算（如 ["car"]）。为空则作用于全部。
    scale_vars: list[str] = Field(default_factory=list)
    unit: str = ""


class EventSpec(BaseModel):
    """事件日。single=同一个日子；column=每行自己的事件日。"""

    mode: Literal["single", "column", "list"] = "single"
    date: str | None = None
    dates: list[str] = Field(default_factory=list)
    column: str | None = None
    # 事件日休市时锚点的取法：公告日当天收盘作为基准，是事件研究的常见做法
    anchor_rule: Literal["last_on_or_before", "first_on_or_after"] = "last_on_or_before"

    @model_validator(mode="after")
    def _check(self):
        if self.mode == "single" and not self.date:
            raise ValueError("event.mode=single 需要 event.date")
        if self.mode == "column" and not self.column:
            raise ValueError("event.mode=column 需要 event.column")
        if self.mode == "list" and not self.dates:
            raise ValueError("event.mode=list 需要 event.dates")
        return self


class ReturnsSpec(BaseModel):
    """收益与窗口口径。

    car_mode 决定窗口边界的含义，二者不可混用：
    - close_to_close：窗口 [a,b] = 从 T+a 收盘到 T+b 收盘的持有期收益。
      与多数论文"事件日收盘为基准"的写法一致；[0,5] 即 5 个交易日后。
    - sum_daily：窗口 [a,b] = T+a … T+b 各日日收益之和（标准 CAR[-1,1] 口径）。
    """

    method: Literal["raw", "market_adjusted", "market_model"] = "raw"
    car_mode: Literal["close_to_close", "sum_daily"] = "sum_daily"
    windows: list[list[int]] = Field(default_factory=lambda: [[-1, 1], [0, 3], [0, 5], [0, 10], [0, 20], [0, 30], [0, 40], [0, 60]])
    estimation_start: str | None = None
    estimation_end: str | None = None
    min_est_obs: int = 120
    min_window_obs: int = 1


class FilterSpec(BaseModel):
    """一条样本剔除规则。三种写法可叠加，reason 会进样本流表。"""

    name: str = ""
    drop_if: str | None = None          # pandas query，为真的行被剔除
    keep_if: str | None = None
    drop_flags: dict[str, Any] = Field(default_factory=dict)  # {isST: 1} 表示剔除该列等于该值的行
    require_nonnull: list[str] = Field(default_factory=list)
    reason: str = ""


class GroupSpec(BaseModel):
    """分组定义。expr 是布尔表达式；values 是按原值分组。"""

    name: str
    expr: str | None = None
    values: list[Any] | None = None
    labels: list[str] | None = None


class DeriveSpec(BaseModel):
    """派生变量：name + pandas 表达式。在过滤之前执行。"""

    name: str
    expr: str


class SampleSpec(BaseModel):
    derive: list[DeriveSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    groups: list[GroupSpec] = Field(default_factory=list)


class StudySpec(BaseModel):
    name: str
    data: DataSpec
    event: EventSpec | None = None
    returns: ReturnsSpec = Field(default_factory=ReturnsSpec)
    sample: SampleSpec = Field(default_factory=SampleSpec)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    output_dir: str = "output"

    def scale_for(self, var: str) -> float:
        if self.data.scale == 1.0:
            return 1.0
        if not self.data.scale_vars:
            return self.data.scale
        return self.data.scale if any(var.startswith(p) for p in self.data.scale_vars) else 1.0

    @model_validator(mode="after")
    def _need_event(self):
        if not self.data.panel and self.event is None:
            raise ValueError("非 panel 模式需要 event（事件日）")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "StudySpec":
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        spec = cls(**_fix_yaml_keys(raw))
        spec._source = str(p.resolve())
        return spec

    def hash(self) -> str:
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
