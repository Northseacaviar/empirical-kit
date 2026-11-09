"""结果容器：每条结论都带 status、why 和 provenance。

status 三档，含义固定，agent 直接按它决定下一步：
- ok    : 结果可用，无需处理
- warn  : 结果可用但口径有风险（样本小、缺失多、聚类数少等），写论文时要说明
- fail  : 结果不可用（估计失败、样本不足、口径冲突），不能进论文
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Finding:
    id: str
    type: str
    status: str = "ok"
    why: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "status": self.status, "why": self.why,
                "data": self.data, "provenance": self.provenance, "warnings": self.warnings}


class Findings:
    def __init__(self, spec_name: str, spec_hash: str):
        self.spec_name = spec_name
        self.spec_hash = spec_hash
        self.started = datetime.now().isoformat(timespec="seconds")
        self.items: list[Finding] = []
        self.sample_flow: list[dict] = []
        self.meta: dict[str, Any] = {}

    def add(self, f: Finding) -> Finding:
        f.provenance.setdefault("spec", self.spec_name)
        f.provenance.setdefault("spec_hash", self.spec_hash)
        self.items.append(f)
        return f

    def summary(self) -> dict:
        counts = {"ok": 0, "warn": 0, "fail": 0}
        for f in self.items:
            counts[f.status] = counts.get(f.status, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "spec": {"name": self.spec_name, "hash": self.spec_hash},
            "run": {"started": self.started, "finished": datetime.now().isoformat(timespec="seconds")},
            "meta": self.meta,
            "sample_flow": self.sample_flow,
            "summary": self.summary(),
            "findings": [f.as_dict() for f in self.items],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p
