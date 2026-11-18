"""事实与冲突的数据结构。字段与任务书 G1-A 第 5.1 节一致。"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Fact:
    admission_id: str
    doc_seq: int
    doc_type: str
    section: str
    start: int
    end: int
    field: str
    value_norm: str
    extractor_rule_id: str

    def record(self):
        return asdict(self)


@dataclass(frozen=True)
class Conflict:
    pair_id: str
    left_section: str
    left_start: int
    left_end: int
    right_section: str
    right_start: int
    right_end: int
    conflict_axis: str  # 数值 / 侧别 / 有无

    def record(self):
        return asdict(self)
