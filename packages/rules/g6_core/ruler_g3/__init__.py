"""关3 临床正确与安全的尺子（g3-rules-0.2）。

只查有白纸黑字出处、不看具体病情也成立的硬错误：
- D：用药硬规则（年龄禁忌、妊娠禁忌、性别不符、剂量越界），规则表见 data/drug_rules.csv，出处见 data/sources.csv；
- N1：生理上不可能、量级写错的数值；
- 具体度：写了剂量的药名占比（拦"予抗感染治疗"一类含糊写法）。

入口：
- adapters.from_care7(record) / adapters.from_admission(adm) 把病历变成 Record；
- engine.evaluate(rec) 返回命中和计数。
"""
from .engine import evaluate, summarize
from .adapters import Record, from_care7, from_admission

RULES_VERSION = "g3-rules-0.2"
