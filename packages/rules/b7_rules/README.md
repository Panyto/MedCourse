# b7_rules（b7-rules-0.1）

语料级改写矛盾计数（任务书 G1-D 第 4 节）。由总经理编写和维护；伙计原样使用，不改代码。要改规则，写进 `proposals/`，由总经理批准后出新版本。

- 依赖：ruler_g1 核心 v0.4（g1-rules-0.4），放在 PYTHONPATH 上。
- 测试：`PYTHONPATH=.:<核心目录> python -m unittest -v tests/test_b7.py`，21 条虚构用例应全部通过。
- 入口：`check_record(record)`，record 是 7 字段 dict；返回命中列表（规则名、字段、字段内位置、命中值）。规则说明见 `b7_rules.py` 开头。
- 已知取舍：宁可漏抽，不可错抽。"N年后/N年内"、随访复查语境、家属句、孕产语境、否认/避免类语境一律跳过。
