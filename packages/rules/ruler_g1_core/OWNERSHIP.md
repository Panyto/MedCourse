# 代码归属（自 G1-A v1.1 起）

| 文件 | 归属 | 伙计能做什么 |
|---|---|---|
| `ruler_g1/sections.py`、`ruler_g1/extractors.py`、`ruler_g1/compare.py`、`ruler_g1/models.py` | 总经理 | 只读。发现问题写成"修改提案"（见任务书 v1.1 第 4 节），不得直接改 |
| `ruler_g1/config/*.json` | 总经理 | 只读。可按取证结果提出增补建议 |
| `tests/test_core.py` | 总经理 | 只读。不得为了通过而改用例；失败如实报告 |
| `ruler_g1/safety.py`、数据库读取、批量运行、统计、扰动构造、打包、本地查看界面 | 伙计 | 正常开发，按守则 |

改动核心文件的唯一途径：伙计在 `proposals/` 下提交提案和对照计数，总经理批准后由总经理发新版本。
