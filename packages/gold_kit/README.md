# gold_kit v0.3（S2 抽取金标）

2026-11-06，总经理。按 `PREREG_S2G_v4.md` 抽题、起本机标注页、汇总，另加纠偏的开发检查。只用标准库。抽取代码、类型和状态的定义都从 `medcourse_core` v0.1.11.1 读（抽取 v0.3）。v0.2 配 v0.1.10、v0.1.10.1，留档。

**v0.3 相对 v0.2（S2G3 之后）：**
- **抽题（`sample.py`）：** A 部分另抽一层名称回退（`stratum=fallback`，20 条）：抽取 v0.3 引文对不上、按名称找回来的七类事件。先抽这一层，按类型抽时不重复。每题带 `stratum`。
- **汇总（`summarize.py`）：** 名称回退层单算成立和状态（`fallback:*`），不进总体、按类型的读数和准入。
- **开发检查（`rescore.py`）：**
  - 默认按 v0.2 的题和标签算（S2G3 的 180 题），v1 的 300 题加 `--legacy-v1`。
  - 多出类型改对（`type_fixed`）：不成立、原因是类型错的题，纠偏正好改成老板选的那一类。
  - 多出逐条规则 × 类型的状态翻转（`up`、`down`、`side`）。
  - 多出各类改后读数（`after_*`，只作描述）。
- **标注页（`serve.py`）没改：** 定义从 `schema` 读，自动换成 D1–D4 的新写法。

自检 11 条（新增 2 条）。

**v0.2 相对 v0.1（金标 S2G v1 判"不可用"之后）：**
- 标注页的类型、状态的名字和定义，直接从 `medcourse.extract.schema` 读，和抽取提示词是同一份。v1 的提示词把体征算进症状，标注页只写"症状"，两边对不上。
- 只抽下游要用的事件（`contract.is_core`）。体征、七类以外、不是这个病人不出题；B 部分也只列七类事件，被错分到出口里的核心事件算漏抽。
- A 部分每类 20 题、共 140 题；B 部分 40 个切片。
- 不成立的题多问一句原因，类型错的再选应该是哪类。
- 判定按类型准入，没有总成立率底线。
- 新增 `gold/rescore.py`：把纠偏层套在 v1 的 300 题上用老板已有的标签重新计分，看误删多少（超过成立题的 5% 就停）。

| 文件 | 做什么 |
|---|---|
| `gold/sample.py` | 从纠偏后的 `events_norm.jsonl`、`done_segments.txt` 和原文抽题，写 `items.jsonl`（带原文切片，只能放 `MC_DERIVED_ROOT` 下），`sample_counts.csv` 只有计数（含分到出口的事件数） |
| `gold/serve.py` | 标注页：只绑 127.0.0.1，每个请求都要口令；不引外部资源，有内容安全策略；不记请求日志；只收短标签；说明里列出全部定义 |
| `gold/summarize.py` | 准确率、召回、状态混淆、不成立原因、类型错的去向、按类型准入，只有计数和类型名 |
| `gold/rescore.py` | 纠偏的开发检查，只有计数 |

运行时 `PYTHONPATH` 要同时有 `gold_kit` 和 `medcourse_core`，环境变量同 `medcourse_core`（`BENCH_G2` 等）。

自检：`MC_CORE=medcourse_core 目录 PYTHONPATH=gold_kit:medcourse_core python -B -m unittest discover -s tests`，9 条，全部虚构数据。
