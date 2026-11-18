# bench_core v0.2（S0 统一评测台）

2026-11-01，总经理。一条命令跑全部关卡：输入 `AdmissionJSONL` 或 7 字段行，输出记分牌、负对照矩阵、基线复现表。代码之外只依赖已封存的规则包，规则一条没重写。

## v0.2 改了什么

依据 `S0_pkg1_r3`、`S0_pkg2_serve_part1_r2` 和 `PREREG_S0_v2.md`。v0.1 的首跑结果留档，不覆盖。

| 地方 | v0.1 | v0.2 |
|---|---|---|
| `registry.py` | A 组只标了 `A_date_order` 主打，和预登记"主打指标（A、B 组）"不一致 | A 组 19 项全标主打 |
| `readers/extra.py` | 文书头只看前 40 字，真实行分母 0 | 前 40 字读法不变；40–120 字只认书写时间戳写法，带字段名或介词的日期不认 |
| `date_diag.py` | 无 | 新增。跑记分牌前确认读到的是每篇自己的书写日 |
| `negcontrols.py` | 七种 | 加 `NC-dxflip`（诊断反复横跳），`B_dxchg` 的定义不动 |
| `aggregate.py` | 判据 (c) 只认严格改进，双侧门线不判 | 相等也算不差；双侧门线看离带中点的距离；A 组的标记只列表；新增 `S0_neg_paired.csv` |
| `readers/g4old.py`、`data/anchors.csv` | 关4 旧锚点对着 0.2 读 | Q45/Q4_09 的 33 条走 `g4-rules-0.1`，数字一个没改 |
| `readers/g2.py`、`run.py` | tf-idf 词表用 187 份拟合 | 配 `tfidf_reference` 后用 T44 当年的真实 200 份，只拟合不出读数 |
| `rvr.py` | 科室名原样配对，0 对；选过的住院没有去重 | 三层科室名和计数诊断；组间不重叠；`--dept-only`；`MIN_PAIRS` 不降 |
| `perf_probe.py` | schema 探针用改写提示词、512 token | 抽取提示词、默认 2048；记录 `finish_*`、`json_ok_per_s`、`mean_events` |

## 依赖

| 环境变量 | 指向 | 版本 |
|---|---|---|
| `BENCH_G1` | `ruler_g1_core` | 0.5 |
| `BENCH_B7` | `b7_rules` | 0.1 |
| `BENCH_G2` | `g2_core` | 0.4 |
| `BENCH_G3` | `g3_core` | **0.3** |
| `BENCH_G3_OLD` | `g3_core` | 0.2（只用于复现旧锚点） |
| `BENCH_G45` | `g45_core` | 0.2 |
| `BENCH_G45_OLD` | `g45_core` | **0.1（只用于复现 Q45/Q4_09）** |
| `BENCH_G6` | `g6_core` | 0.1 |

Python 3.10，判别器、Vendi、MAUVE 需要 numpy、scikit-learn；其他只用标准库。

自检（在本目录，环境变量设好后）：`python -B -m unittest discover -s tests -v`，**62 条**。没有 scikit-learn 时跳过判别器相关的条目，不设环境变量时端到端部分跳过。测试数据全部虚构。

## 为什么用子进程

`g45_core`、`g6_core` 各带一份 `ruler_g3` 0.2，`g2_core` 带一份 `ruler_g1`。同一个进程里 import 会撞车，所以 `bench.run` 读完所有行后，每个读数起一个子进程，`PYTHONPATH` 只放这个读数需要的包（`bench/paths.py`），行走 stdin，不落盘。**负对照里的真实派生文本因此只在内存。** 读数结果写盘前过 `safe.numeric_only`：只许数字、布尔、标识字段。

## 数据契约

`AdmissionJSONL`（产品和 run 9 导出同格式）：

```
{"admission_id": "...", "docs": [{"doc_type": "hospital_record|progress_note|discharge_summary",
                                   "day_offset": 0, "text": "...", "doc_type_cn": "...", "k": 0, "role": "HR"}],
 "meta": {"admit_date": "YYYY-MM-DD"}}
```

`doc_type_cn`、`k`、`role` 可选。旧合成行（`slot_id`、`day`、`date`）`bench/canon.py` 自动转。7 字段行：`{"slot_id": ..., "record": {...}}`。

`view`：`core` 用 `ruler_g2.core_set` 选核心集（R、产品行）；`full` 全部文书；旧合成行本来就是选好的，用 `full`，并且带 `doc_type_cn`、`k` 时关2 走 `build_synth`，和当年一样。

## 用法

```
python -B -m bench.date_diag --row R-full=导出文件 --real R-full --out tables
python -B -m bench.mauve --rows rows_S0.json --out out
python -B -m bench.run --rows rows_S0.json --out out --neg R,G-E2
python -B -m bench.anchors --out out
python -B -m bench.g3diff --out out --changed-ids work/g3_changed.csv
```

| 模块 | 产物 |
|---|---|
| `bench.date_diag` | `S0_date_diag.csv`：读出比例、规则分布、位置分布、和日程的一致率。D1–D3 不过退出码 3 |
| `bench.run` | `tables/S0_scoreboard.csv`（每行每指标：值、区间、分子分母、门线、过线与否、可用状态）；`S0_neg_matrix.csv`、`S0_neg_summary.csv`、`S0_neg_paired.csv`；`S0_rows.csv`、`S0_readers.csv`；`raw/`（逐住院数字） |
| `bench.anchors` | `S0_anchor_check.csv`、`S0_anchor_summary.csv`：132 条归档定稿数字逐条对 |
| `bench.g3diff` | `S0_g3_diff.csv`：规则 0.2 到 0.3，每行每类硬错误减少、增加多少住院 |
| `bench.mauve` | `raw/all.mauve.row.jsonl`：MAUVE（标准算法，字符特征；`--embed-model` 换本机嵌入） |
| `bench.rvr` | 小实验 1：R 对 R 的判别下限；`S0_rvr_dept_diag.csv`：科室配对计数 |
| `bench.copyops` | 小实验 2：相邻病程的拷贝操作分布 |
| `bench.profile_rows` | 小实验 3：真实住院画像 |
| `bench.s4b_recount` | 手术日注入重数 |
| `bench.sealcheck` | 分折封存表 |
| `bench/perf_probe.py` | 算力探针，单文件，可以拷走用；抽取用 schema 在 `data/extract_schema*.json` |

## 指标

`bench/registry.py` 是唯一一张表，门线抄自产品工程计划 v2 第 6、7 节。状态四种：可用、只作诊断（G1-4、K1–K5、B2，精确率不够，不进闸门）、只报告（Vendi、MAUVE、Self-BLEU、L9）、待接入（年龄读者、人盲测、真未来泄露、32B 完整性，写明在哪一阶段接）。

新增一个读数：`date_viol`（按日程排好的病程，文书头日期有没有倒退）。打乱型负对照不一定动 tf-idf，要靠它拒绝。真实行的日程可能就是从这个日期推出来的，那样真实行的值只说明读对了，有效的检验是负对照和合成行（`S0_date_diag.csv` 的 `tautology_flag`）。

## 负对照

`bench/negcontrols.py`：固定模板、恒定输出、打乱病程、截断、重复段、计划转储、张冠李戴、诊断反复横跳。判据见 `PREREG_S0_v2.md` 第 4 节，比计划 v2 原来的"每个指标都失败"收紧了说法：硬冲突率对模板本来就是 0，要求每个指标都失败是做不到的假判据。

## 已知的取舍

- 负对照经 `build_real` 从正文标题认文书类型，所以只有带标题的正文才能走核心集；这是有意的，产品行也是这条路。
- `NC-dxflip` 用固定诊断词，不解析真实诊断；它只用来打破诊断变化步数，不代表别的指标也该被它拒绝。
- `copyops` 句子切分不切逗号，句子 4 字以下不计。
- `s4b_recount` 用手术名字符 2-gram 重合 ≥0.5 判"是不是同一台手术"，粗但透明，认不出单列一类，不猜。
- 科室大科靠关键词归并，认不出的留原名单独成一类，不猜。
- MAUVE 字符特征和嵌入特征不可比，报告时写明用的哪种。
- schema 探针的 2048 和 20 条上限只用来量产能，不是产品抽取契约；契约在 S1 定。
