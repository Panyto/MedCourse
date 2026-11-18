# gm_tools v0.1（gm-tools-0.1）

2026-11-10，总经理。M1 正式评测和 fold 2–7 S2 第一段抽取要用的几个小工具。

**不改核心包：**
- `medcourse_core` v0.1.13.1（M1 冻结版）、v0.1.11.1（S2 抽取）一个字节都不动；
- 这里的模块只 import 它们。

## 模块

| 模块 | 版本串 | 做什么 | 依赖 |
|---|---|---|---|
| `gmtools.m1_direct` | `m1-direct-0.1` | M1 第三组：同一写手，不给规划事实，直接写整套文书 | `medcourse` v0.1.13.1（`writer.SYSTEM`、`writer.clean`、`llm.WRITE`、`planner.seed_of`） |
| `gmtools.m1_patients` | `m1-patients-0.1` | M1 第三组病人层：让写手自己出 256 个病人头；和第一组的病人头并列计数 | 同上 |
| `gmtools.m1_table` | `m1-table-0.1` | 从 bench 记分牌、运行计数、泄露表出 M1 的 10 行表和胜负 | 只用标准库 |
| `gmtools.s2_stage1` | `s2-stage1-0.1` | fold 2–7 S2 第一段：只放行入院记录、出院记录的切片 | `medcourse.extract.run`（v0.1.11.1；v0.1.13.1 的 `extract/` 逐字节相同） |

### `m1_direct` 和第一组的差别

**只差一样：** 写手手里没有规划给的临床事实。其余照第一组，偏向第三组、保守：
- 同一批病人、同样的篇数、文书类型、住院日、程序写的抬头（第一组的 `admissions.jsonl`）；
- 同样的节目录；每节的目标字数取第一组那一节实际写出的字数；
- 同一个模型、同一句 SYSTEM、同一组采样参数和 `max_tokens` 算法、同一个 `clean`。

**多给两样：**
- 主要诊断（出院诊断名）和住院天数；
- 本次住院已写好的病历，最近 3,000 字（8101 的上下文是 8,192 token）。

**没有的东西：**
- 闸门、重写、拷贝都没有；
- 真未来泄露没有时间线可对，不适用。

**住院号前加 `D`。** 产物 `counts.json` 写 `mode: free`、`writer: llm`，`synth_export` 照常能用。

### `s2_stage1` 为什么这样做

- 包一层，`medcourse.extract.run` 一行不改。
- 文书序号和切片位置都是原文件里的，纠偏 `--source` 用原文件。
- 第二段直接跑原程序、原文件、同一输出目录，`done_segments.txt` 会跳过第一段做完的切片（自检里有这一条）。

## 自检

14 条，全部虚构数据、假客户端，不连服务。

```bash
export PYTHONPATH=<medcourse_core 目录>:<gm_tools 目录>    # 外加评测台环境变量，同 medcourse 自检
cd gm_tools && $PY -B -m unittest discover -s tests
```

本机 2026-11-10：配 v0.1.13.2 的 `medcourse_core` 跑，14 条全过。用到的 `writer.py`、`llm.py`、`safe.py`、`extract/run.py`、`extract/segments.py`，在 v0.1.11.1、v0.1.13.1 里逐字节相同。
