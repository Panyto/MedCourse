# MedCourse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panyto/MedCourse/blob/main/colab/MedCourse_Colab_Ultra.ipynb)

**MedCourse** 是一个面向**整次住院全周期连续中文病历合成**与**临床路径多轮检索建表**的模块化工程框架，覆盖入院记录（`HR`）、首次病程（`FP`）、上级查房（`SUP`）、日常病程（`DAILY`）、术前小结（`PRE`）、术后首次病程（`POST`）及出院记录（`DS`）7 类连续文书的轨迹规划、按日可见性编译、受控成文、确定性门控与独立评测。

> **隐私与合规边界声明**：本开源仓库仅包含**纯代码模块、确定性临床规则表、虚构单测夹具、公开卫健委临床路径纯文本以及 Colab 运行工具**；**不包含**任何医院原始或去标识住院病历（`deid_run9`）、真实抽取事件或统计参数表（`stats_v*.json`）及受控评估结果。涉真数据抽取、统计建模与私域微调须在课题组自有受控离线服务器执行。

---

## 一、仓库目录结构

```text
MedCourse/
├── packages/
│   ├── medcourse_core/         # 主线核心引擎 (v0.3.4): 统计与路线B kb_v22/四类主诉与临床常识护栏/S4前缀非预期与认知消融/轻量化门控/S2抽取
│   ├── route_b_kit/            # 路线 B 核心包 (v0.5): 开源 8B 多轮检索式临床路径建表与引文核验
│   ├── bench_core/             # 独立评测台 (S0 v0.2): 六维度量化指标、负对照与记分牌聚合
│   ├── gm_tools/               # 基线对比工具与对照实验脚本 (S1)
│   ├── gold_kit/               # 事件抽取金标分层抽样、局部标注服务与评分工具
│   └── rules/                  # 独立冻结的各关卡确定性临床规则包
│       ├── ruler_g1_core/      # 关1：单篇事实一致性规则
│       ├── b7_rules/           # 关1：常识与临床逻辑校验规则
│       ├── g2_core/            # 关2：跨文书时序与标准检查词表
│       ├── g3_core/            # 关3：药典与生理极值规则 (0.3)
│       ├── g3_core_v02/        # 关3：历史基准兼容版 (0.2, 供 bench_core 对比)
│       ├── g45_core/           # 关4/5：信息单元、真实感与 C2ST 特征工具
│       └── g6_core/            # 关6：多样性、长尾分布、Vendi 与留出词表覆盖
├── public_data/
│   └── route_b_sources/        # 路线 B 公开中文临床路径与《抗菌药物临床应用指导原则》文本 (29 文件)
├── colab/
│   ├── MedCourse_Colab_Ultra.ipynb   # Google Colab Ultra (A100/L4) 一键运行 Notebook
│   ├── colab_bootstrap.py            # GPU 检查、127.0.0.1 vLLM 启动、路线 B 与虚构演示封装
│   └── README_Colab.md               # Colab Ultra 启用条件与操作说明
└── scripts/
    ├── run_all_tests.py        # 全仓 MANIFEST + 5 套单元测试一键运行脚本
    └── sync_from_workspace.py  # 工作区同步与公开仓库安全门禁脚本
```

---

## 二、快速验证与本地测试

在仓库根目录执行以下命令，可一键校验全部 12 个代码包的 `MANIFEST.sha256`、公开仓库安全扫描以及全套单元测试（测试均使用内置虚构夹具，不依赖真实病历或外部 GPU）：

```bash
python scripts/run_all_tests.py
```

各模块包均保留原生目录结构与 `MANIFEST.sha256` 校验文件：
- `medcourse_core`（`v0.3.4`，冻结版 `s1-pipeline-0.3.4`）：**218 项单元测试**（涵盖 `mc-stats-0.21` 动态表回退、路线 B `kb_v22.json` 协同与 `--kb-only` 冷启动、`Copula` 化验校准、四类主诉状态机与诊断性影像耦合、双时间轴与同日术前 `k < post_k` 编译、S4 反事实前缀非预期性收口、`--ablate-epistemic` 与 `--save-first-try` 对称认知消融开关、无切口/产科早孕未娩/新生儿与儿童剂量/`Z51` 肿瘤专科与裸影像 8 项临床常识护栏、轻量化节级/篇级 `gate`、两趟抽取 `extract`）
- `route_b_kit`：43 项单元测试（动态规划章节切分、去病名 BM25、`DECIDE -> ANSWER` 先置引用与 `provenance` 核验）
- `gold_kit`：11 项单元测试
- `gm_tools`：14 项单元测试
- `bench_core`：62 项单元测试（在安装 `scikit-learn` 的环境如 Colab 下运行全部端到端测试；纯标准库环境自动跳过需 `sklearn` 的 `TestEndToEnd`）

### `medcourse_core v0.3.2 -> v0.3.4` 核心演进摘要

- **临床常识封口与 S5/S6 前生成端冻结（`v0.3.4`）**：在 `icd.py`/`drug_class.py`/`kb.py`/`planner.py`/`compiler.py`/`gate.py`/`writer.py` 合入 8 项临床常识护栏（非手术无切口全域贯穿、妊娠 `<14w`/`14-19w`/`>=20w` 分段与 `<18w` 禁胎动/未娩禁产褥、`脊柱`/`咽喉` 映射与模态感知部位消解裸影像、`takes_tissue` 排除碎石/取石/内固定/剖宫产/修补与治疗性入院防错配病理、新生儿 `3-27 天` 定标与扩展儿科按体重激素/止吐/解热镇痛极量、`Z51` 肿瘤专科与抗肿瘤适应证对齐、`COURSE` 诊断事件 `must=True` 保留与 `_CARRY_ORDER_WORDS` 去泛化粘连，并从查体模板移除“主诉”以清零 `qa_adm_pe_history`），新增 `TestV034ClinicalAntiHallucination` 6 项单测（合计 **`218/218`** 通过）。内网受控服务器 `ys` `Qwen3-32B-AWQ` 复核：`BLIND10` `10/10`、`P80` 自由模式成活 `78/80`（`llm_errors=0`，`gate/b7/g3` 全 `0`，`oracle2 O/A/B = 0/0/0`，8 项 `v034` 专项 `0/78`、`qa_adm_pe_history` 由 `17/255` 降至 `0/78`）；自 `v0.3.4` 起正式冻结生成端以启动 `fold 9` S5/S6 独立终验。

- **S4 反事实前缀非预期性与对称认知消融（`v0.3.3`）**：收紧 `compiler.course_focus` 长程摘要与 `copy_banned(j, day)` 的停药/症状前缀可观测边界，并在 `TestV033` 固化反事实单测；新增 `--ablate-epistemic`（保持可见已回报、已开待回检查及计划手术/医嘱历史逐字段一致，仅剥离显式认知状态提示与 `unsettled_dx` 门控）与 `--save-first-try`（留存 `attempt=0` 首轮成文快照）。在 `--kb-only` `N=60`（`Qwen3-32B-AWQ`）对称对照中，显式认知状态在 `First-Try` 阶段即将证据到位后诊断遗漏更新由 **89.7%（26/29）降至 48.3%（14/29，确诊漏更新 18/21 → 7/21）**、双窗严格对齐率由 **10.3% 提升至 51.7%（+41.4pp）**、前窗宽松过早断言由 **28.2% 降至 23.1%**（仅增 `+0.42%` 调用，1,431 vs 1,425 次）；内网受控服务器 `v0.3.3` `32B` 端到端实测 `BLIND10` `10/10` 零泄露、`A256` 成活 `255/256`（`oracle2 O/A/B = 1/0/0`，`g3/b7` 全零违规）。
- **跨院通用国标成文与路线 B 融合**：接通 17 病种公开临床路径表 `medcourse/data/kb_v22.json` 与 `build_b_only_stats`（`--kb-only` 零私有数据冷启动）；成文提示全面对齐国家卫健委《病历书写基本规范》（支持 `Qwen/Qwen3-32B-AWQ` 免微调通用高质合成，无需单院文风拟合），关闭句级硬拼贴（`splice_copy=False`），改用前置延续注入与错误码重采样（`RETRY_GUIDANCE`）。
- **四类主诉状态机与诊断影像耦合**：区分 `scheduled`/`obstetric`/`lesion`/`acute` 四类主诉演化，扩展产科孕周与儿科 CKD/贫血/按体重极量校验，确保器官明确的病例生成靶器官阳性诊断影像并钳位外科术前回报时序（`diag_img_v032` 仅以器官系统与既有检查为输入，不以 `surgical` 标志触发）。
- **时点与临床门禁收口 + 正则瘦身**：补齐手术当日术前边界（`day == op_d0 and k < post_k`）、经自然腔道/介入无切口约束（`no_incision_leak`）、面板级异常化验断言作用域及 `DS` 入院日期显式锚定；通过词表 `_has_any`、纯 Python `_cjk5` 前缀抽取与统一删句器 `_drop_clauses_if`，核心 4 模块（`planner.py`/`gate.py`/`compiler.py`/`kb.py`）`re.*` 调用由 157 降至 **85 处（`-45.9%`）**。

---

## 三、Google Colab Ultra 备用方案（公开/无隐私任务）

当自有受控服务器算力紧张且任务**无需接触任何真实医院数据**（如扩展公开临床路径知识库、`route_b_kit` 消融测试、`--kb-only` 纯公开病历合成演示或虚构夹具端到端调试）时，可点击顶部 **`Open in Colab`** 徽章启动 [`MedCourse_Colab_Ultra.ipynb`](colab/MedCourse_Colab_Ultra.ipynb)：

1. Runtime 选择 **`A100 GPU` + `High-RAM`**（或 `L4 GPU 24GB`，Colab Ultra）；
2. 可选挂载 Google Drive 持久化缓存模型权重与输出；
3. 自动拉起绑定 `127.0.0.1:8101` 的 vLLM 服务（`Qwen/Qwen3-32B-AWQ` 适配 `v0.3.2` 国标合成，`Qwen/Qwen3-8B` 适配路线 B 建表），支持 `run_route_b`、`run_kb_only_demo`（`--kb-only` 17 病种公开路径合成）及 `run_synthetic_fixture_demo`。详细说明见 [`colab/README_Colab.md`](colab/README_Colab.md)。
