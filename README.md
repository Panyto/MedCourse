# MedCourse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panyto/MedCourse/blob/main/colab/MedCourse_Colab_Ultra.ipynb)

**MedCourse** 是一个面向**整次住院全周期连续中文病历合成**与**临床路径多轮检索建表**的模块化工程框架，覆盖入院记录（`HR`）、首次病程（`FP`）、上级查房（`SUP`）、日常病程（`DAILY`）、术前小结（`PRE`）、术后首次病程（`POST`）及出院记录（`DS`）7 类连续文书的轨迹规划、按日可见性编译、受控成文、确定性门控与独立评测。

**核心方法定位**：
1. **双源先验互补分工**：**路线 A（$k \ge 11$ 受控聚合统计）**承担全院疾病谱、多病共存、高斯 `Copula` 化验极端尾部、检查回报延迟及真实早期诊疗异质性底座（在 17 病种 `D0-1` 非既往史统一文本探针下，`Route-B` `--kb-only` 典型化程度极高，`Exams Only AUROC = 0.8259`，而融合路线 A 的 `Route-AB` 回归至 `0.5741`，更接近真实医院对照 `0.6773`）；**路线 B（`route_b_kit v0.5`，`kb_v22.json`）**承担 `--kb-only` 100% 零私有数据冷启动可复现基准、$k < 11$ 儿科与专科细分路径长尾补全，以及 17 标杆病种先验消融对照。
2. **上游状态编译决定纵向演进上限、轻量化门控守住安全与数值底线**：轨迹规划（$s_t$）配合双时间轴可见性与显式诊断断言编译（$b_{t,k}$），在**固定临床规则、固定 2,000 条轨迹且未经任何 `gate.py` 重写的 `First-Try` (`attempt=0`) 阶段**，即独立消除了 **`97.98%`** 的未来结果剧透（`OracleLeak 99.2% -> Full First-Try 2.0%`），并将证据后诊断漏更新率由 `VisOnly` 的 **`95.44%`**（确诊 `89.34%` / 排除 `100.0%`）骤降至 **`5.61%`**（确诊 `6.56%` / 排除 `4.91%`，贡献了全部状态更新增益，而单靠 `gate.py` 重试在 `VisOnly` 上减漏为 `0.00pp`）。针对大模型自审仍高漏检严重用药、常识错配及并行数值互换的短板，闭合式规则白名单与确定性 `gate.py`（`29` 项 `HARD_SECTION_FAILS` + `item_parallel_ambiguous` + `RETRY_GUIDANCE` 定向重写）聚焦拦截尾部表面越界、药典极值与跨篇一致性。
3. **规则边界与透明披露**：手工临床能力表与排除白名单已**冻结至 `v0.3.7`**，严格作为输入事实临床相容性前置过滤器而非核心算法增益来源；在严格执行“无直接检验或偶发佐证的院内合并症候选一律删弃”原则下，预规划账本（`3,020` 臂例次）中约 `82.04%` 缺直接依据候选被剔除，清晰界定了当前合成器覆盖现行冻结规则可支撑的纵向演化范围。

> **隐私与合规边界声明**：本开源仓库仅包含**纯代码模块、确定性临床规则表、虚构单测夹具、公开卫健委临床路径纯文本以及 Colab 运行工具**；**不包含**任何医院原始或去标识住院病历（`deid_run9`）、真实抽取事件或统计参数表（`stats_v*.json`）、留出折（`fold 8 / fold 9`）及受控评估结果。涉真数据抽取、统计建模与下游检验须在课题组自有受控离线服务器执行。

---

## 一、仓库目录结构

```text
MedCourse/
├── packages/
│   ├── medcourse_core/         # 主线核心引擎 (冻结 v0.3.7 / mc-contract-0.2.1): 统计与路线 B kb_v22 / 闭合检查能力白名单与严格排除表 / 双时间轴与显式状态编译 / S4前缀非预期性 / "分别为"一对一数值门控 / S2抽取
│   ├── s5_kit/                 # S5/S6 评测、审计与机制拆解套件: 17病种统一探针 / T1-T3 线性与神经(bge-small-zh)两阶段 TSTR / 零信任语义台账审计 / 4格 First-Try vs Post-Gate 机制拆解 / 阳性对照校准 MIA & 盲评打包
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
    ├── run_all_tests.py        # 全仓 MANIFEST + 6 套单元测试一键运行脚本
    └── sync_from_workspace.py  # 工作区同步与公开仓库安全门禁脚本
```

---

## 二、快速验证与本地测试

在仓库根目录执行以下命令，可一键校验全部 **13 个代码包**的 `MANIFEST.sha256`、公开仓库安全扫描以及全套单元测试（测试均使用内置虚构夹具，不依赖真实病历或外部 GPU）：

```bash
python scripts/run_all_tests.py
```

各模块包均保留原生目录结构与 `MANIFEST.sha256` 校验文件：
- `medcourse_core`（`v0.3.7`，冻结版 `s1-pipeline-0.3.7` / `mc-contract-0.2.1`）：**230 项单元测试**（涵盖 `mc-stats-0.21` 动态表回退、路线 B `kb_v22.json` 协同与 `--kb-only` 冷启动、`Copula` 化验校准、四类主诉状态机与闭合检查-诊断能力表 `_POS_CONFIRM_CAPABILITY_RULES`、亚部位内镜/影像/病理护栏、严格阴性排除与院内合并症方向/术式硬约束、双时间轴与同日 `start_hm <= stop_hm` 编译、S4 反事实前缀非预期性收口、`--ablate-epistemic` 与 `--save-first-try` 对称认知消融开关、`gate.py` 数值单位与 `"分别为"` 一对一顺序配对 + `item_parallel_ambiguous` 阻断，及 `TestV036` / `TestV037ReviewerBCounterexamples` 全部回归反例）
- `s5_kit`：**18 项单元测试**（17 病种 `--kb-only` 预规划校验、全量 `TemplateWriter` `Full`/`VisOnly`/`OracleLeak` 同源一致性、`T1/T2a/T2b/T3` 线性与 `tiny_bert` 神经探针两阶段冻结与 `fold 9` 守卫、`SMOKE10` 失败重启防跳过指纹印章、零信任语义台账 `build_v037_ledgers.py` 故障注入 5 项测试，以及 `First-Try` vs `Post-Gate` 四格核心机制归因拆解；在无 `sklearn`/`torch` 的纯标准库环境自动跳过需相应依赖的 4 项集成测试）
- `route_b_kit`：43 项单元测试（动态规划章节切分、去病名 BM25、`DECIDE -> ANSWER` 先置引用与 `provenance` 核验）
- `gold_kit`：11 项单元测试
- `gm_tools`：14 项单元测试
- `bench_core`：62 项单元测试（在安装 `scikit-learn` 的环境如 Colab 下运行全部端到端测试；纯标准库环境自动跳过需 `sklearn` 的 `TestEndToEnd`）

### `medcourse_core v0.3.2 -> v0.3.7` 与 `s5_kit` 核心演进摘要

- **闭合能力规则收口、并列数值门控与零信任审计（`v0.3.5 -> v0.3.7` 规则冻结 + `s5_kit`）**：
  - **严格阳性/阴性证据闭环**：彻底移除 `planner.py` 末尾基于粗粒度器官/模态的默认放行兜底，以闭合表 `_POS_CONFIRM_CAPABILITY_RULES`、`site_ok` 与亚部位关键词阻断越界检查确诊；阴性排除仅认可匹配 `_STRICT_NEGATIVE_EXCL_RULES` 的专项阴性检查；院内新增合并症必须绑定严格检验方向、有限偶发形态学发现或性别+产科/器官匹配术式，否则删弃候选并记入 `comorb_ledger`（于 `3,020` 臂例次跨章透明披露 `82.04%` 候补删弃界限）；冻结 `mc-contract-0.2.1` 全量校验与日内 `HH:MM` 单调排序。
  - **并行数值门控防漏放**：升级 `gate.audit_item_numbers`，取消 `"分别为"` 并列句静默跳过，执行一对一顺序配对（数值互换立即拦为 `item_number_mismatch`、混入旧值拦为 `item_stale_value`，歧义或不配对句式触发 `item_parallel_ambiguous` 并进入重写指引）。
  - **机制归因与零信任审计固化**：[`build_v037_ledgers.py`](packages/s5_kit/build_v037_ledgers.py) 不轻信预置标记，独立按目标诊断与引用 `evidence_ids` 重核相容性（在 `3,020` 臂例次规划 trajectories 上实现契约与语义规则违例 `0/0`）；[`analyze_mechanism_contribution.py`](packages/s5_kit/analyze_mechanism_contribution.py) 在固定规则与同一批规划轨迹上，将上游显式状态编译（`First-Try`）与门控重试（`Post-Gate`）解耦。

- **临床常识封口与无微调 `32B` 主线定型（`v0.3.4`）**：合入无切口/产科早孕未娩/新生儿与儿童剂量/`Z51` 肿瘤专科与裸影像 8 项临床常识护栏（单测至 `218/218`），经受控服务器实测 `BLIND10` `10/10`、`P80` 成活 `78/80`（`oracle2 O/A/B = 0/0/0`）；基于单院 LoRA 微调在 `S3` 暴露的低频片段复现（`254/254`, `stay_1nn_cos AUC = 0.9682`），确立 **`Qwen3-32B-AWQ` + 国家卫健委病历规范免微调成文**并将单院 LoRA 转为 `S6-1` 阳性标尺，导出经 `8,980` 参考库 30-gram 严格筛查（留存 `45.46%`、`after_g30=0`）使成员推断 `combined AUC` 稳定在 `0.4844`。

- **S4 反事实前缀非预期性与对称认知消融（`v0.3.3` -> `S5-1` `N=2,000`）**：收紧 `compiler.course_focus` 长程摘要与 `copy_banned` 停药/症状前缀可观测边界，通过 `--ablate-epistemic` 与 `--save-first-try` 对称对照证实：下游神经学生模型（`BAAI/bge-small-zh-v1.5`）在去静默 `T2b` 更新检出上，`Synth-Full (FirstTry)` 即大胜 `VisOnly (FirstTry)` **`+0.3905 (p=0.003)`**（`PostGate +0.4428, p=0.001`；统一真实文本标签 `2x2` `Full|text_rule` 仍胜 `VisOnly|text_rule` `+0.3937, p=0.001`），等量 `Matched N=600` 在 `T1a` 上大胜同底座无规划 `DIRECT-32B` **`+0.1945 (p<0.001)`**。

- **跨院通用国标成文与路线 B 融合（`v0.3.0 - v0.3.2`）**：接通 17 病种公开临床路径表 `medcourse/data/kb_v22.json` 与 `build_b_only_stats`（`route_b_kit` 187 格金标可用率 `86.1%`、引文覆盖 `93.6%`）；关闭正文句级硬拼贴（`splice_copy=False`），改用前置延续注入与 `RETRY_GUIDANCE` 错误码重采样，核心模块通过词表与共享谓词实现大幅瘦身。

---

## 三、Google Colab Ultra 备用方案（公开/无隐私任务）

当自有受控服务器算力紧张且任务**无需接触任何真实医院数据**（如扩展公开临床路径知识库、`route_b_kit` 消融测试、`--kb-only` 纯公开病历合成演示或虚构夹具端到端调试）时，可点击顶部 **`Open in Colab`** 徽章启动 [`MedCourse_Colab_Ultra.ipynb`](colab/MedCourse_Colab_Ultra.ipynb)：

1. Runtime 选择 **`A100 GPU` + `High-RAM`**（或 `L4 GPU 24GB`，Colab Ultra）；
2. 可选挂载 Google Drive 持久化缓存模型权重与输出；
3. 自动拉起绑定 `127.0.0.1:8101` 的 vLLM 服务（`Qwen/Qwen3-32B-AWQ` 适配 `v0.3.2` 国标合成，`Qwen/Qwen3-8B` 适配路线 B 建表），支持 `run_route_b`、`run_kb_only_demo`（`--kb-only` 17 病种公开路径合成）及 `run_synthetic_fixture_demo`。详细说明见 [`colab/README_Colab.md`](colab/README_Colab.md)。
