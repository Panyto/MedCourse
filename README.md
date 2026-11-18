# MedCourse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panyto/MedCourse/blob/main/colab/MedCourse_Colab_Ultra.ipynb)

**MedCourse** 是一个面向**整次住院全周期连续中文病历合成**与**临床路径多轮检索建表**的模块化工程框架，覆盖入院记录（`HR`）、首次病程（`FP`）、上级查房（`SUP`）、日常病程（`DAILY`）、术前小结（`PRE`）、术后首次病程（`POST`）及出院记录（`DS`）7 类连续文书的轨迹规划、按日可见性编译、受控成文、确定性门控与独立评测。

> **隐私与合规边界声明**：本开源仓库仅包含**纯代码模块、确定性临床规则表、虚构单测夹具、公开卫健委临床路径纯文本以及 Colab 运行工具**；**不包含**任何医院原始或去标识住院病历（`deid_run9`）、真实抽取事件或统计参数表（`stats_v*.json`）及受控评估结果。涉真数据抽取、统计建模与私域微调须在课题组自有受控离线服务器执行。

---

## 一、仓库目录结构

```text
MedCourse/
├── packages/
│   ├── medcourse_core/         # 主线核心引擎 (v0.2.1): 统计建表/轨迹规划/按日可见编译/门控/S2抽取
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
- `medcourse_core`：186 项单元测试（涵盖 `mc-stats-0.21` 动态表回退、Copula 化验校准、双时间轴 `compiler`、节级/篇级 `gate`、两趟抽取 `extract`）
- `route_b_kit`：43 项单元测试（动态规划章节切分、去病名 BM25、`DECIDE -> ANSWER` 先置引用与 `provenance` 核验）
- `gold_kit`：11 项单元测试
- `gm_tools`：14 项单元测试
- `bench_core`：62 项单元测试（在安装 `scikit-learn` 的环境如 Colab 下运行全部端到端测试；纯标准库环境自动跳过需 `sklearn` 的 `TestEndToEnd`）

---

## 三、Google Colab Ultra 备用方案（公开/无隐私任务）

当自有受控服务器算力紧张且任务**无需接触任何真实医院数据**（如扩展公开临床路径知识库、`route_b_kit` 消融测试或虚构夹具端到端调试）时，可点击顶部 **`Open in Colab`** 徽章启动 [`MedCourse_Colab_Ultra.ipynb`](colab/MedCourse_Colab_Ultra.ipynb)：

1. Runtime 选择 **`A100 GPU` + `High-RAM`**（Colab Ultra）；
2. 可选挂载 Google Drive 持久化缓存 `Qwen/Qwen3-8B` 权重与输出；
3. 自动拉起绑定 `127.0.0.1:8101` 的 vLLM 服务并发跑通 `route_b.build_kb_agentic` 与虚构病历合成演示。详细说明见 [`colab/README_Colab.md`](colab/README_Colab.md)。
