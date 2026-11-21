# MedCourse Colab Ultra 备用运行指南

本目录用于在**自有服务器显卡占满**且**任务完全不涉及真实患者/医院数据**时，临时借助 **Google Colab Ultra（A100 / L4）** 快速执行公开或虚构合成实验。

## 一、使用边界（合规声明）

1. **允许在 Colab 上运行的任务**：
   - **路线 B 扩展与复现**（`route_b.build_kb_agentic`）：输入仅含 `public_data/route_b_sources/` 中的公开卫健委临床路径及《抗菌药物临床应用指导原则》文本；
   - **全套代码单测与框架联调**：`scripts/run_all_tests.py` 与 `fixtures_mc.stats()` 虚构夹具合成演示。
2. **禁止上传或接入 Colab 的数据**：
   - 任何去标识或原始病历语料（如 `deid_run9*`）、`S2` 真实抽取事件（`events*.jsonl`）、医院派生统计表（`stats_v*.json`）及受控折数据（`fold 8 / fold 9`）。`colab_bootstrap.py` 内置路径命名拦截保护。

---

## 二、Colab Ultra 推荐配置

- **Runtime Type**：`A100 GPU`（40GB / 80GB VRAM，推荐）或 `L4 GPU`（24GB VRAM）+ 开启 **`High-RAM`**；
- **Background Execution**：Colab Ultra 支持后台持续运行，建议开启 Google Drive 挂载：
  - 将 HuggingFace 模型缓存指向 `/content/drive/MyDrive/MedCourse_Ultra_Cache/hf`，重连时免去重复下载 16GB 权重；
  - 将路线 B 输出目录指向 `/content/drive/MyDrive/MedCourse_Ultra_Cache/route_b_out`，防止实例超时回收丢进度。

---

## 三、主要接口速查（`colab/colab_bootstrap.py`）

| 函数 | 用途 |
|---|---|
| `inspect_gpu()` | 检测挂载显卡型号与显存余量，核对是否符合 A100/L4 要求（兼容 `Qwen3-32B-AWQ` 与 `Qwen3-8B`） |
| `setup_env()` | 自动配置 `BENCH_G1~G6`、`MC_CORE` 与 `PYTHONPATH` |
| `start_vllm(model="Qwen/Qwen3-32B-AWQ", port=8101, ...)` | 在 `127.0.0.1` 后台拉起兼容 `medcourse.llm.Client` 的 vLLM 服务（自动适配 AWQ `dtype="auto"`） |
| `run_route_b(out_dir, only="", concurrency=6)` | 运行路线 B 公开临床路径知识抽取（推荐配合 `Qwen/Qwen3-8B`） |
| `run_kb_only_demo(out_dir, n=4, use_llm=False)` | 基于 `kb_v22.json` 跑通 `--kb-only` 纯公开路线 B 国标 7 类文书合成演示（推荐配合 `Qwen/Qwen3-32B-AWQ`） |
| `run_synthetic_fixture_demo(out_dir, n=4, use_llm=False)` | 基于纯虚构统计夹具跑通 `medcourse_core` 7 类连续文书合成流水线 |
| `stop_vllm(handle)` | 停止后台 vLLM 进程并释放显存 |
