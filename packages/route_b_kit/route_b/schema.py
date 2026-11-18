# -*- coding: utf-8 -*-
"""kb_facts：一个病种一份，按公开路径名 + 版本 + ICD 做主键。每条事实是
{"value": …, "source": "pathway" | "guideline" | "model_prior", "section": 小节代号, "quote": 引文, "grounded": bool}。

字段分三组（定义只写在这里一处）：
- PATHWAY_FIELDS  路径原文预计有的：住院日、检查清单、手术日、麻醉、术后天数、抗菌药停药条件、出院标准、逐日表单
- GAP_FIELDS      路径原文预计没有的：检查回报天数、影像客观所见、严重度比例、个人史危险因素比例。
                  有出处就用，没有就标 model_prior，单独计数，不说成"知识库出处"
- SEED_FIELDS     不由路径给，取自种子（A 的 k≥11 统计或公开人口学）：年龄、性别、住院天数
"""
META_KEYS = ("pathway", "version", "icd10", "issuer", "url", "retrieved", "text_sha256")

PATHWAY_FIELDS = (
    "chief_symptoms",      # 诊断依据里的症状
    "signs",               # 诊断依据里的体征
    "los_max",             # 标准住院日（0.4.1 起 8B 版给 {min, max}，value_max 取上限；正则版 v0 是上限整数）
    "exams_required",      # 必需检查（每项另带 bench_key：映射到评测台 54 个检查键，映射不上为 None）
    "exams_optional",
    "workup_within_days",  # 入院后几天内完成检查
    "procedure",           # 术式
    "anesthesia",          # 麻醉方式
    "surgery_day_max",     # 手术日：入院第几天以内
    "postop_days",         # 术后住院恢复天数
    "antibiotic_rule",     # 抗菌药选择与停药条件
    "discharge_criteria",
    "variations",          # 变异及原因：非典型分支的来源
    "daily_form",          # 逐日表单：[(起日, 止日)] 及每列的诊疗工作、医嘱
)
GAP_FIELDS = ("exam_turnaround_days", "exam_findings", "severity_mix", "risk_factor_rates")
# 8B 提问检索版（kb_v1_agentic）另有的：危险因素和轻重分档的内容（比例另由公开流调和 A 的抽样值定）、术后切口引流观察、8B 自己提的追问
AGENTIC_FIELDS = ("risk_factors", "severity_levels", "postop_care", "extra_qa", "surgery_gate")
# 0.4.1：surgery_gate 是 8B 答的"做不做手术或有创操作"，只用来门控手术相关字段，不进 A、B 两组的格子
# 汇总分组（老板 11-11）：A 组是路径应有项，主看有出处率和检查键映射率，和正则 v0 正面比；
# B 组是路径未载、靠追问补的项，主看补全覆盖率（v0 几乎为 0）。两组分开合计，不混成一个分母
GROUP_A = ("los_max", "chief_symptoms", "signs", "exams_required", "exams_optional", "procedure", "anesthesia",
           "surgery_day_max", "postop_days", "antibiotic_rule", "discharge_criteria")
GROUP_B = ("exam_turnaround_days", "exam_findings", "risk_factors", "severity_levels", "postop_care", "extra_qa")
SEED_FIELDS = ("age", "sex", "los")
SOURCES = ("pathway", "guideline", "model_prior", "not_applicable")   # not_applicable：门控答"不做"，手术相关字段记"不适用"


def validate(kb):
    """结构校验，返回错误代码列表（不带正文）。"""
    errs = [f"meta_{k}" for k in META_KEYS if not kb.get("meta", {}).get(k)]
    facts = kb.get("facts", {})
    for k, items in facts.items():
        if k not in PATHWAY_FIELDS + GAP_FIELDS + AGENTIC_FIELDS:
            errs.append(f"unknown_field:{k}")
            continue
        for f in items if isinstance(items, list) else [items]:
            if f.get("source") not in SOURCES:
                errs.append(f"bad_source:{k}")
            if f.get("source") not in ("model_prior", "not_applicable") and not f.get("quote"):
                errs.append(f"no_quote:{k}")
            if "value" not in f:
                errs.append(f"no_value:{k}")
    return errs
