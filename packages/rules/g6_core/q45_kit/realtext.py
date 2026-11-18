"""关5 H5-1 的训练集：fold 0 每次住院照 R7 的截法截成 7 字段（"R7-all"），只在内存里用。

截法与关1 G1-D 冻结代码 `g1d_r2_j32.py` 的 `r7_records()` 逐字相同：第一份入院记录取"主诉""现病史"两节，
第一份出院记录取"出院诊断""诊疗经过""出院医嘱"三节。区别只在性别、年龄：R7 取 G1-B 槽位表，这里每次住院都没有槽位，
改用关1 尺子 `extract_facts` 从入院记录表头读（与 G1-E 配额表 `g1e_quota_export.py` 的读法相同：年龄 < 1 岁不要）。

`split_sections`、`extract_facts` 由调用方传进来（就是 ys 上 `gates/G1C/internal/j32_runtime_20261017/ruler_g1_core`
里的 `ruler_g1`，与 `r7_records()` 用的是同一份），这样本包不用再带一份关1 尺子。
"""
import math

REASONS = ("ok", "missing_doc", "no_sex", "no_age", "age_lt1")


def _first_doc(docs, kind):
    return next(((i, d) for i, d in enumerate(docs) if d.get("doc_type") == kind), None)


def _first_section(split_sections, text, wanted):
    return next((part.text.strip() for part in split_sections(text) if part.name == wanted), "")


def care7_from_admission(adm, split_sections, extract_facts):
    """返回 (7 字段, 原因)。
    - 没有入院记录或出院记录：(None, "missing_doc")；
    - 否则 7 字段总是截出来；性别、年龄读不出来（或 <1 岁）时那一栏是 None，原因写 "no_sex"、"no_age"、"age_lt1"
      （先查性别）。只有原因是 "ok" 的才进年龄、性别模型的训练集；截法核对（任务书 3.1 第 3 条）不看原因，只比五段文字。
    Gender 是"男"或"女"，Age 是岁数的字符串（如 "67"、"1.5"）。"""
    docs = adm.get("docs") or []
    hr, ds = _first_doc(docs, "hospital_record"), _first_doc(docs, "discharge_summary")
    if hr is None or ds is None:
        return None, "missing_doc"
    hr_no, hr_doc = hr
    hr_text, ds_text = hr_doc.get("text") or "", ds[1].get("text") or ""
    facts, _ = extract_facts(hr_text, str(adm.get("admission_id")), hr_no, "hospital_record")
    sex_fact = next((f for f in facts if f.field == "header_sex"), None)
    age_fact = next((f for f in facts if f.field == "header_age"), None)
    sex = sex_fact.value_norm if sex_fact else None
    try:
        age = float(str(age_fact.value_norm).split("|", 1)[0]) if age_fact else None
    except (TypeError, ValueError):
        age = None
    if age is not None and not math.isfinite(age):
        age = None
    if sex not in ("男", "女"):
        reason = "no_sex"
    elif age is None:
        reason = "no_age"
    elif age < 1:
        reason = "age_lt1"
    else:
        reason = "ok"
    record = {
        "Gender": sex if sex in ("男", "女") else None,
        "Age": f"{age:g}" if age is not None and age >= 1 else None,
        "Primary Diagnosis": _first_section(split_sections, ds_text, "出院诊断"),
        "Chief Complaint": _first_section(split_sections, hr_text, "主诉"),
        "History of Present Illness": _first_section(split_sections, hr_text, "现病史"),
        "Hospital Course": _first_section(split_sections, ds_text, "诊疗经过"),
        "Discharge Instructions": _first_section(split_sections, ds_text, "出院医嘱"),
    }
    return record, reason
