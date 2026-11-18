"""把其他格式的病历渲染成尺子能读的带标题文本。由总经理维护。

render_summary7(record)：LLM-CARe 论文的 7 字段 JSON（英文键或中文键均可）→ doc_type="summary7" 的文本。
"""
import json
import re

_KEYS = {
    "sex": ("Gender", "gender", "Sex", "性别"),
    "age": ("Age", "age", "年龄"),
    "dx": ("Primary Diagnosis", "primary diagnosis", "Diagnosis", "主要诊断", "出院诊断"),
    "cc": ("Chief Complaint", "chief complaint", "主诉"),
    "hpi": ("History of Present Illness", "history of present illness", "现病史"),
    "hc": ("Hospital Course", "hospital course", "住院经过", "诊疗经过"),
    "di": ("Discharge Instructions", "discharge instructions", "出院医嘱"),
}
_SEX = {"男": "男", "男性": "男", "male": "男", "m": "男", "女": "女", "女性": "女", "female": "女", "f": "女"}
_AGE = re.compile(r"(\d{1,3})\s*(岁|个月|月|天|日)?")


def _get(record, field):
    for k in _KEYS[field]:
        if k in record and record[k] is not None:
            v = record[k]
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            return str(v).strip()
    return ""


def _flat(s):
    # 正文里的换行改成空格，避免模型自己写的小标题被当成段落标题
    return re.sub(r"\s*\n\s*", " ", s)


def render_summary7(record):
    sex_raw = _get(record, "sex")
    sex = _SEX.get(sex_raw.strip().lower(), _SEX.get(sex_raw.strip(), sex_raw))
    age_raw = _get(record, "age")
    m = _AGE.search(age_raw)
    age = f"{m.group(1)}{m.group(2) or '岁'}" if m else age_raw
    parts = [
        f"性别：{sex}",
        f"年龄：{age}",
        f"出院诊断：{_flat(_get(record, 'dx'))}",
        f"主诉：{_flat(_get(record, 'cc'))}",
        f"现病史：{_flat(_get(record, 'hpi'))}",
        f"住院经过：{_flat(_get(record, 'hc'))}",
        f"出院医嘱：{_flat(_get(record, 'di'))}",
    ]
    return "\n".join(parts) + "\n"
