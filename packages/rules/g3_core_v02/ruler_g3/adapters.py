"""把两种病历变成同一个 Record：
- LLM-CARe 的 7 字段病历（关1 的 S1、S2、修复后、R7）；
- 整次住院（run 9 导出的真实住院、关2/G3 的合成住院）：docs 列表，每篇有 doc_type 或 doc_type_cn 或 role，和 text。
"""
import json
from dataclasses import dataclass, field

from .parse import parse_age_range, parse_sex, header_age_range, dx_sections

CARE_KEYS = {
    "sex": ("Gender", "gender", "Sex", "性别"),
    "age": ("Age", "age", "年龄"),
    "dx": ("Primary Diagnosis", "primary diagnosis", "Diagnosis", "主要诊断", "出院诊断"),
    "cc": ("Chief Complaint", "chief complaint", "主诉"),
    "hpi": ("History of Present Illness", "history of present illness", "现病史"),
    "hc": ("Hospital Course", "hospital course", "住院经过", "诊疗经过"),
    "di": ("Discharge Instructions", "discharge instructions", "出院医嘱"),
}


@dataclass
class Record:
    rid: str
    age: float = None          # 年龄下限（岁），年龄规则用它
    age_hi: float = None       # 年龄上限（岁），儿童剂量的体重用它；见 parse.parse_age_range
    sex: str = None
    dx_text: str = ""
    segs: list = field(default_factory=list)   # [(名字, 文字, 类别)]，类别 hr / course / discharge / other
    meta: dict = field(default_factory=dict)


def care_get(record, key):
    for k in CARE_KEYS[key]:
        if k in record and record[k] is not None:
            v = record[k]
            if isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            return str(v).strip()
    return ""


def from_care7(record, rid=""):
    lo, hi = parse_age_range(care_get(record, "age"))
    return Record(
        rid=rid,
        age=lo,
        age_hi=hi,
        sex=parse_sex(care_get(record, "sex")),
        dx_text=care_get(record, "dx"),
        segs=[("CC", care_get(record, "cc"), "other"),
              ("HPI", care_get(record, "hpi"), "hr"),
              ("COURSE", care_get(record, "hc"), "course"),
              ("DISCHARGE", care_get(record, "di"), "discharge")],
        meta={"format": "care7"},
    )


def doc_kind(doc):
    t = str(doc.get("doc_type") or "")
    cn = str(doc.get("doc_type_cn") or "")
    role = str(doc.get("role") or "")
    if t == "hospital_record" or cn.startswith("入院记录") or role == "HR":
        return "hr"
    if t == "discharge_summary" or cn.startswith("出院记录") or cn.startswith("死亡记录") or role in ("DS", "DIS"):
        return "discharge"
    return "course"


def from_admission(adm, rid=None):
    docs = adm.get("docs") or []
    segs, dx = [], []
    age = age_hi = sex = None
    for i, d in enumerate(docs):
        text = d.get("text") or ""
        kind = doc_kind(d)
        k = d.get("k", i)
        segs.append((f"{k}:{d.get('doc_type') or d.get('doc_type_cn') or d.get('role') or ''}", text, kind))
        if kind == "hr" and age is None and sex is None:
            age, age_hi, sex = header_age_range(text)
        dx.append(dx_sections(text))
    if age is None and sex is None and docs:
        age, age_hi, sex = header_age_range(docs[0].get("text") or "")
    rid = rid or str(adm.get("admission_id") or adm.get("slot_id") or "")
    return Record(rid=rid, age=age, age_hi=age_hi, sex=sex, dx_text="\n".join(x for x in dx if x), segs=segs,
                  meta={"format": "admission", "n_docs": len(docs)})
