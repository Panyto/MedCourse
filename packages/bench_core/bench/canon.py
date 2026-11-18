"""旧行格式 → 统一行。

统一行（内存和子进程管道里传，不落盘）：
  {"row": 行名, "id": 住院号或槽位号, "kind": "admission", "docs": [...], "raw_docs": [...], "meta": {...}, "view": ...}
  {"row": 行名, "id": 槽位号, "kind": "care7", "record": {...7 字段...}, "meta": {...}}

docs 每项至少有 doc_type、day_offset、text；旧合成行的 k、role、date、doc_type_cn 原样带着。
AdmissionJSONL（产品和 run 9 导出同格式）：{admission_id, docs:[{doc_type, day_offset, text, doc_type_cn?, k?, role?}], meta?}
"""
import datetime as dt
import json

from .safe import SafeError

CN_TO_EN = {"入院记录": "hospital_record", "出院记录": "discharge_summary"}
DOC_TYPES = ("hospital_record", "progress_note", "discharge_summary")


def row_id(obj):
    for k in ("admission_id", "slot_id", "id", "rid"):
        if obj.get(k) is not None:
            return str(obj[k])
    raise SafeError("line_without_id")


def _admit_from_dates(docs):
    for d in docs:
        date, day = d.get("date"), d.get("day_offset")
        if date and day is not None:
            try:
                return (dt.date.fromisoformat(str(date)[:10]) - dt.timedelta(days=int(day))).isoformat()
            except ValueError:
                continue
    return None


def admission_line(obj, row, view="full"):
    rid = row_id(obj)
    docs = []
    for i, x in enumerate(obj.get("docs") or []):
        d = dict(x)
        if not isinstance(d.get("text"), str):
            raise SafeError("doc_without_text", row=row, id=rid, doc=i)
        if not d.get("doc_type") or d["doc_type"] not in DOC_TYPES:
            cn = d.get("doc_type_cn")
            if not cn:
                raise SafeError("doc_without_type", row=row, id=rid, doc=i)
            d["doc_type"] = CN_TO_EN.get(cn, "progress_note")
        if d.get("day_offset") is None and d.get("day") is not None:
            d["day_offset"] = d["day"]
        docs.append(d)
    meta = dict(obj.get("meta") or {})
    if obj.get("admit_no") is not None:
        meta["admit_no"] = obj["admit_no"]
    if obj.get("split_fold") is not None:
        meta["split_fold"] = obj["split_fold"]
    if not meta.get("admit_date"):
        meta["admit_date"] = obj.get("admit_date") or _admit_from_dates(docs)
    return {"row": row, "id": rid, "kind": "admission", "docs": docs, "meta": meta, "view": view}


CARE_KEYS = ("Primary Diagnosis", "Chief Complaint", "History of Present Illness", "Hospital Course",
             "Discharge Instructions")


def care7_line(obj, row, record_key="record"):
    rid = row_id(obj)
    rec = obj.get(record_key) if isinstance(obj.get(record_key), dict) else obj
    if not any(k in rec for k in CARE_KEYS + ("主诉", "现病史", "dx", "cc")):
        raise SafeError("care7_without_fields", row=row, id=rid)
    return {"row": row, "id": rid, "kind": "care7", "record": rec, "meta": dict(obj.get("meta") or {})}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    raise SafeError("bad_json", path_line=n) from None


def load_row(spec):
    """spec：rows.json 里的一行。kind=admission 或 care7；filter_ids 是一列住院号或槽位号的文件；
    id_map 是 csv（原 id 列, 新 id 列），比如把 admission_id 换成 slot_id 以便和合成行配对。"""
    name, kind = spec["row"], spec.get("kind", "admission")
    keep = None
    if spec.get("filter_ids"):
        with open(spec["filter_ids"], encoding="utf-8") as f:
            keep = {x.strip() for x in f if x.strip()}
    rename = {}
    if spec.get("id_map"):
        import csv
        with open(spec["id_map"], encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rename[str(r[spec.get("id_from", "admission_id")])] = str(r[spec.get("id_to", "slot_id")])
    out, seen = [], set()
    for obj in read_jsonl(spec["path"]):
        rid = row_id(obj)
        if keep is not None and rid not in keep:
            continue
        if kind == "care7":
            line = care7_line(obj, name, spec.get("record_key", "record"))
        else:
            line = admission_line(obj, name, spec.get("view", "full"))
        if rename:
            if rid not in rename:
                continue
            line["id"] = rename[rid]
        if line["id"] in seen:
            raise SafeError("duplicate_id", row=name, id=line["id"])
        seen.add(line["id"])
        out.append(line)
    if spec.get("n_expected") is not None and len(out) != spec["n_expected"]:
        raise SafeError("row_size_mismatch", row=name, got=len(out), expected=spec["n_expected"])
    return out


def apply_core_view(lines):
    """view=core：用 ruler_g2.core_set 选核心集。docs 换成选中的文书并带 role；raw_docs 留给关2 读数自己重建。
    缺入院记录、首次病程或出院记录的住院 core_set 返回 None，这里整行丢掉，并计数返回。"""
    from ruler_g2 import build_real, core_set
    out, dropped = [], 0
    for x in lines:
        if x.get("view") != "core":
            out.append(x)
            continue
        adm = core_set(build_real({"admission_id": x["id"], "docs": x["docs"]}))
        if adm is None:
            dropped += 1
            continue
        chosen = []
        for d in adm.docs:
            one = dict(x["docs"][d.idx])
            one["role"] = d.role
            chosen.append(one)
        y = dict(x)
        y["raw_docs"], y["docs"] = x["docs"], chosen
        out.append(y)
    return out, dropped
