"""H6-2"撒词不看人"：语料层按配额往病历里加的词，和这个病人对不对得上。

做法是**同一份病历里比**：语料层之后的病历，每一处实体标成"插进来的"或"本来就有的"，两类用同一套规则查冲突、
查有没有细节，看插进来的是不是明显更差。这样写法、长短、病种都一样，只差"是不是配额塞的"。

插进来的 = 实体落在语料层前后逐段比对新插入或换掉的文字里（`q45_kit.diffs.inserted`），而且语料层之前这一段里没有这个写法。

冲突规则（每一处实体，只看没被否定的）：
- DRUG：ruler_g3 的药物规则（年龄、妊娠、性别，g3-rules-0.2 原样）命中这一处药名；
- SEX-ENT：只属于男性的部位或操作（前列腺、睾丸……）写在女性病历里，或只属于女性的（子宫、卵巢、妊娠……）写在男性病历里；
- PED-ADULT：14 岁以下写了只会出现在成人身上的东西（冠脉造影、PCI、前列腺、绝经……）；
- NEG-X（只计数、不进主读数）：这一处是肯定的，而同一份病历另一段把同一个实体否定了（"无发热"与"发热"），
  可能是病程前后的正常变化，总经理逐条读。
细节（只对检查和药）：检查后面有结果（ruler_g4 的口径：同一句、25 字内有结果词）、药有剂量（ruler_g3 的剂量读法）。
"""
from q45_kit.diffs import inserted
from ruler_g3.adapters import care_get, from_care7
from ruler_g3.engine import evaluate
from ruler_g3.parse import parse_age, parse_sex
from ruler_g4.ruler import info_units

from .entities import FIELDS4, mentions

MALE_ONLY = ("前列腺", "睾丸", "阴囊", "精索", "包皮", "阴茎")
FEMALE_ONLY = ("子宫", "卵巢", "阴道", "宫颈", "输卵管", "附件", "妊娠", "孕", "月经", "产后", "剖宫产", "绝经", "妇科")
ADULT_ONLY = ("冠脉造影", "冠状动脉造影", "PCI", "PTCA", "CABG", "冠脉支架", "支架植入", "前列腺", "绝经", "剖宫产",
              "妊娠")
CHILD_AGE = 14
SEG_OF = {"cc": "CC", "hpi": "HPI", "hc": "COURSE", "di": "DISCHARGE"}
MAIN_RULES = ("DRUG", "SEX-ENT", "PED-ADULT")


def _overlap(a, b, spans):
    return any(a < e and s < b for s, e in spans)


def label_mentions(before, after):
    """after 里每一处实体：段、实体、是否插进来的。返回 [dict]。"""
    out = []
    for f in FIELDS4:
        b, a = care_get(before, f), care_get(after, f)
        spans = inserted(b, a) if a != b else []
        for m in mentions(a):
            ins = bool(spans) and _overlap(m["start"], m["end"], spans) and m["surface"] not in b
            out.append({**m, "field": f, "inserted": int(ins)})
    return out


def _details(record, ms):
    """检查有没有结果、药有没有剂量：给 ms 里每一处加 detail（1 有、0 没有、None 不适用）。"""
    exam_spans = {f: [(u["start"], u["end"]) for u in info_units(care_get(record, f)) if u["type"] == "EXAM"]
                  for f in FIELDS4}
    rec = from_care7(record, "b6")
    res = evaluate(rec)
    dose_spans = {}
    for m in res["mentions"]:
        dose_spans.setdefault(m["seg"], []).append((m["start"], m["end"], m["dose"] is not None))
    for m in ms:
        if m["cat"] in ("AUX", "LAB"):
            m["detail"] = int(_overlap(m["start"], m["end"], exam_spans[m["field"]]))
        elif m["cat"] == "MED":
            ds = [d for s, e, d in dose_spans.get(SEG_OF[m["field"]], []) if s < m["end"] and m["start"] < e]
            m["detail"] = int(any(ds)) if ds else None
        else:
            m["detail"] = None
    return res


def conflicts(record, ms):
    """给 ms（label_mentions 的输出）里每一处加 rules：[规则号]。返回 ms。"""
    sex = parse_sex(care_get(record, "sex"))
    age = parse_age(care_get(record, "age"))
    res = _details(record, ms)
    drug_hits = {}
    for h in res["hits"]:
        if h["category"] in ("AGE", "PREG", "SEX"):
            drug_hits.setdefault(h["seg"], []).append((h["start"], h["end"], h["rule_id"]))
    negated = {}
    for m in ms:
        if m["neg"]:
            negated.setdefault(m["key"], set()).add(m["field"])
    for m in ms:
        rules = []
        if not m["neg"]:
            s = m["surface"]
            if sex == "F" and any(w in s for w in MALE_ONLY):
                rules.append("SEX-ENT")
            if sex == "M" and any(w in s for w in FEMALE_ONLY):
                rules.append("SEX-ENT")
            if age is not None and age < CHILD_AGE and any(w in s for w in ADULT_ONLY):
                rules.append("PED-ADULT")
            if m["cat"] == "MED":
                for a, b, rid in drug_hits.get(SEG_OF[m["field"]], []):
                    if a < m["end"] and m["start"] < b:
                        rules.append("DRUG")
                        m.setdefault("drug_rules", []).append(rid)
            if negated.get(m["key"], set()) - {m["field"]}:
                rules.append("NEG-X")
        m["rules"] = rules
    return ms


def summarize(ms):
    """插进来的、本来就有的两类：实体数、主规则冲突数（DRUG、SEX-ENT、PED-ADULT 任一）、NEG-X 数、
    检查和药里有细节的比例的分子分母。"""
    out = {}
    for lab, flag in (("inserted", 1), ("native", 0)):
        rows = [m for m in ms if m["inserted"] == flag and not m["neg"]]
        det = [m for m in rows if m.get("detail") is not None]
        out[lab] = {
            "n": len(rows),
            "conflict": sum(1 for m in rows if any(r in MAIN_RULES for r in m["rules"])),
            "negx": sum(1 for m in rows if "NEG-X" in m["rules"]),
            "detail_n": len(det),
            "detail_yes": sum(m["detail"] for m in det),
            "by_rule": {r: sum(1 for m in rows if r in m["rules"]) for r in MAIN_RULES + ("NEG-X",)},
        }
    return out
