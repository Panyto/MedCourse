"""关4：7 字段读数（整次住院按 realtext 投影），加整次住院的逐篇读数（病程空句、拷贝）。"""
from q45_kit.realtext import care7_from_admission
from ruler_g1 import extract_facts, split_sections
from ruler_g4 import analyze_admission, analyze_care7

TYPES = ("NUM", "DRUG", "EXAM", "SYM", "DX", "PROC")
L9 = "L9_出院带药均有剂量"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run(rows, config):
    per = []
    for x in rows:
        row = {"row": x["row"], "id": x["id"]}
        if x["kind"] == "care7":
            rec = x["record"]
        else:
            adm = {"admission_id": x["id"], "docs": x["docs"]}
            rec, _ = care7_from_admission(adm, split_sections, extract_facts)
            docs = analyze_admission(adm)
            for role, tag in (("course", "course"), ("hr", "hr"), ("discharge", "ds"), ("fp", "fp")):
                part = [d for d in docs if d["role"] == role]
                row[f"{tag}_empty"] = _mean(d["empty_share"] for d in part)
                row[f"{tag}_iu_kchar"] = _mean(d["iu_per_kchar"] for d in part)
                row[f"{tag}_copy5"] = _mean(d["copy5_prev"] for d in part)
        if rec is None:
            row["proj_ok"] = False
            per.append(row)
            continue
        a = analyze_care7(rec)
        row.update({"proj_ok": True, "chars": a["chars"], "iu_distinct": a["iu_distinct"],
                    "iu_per_kchar": a["iu_per_kchar"], "empty_share": a["empty_share"], "dup_share": a["dup_share"],
                    "on_list": a["on_list"], "off_list": a["off_list"], "anchor_n": a["anchor_n"],
                    "date_n": a["date_n"], "relday_n": a["relday_n"], "L9": bool(a["coverage"][L9])})
        row.update({t: a["iu_by_type"][t] for t in TYPES})
        per.append(row)
    return per, []
