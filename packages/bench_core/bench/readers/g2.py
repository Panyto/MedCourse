"""关2：g2_core 0.4 的全部读数、C2 三分量、相邻 tf-idf。7 字段行没有时间线，不进这里。

相邻 tf-idf 的词表和 IDF 只由参考行的病程拟合，口径同 G3 表 T44：配了 tfidf_reference（T44 用的真实 200 份）就用它，
没配退回 config["reference"]。参考行只用来拟合，不出读数。
"""
from ruler_g2 import READINGS, build_real, build_synth, core_set, evaluate_admission, k_any
from ruler_g2 import c2parts, similarity
from ruler_g2.timeline import SYNTH_TYPE


def build(x):
    if x.get("view") == "core":
        raw = x.get("raw_docs") or x["docs"]
        return core_set(build_real({"admission_id": x["id"], "docs": raw}))
    docs = x["docs"]
    if all(d.get("doc_type_cn") in SYNTH_TYPE and d.get("k") is not None and d.get("day_offset") is not None for d in docs):
        slot = {"slot_id": x["id"], "admit_date": (x.get("meta") or {}).get("admit_date")}
        one = [{"k": d["k"], "doc_type_cn": d["doc_type_cn"], "day": d["day_offset"], "text": d["text"],
                "role": d.get("role", "")} for d in docs]
        return build_synth(slot, one, f"synth:{x['row']}")
    return build_real({"admission_id": x["id"], "docs": docs})


def run(rows, config):
    adms = [(x, build(x)) for x in rows if x["kind"] == "admission"]
    fit_only = config.get("tfidf_reference_row")
    ref = [t for x, a in adms if x["row"] == (fit_only or config.get("reference")) and a is not None
           for t in similarity.progress_between(a)]
    model = similarity.RefTfidf(ref) if ref else None
    per = []
    for x, adm in adms:
        if adm is None or x["row"] == fit_only:
            continue
        res = evaluate_admission(adm)
        rs = res["readings"]
        row = {"row": x["row"], "id": x["id"]}
        for k in READINGS:
            r = rs[k]
            row[f"{k}_chk"], row[f"{k}_hit"] = bool(r["adm_checkable"]), bool(r["adm_conflict"])
            v = r.get("value")
            row[f"{k}_val"] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        chk, hit = k_any(res)
        row["Kany_chk"], row["Kany_hit"] = bool(chk), bool(hit)
        parts = c2parts.c2_components(adm)
        row.update({"exam_kinds": parts["exam_kinds"], "surgery": parts["surgery"], "dx_changes": parts["dx_changes"]})
        row["tfidf_adj"] = similarity.adjacent_tfidf(model, similarity.progress_between(adm)) if model else None
        per.append(row)
    return per, []
