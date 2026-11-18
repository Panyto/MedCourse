"""关1：G1-1…G1-6 逐篇判定（住院层取"任一篇"），b7 年龄对病史、儿童对成人内容。

整次住院先按 realtext 的截法投到 7 字段再跑 b7，7 字段行直接跑。AGE-UNPARSED 不算命中，记不可比。
"""
from b7_rules import check_record
from q45_kit.realtext import care7_from_admission
from ruler_g1 import PAIRS, evaluate_document, extract_facts, is_first_progress, render_summary7, split_sections


def _doc_type(d):
    if d["doc_type"] == "progress_note":
        return "first_progress" if is_first_progress(d["text"]) else "progress_note"
    return d["doc_type"]


def run(rows, config):
    per = []
    for x in rows:
        row = {"row": x["row"], "id": x["id"]}
        chk = {p: 0 for p in PAIRS}
        hit = {p: 0 for p in PAIRS}
        if x["kind"] == "care7":
            rec = x["record"]
            docs = [(render_summary7(rec), "summary7")]
        else:
            rec, _ = care7_from_admission({"admission_id": x["id"], "docs": x["docs"]}, split_sections, extract_facts)
            docs = [(d["text"], _doc_type(d)) for d in x["docs"]]
        for i, (text, kind) in enumerate(docs):
            res = evaluate_document(text, x["id"], i, kind)
            for p in PAIRS:
                chk[p] += res["checkable"].get(p, 0)
            for c in res["conflicts"]:
                hit[c.pair_id] = hit.get(c.pair_id, 0) + 1
        for p in PAIRS:
            key = p.replace("-", "_")
            row[f"{key}_chk"], row[f"{key}_hit"] = chk[p] > 0, hit[p] > 0
        if rec is None:
            row["b7_chk"], row["b7_hit"] = False, False
        else:
            hits = check_record(rec)
            unparsed = any(h["rule"] == "AGE-UNPARSED" for h in hits)
            row["b7_chk"], row["b7_hit"] = not unparsed, any(h["rule"] != "AGE-UNPARSED" for h in hits)
        per.append(row)
    return per, []
