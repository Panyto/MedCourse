# -*- coding: utf-8 -*-
"""消融对比：kb_v0（正则抽取，靶子）vs kb_v1_agentic（8B 提问检索）。

python -m route_b.compare_kb --v0 kb_v0 --v1 kb_v1_agentic --out compare_v0_v1.csv
每个字段比：抽到的病种数、有出处的病种数；8B 版另报模型先验数、引文不实数。
检查清单另比：项数、能落到评测台检查键的项数（v0 是程序按别名表映射，v1 是 8B 从候选键里选）。
"""
import argparse
import csv
import glob
import json
import os

FIELDS = ("los_max", "chief_symptoms", "signs", "exams_required", "exams_optional", "procedure", "anesthesia", "surgery_day_max",
          "postop_days", "antibiotic_rule", "discharge_criteria", "exam_turnaround_days", "exam_findings", "risk_factors",
          "severity_levels", "postop_care", "extra_qa")
NO_KEY = "无对应"


def load(d):
    return {os.path.basename(p)[3:-5]: json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(d, "kb_*.json")))}


def exam_stats(kbs, version):
    items = mapped = 0
    for kb in kbs.values():
        for f in kb["facts"].get("exams_required", []):
            if version == "v0":
                items += len(f["value"])
                mapped += sum(1 for b in f.get("bench_key") or [] if b)
            else:
                items += len(f["value"] or [])
                mapped += sum(1 for e in f["value"] or [] if e.get("key") and e["key"] != NO_KEY)
    return items, mapped


LIST_FIELDS = ("chief_symptoms", "signs", "exams_required", "exams_optional", "discharge_criteria", "risk_factors", "severity_levels", "postop_care")


def _n_items(value):
    if isinstance(value, dict):
        return len(value.get("watch") or [])
    if isinstance(value, list):
        return len(value)
    return 1 if value not in (None, "") else 0


def item_counts(facts, version):
    """列表型字段的逐项有出处（老板 11-10 提醒）：(项数, 有出处的项数)。
    v0 的项都是从引文里切出来的，整条有出处就全算；v1 用 items_in_quote（只对核对通过的引文数），8B 说原文没写的算 0。"""
    n = g = 0
    for f in facts:
        if version == "v1" and "items_in_quote" in f:
            g += f["items_in_quote"][0]
            n += f["items_in_quote"][1]
        else:
            k = _n_items(f.get("value"))
            n += k
            g += k if (version == "v0" and f.get("grounded")) else 0
    return n, g


def compare(v0, v1):
    ids = sorted(set(v0) & set(v1))
    rows = []
    for fld in FIELDS:
        f0 = [v0[i]["facts"].get(fld, []) for i in ids]
        f1 = [v1[i]["facts"].get(fld, []) for i in ids]
        row = {"field": fld, "diseases": len(ids),
               "v0_with_fact": sum(1 for x in f0 if x), "v0_grounded": sum(1 for x in f0 if any(f["grounded"] for f in x)),
               "v1_with_fact": sum(1 for x in f1 if x), "v1_grounded": sum(1 for x in f1 if any(f["grounded"] for f in x)),
               "v1_model_prior_facts": sum(1 for x in f1 for f in x if f["source"] == "model_prior" and not f.get("quote_failed")),
               "v1_quote_failed_facts": sum(1 for x in f1 for f in x if f.get("quote_failed")),
               "v1_not_applicable": sum(1 for x in f1 for f in x if f.get("source") == "not_applicable"),
               "v0_items": "", "v0_items_grounded": "", "v1_items": "", "v1_items_in_quote": "", "v1_item_rate": ""}
        if fld in LIST_FIELDS:
            n0, g0 = item_counts([f for x in f0 for f in x], "v0")
            n1, g1 = item_counts([f for x in f1 for f in x], "v1")
            row.update(v0_items=n0, v0_items_grounded=g0, v1_items=n1, v1_items_in_quote=g1, v1_item_rate=round(g1 / n1, 3) if n1 else "")
        rows.append(row)
    blank = {k: "" for k in rows[0]}
    # A、B 两组合计（老板 11-11）：按"病种 × 字段"的格子数，A 组看有出处，B 组看补全覆盖
    from .schema import GROUP_A, GROUP_B
    for name, flds in (("GROUP_A_路径应有项（格子）", GROUP_A), ("GROUP_B_追问补全项（格子）", GROUP_B)):
        def cells(kbs, pred):
            return sum(1 for i in ids for fld in flds if pred(kbs[i]["facts"].get(fld, [])))
        f1 = [f for i in ids for fld in flds for f in v1[i]["facts"].get(fld, [])]
        lf0 = [f for i in ids for fld in flds if fld in LIST_FIELDS for f in v0[i]["facts"].get(fld, [])]
        lf1 = [f for i in ids for fld in flds if fld in LIST_FIELDS for f in v1[i]["facts"].get(fld, [])]
        n0, g0 = item_counts(lf0, "v0")
        n1, g1 = item_counts(lf1, "v1")
        rows.append(dict(blank, field=name, diseases=len(ids) * len(flds),
                         v0_with_fact=cells(v0, bool), v0_grounded=cells(v0, lambda x: any(f["grounded"] for f in x)),
                         v1_with_fact=cells(v1, bool), v1_grounded=cells(v1, lambda x: any(f["grounded"] for f in x)),
                         v1_model_prior_facts=sum(1 for f in f1 if f["source"] == "model_prior" and not f.get("quote_failed")),
                         v1_quote_failed_facts=sum(1 for f in f1 if f.get("quote_failed")),
                         v1_not_applicable=sum(1 for f in f1 if f.get("source") == "not_applicable"),
                         v0_items=n0, v0_items_grounded=g0, v1_items=n1, v1_items_in_quote=g1, v1_item_rate=round(g1 / n1, 3) if n1 else ""))
    i0, m0 = exam_stats({i: v0[i] for i in ids}, "v0")
    i1, m1 = exam_stats({i: v1[i] for i in ids}, "v1")
    rows.append(dict(blank, field="exams_required_items→bench_key", diseases=len(ids), v0_with_fact=i0, v0_grounded=m0, v1_with_fact=i1, v1_grounded=m1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v0, v1 = load(a.v0), load(a.v1)
    rows = compare(v0, v1)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("compare_kb_done", rows[0]["diseases"])


if __name__ == "__main__":
    main()
