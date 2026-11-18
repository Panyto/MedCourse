"""从归档的计数表生成 data/anchors.csv。只读 tables/ 里的计数表。总经理在本地跑一次，产物进核心包，伙计不用跑。

python tools/make_anchors.py 归档根目录 输出.csv
"""
import csv
import io
import os
import sys

DET = (1e-6, 1e-6)      # 确定性规则：绝对 1e-6、相对 1e-6
C2 = (0.005, 0.0)       # 判别器：同版本 sklearn 应完全一致，留一点余地给版本差
NULL = (0.01, 0.0)


def rows(path):
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    return float(x) if x not in ("", None) else None


def main(root, out):
    g2 = os.path.join(root, "关2_跨文书时序一致", "3_结果", "结果包")
    g3p = os.path.join(g2, "G3_pkg1_20261025", "tables")
    g2p = os.path.join(g2, "G2_pkg1_20261025", "tables")
    q3 = os.path.join(root, "关3_临床正确与安全", "3_结果", "结果包", "Q3_pkg1_20261027", "tables")
    q45 = os.path.join(root, "关4关5_写全不注水与真实感", "3_结果", "结果包")
    q4a = os.path.join(q45, "Q45_pkg1_20261028", "tables")
    q4b = os.path.join(q45, "Q45b_pkg1b_20261028", "tables")
    q6 = os.path.join(root, "关6_多样性与长尾", "3_结果", "结果包", "Q6_pkg1_20261029", "tables")
    out_rows = []

    def add(source, row, reader, how, field, value, chk="", den="", n=None, d=None, tol=DET):
        if value is None:
            return
        out_rows.append({"anchor_id": f"A{len(out_rows) + 1:03d}", "source": source, "row": row, "reader": reader, "how": how, "field": field,
                         "chk": chk, "den": den, "value": value, "numerator": n, "denominator": d, "abs_tol": tol[0], "rel_tol": tol[1]})

    arms = ("G-D", "G-E2")
    for r in rows(os.path.join(g3p, "T44_tfidf.csv")):
        if r["row"] in ("R",) + arms:
            add("G3/T44", r["row"], "g2", "mean", "tfidf_adj", num(r["mean"]))
    for r in rows(os.path.join(g3p, "T41b_c2_parts_with_GE2.csv")):
        if r["analysis_set"] == "main" and r["arm"] in ("R",) + arms and r["row_type"] == "estimate_vs_R" and r["component"] != "C2":
            add("G3/T41b", r["arm"], "g2", "mean", r["component"], num(r["mean"]))
    for r in rows(os.path.join(g3p, "T45_ruler.csv")):
        if r["row"] != "G-E2":
            if r["row_type"] != "estimate" and r["reading"] in ("C1a", "C1b", "C1c", "C2", "B3") and r.get("paired_n") == "187":
                add("G3/T45", "R", "g2", "mean", r["reading"] + "_val", num(r["reference_mean"]))
            continue
        if r["row_type"] == "estimate":
            rid = r["reading"]
            if rid in ("S-any", "K-any4", "K3s"):
                continue
            if rid == "K-any5":
                add("G3/T45", "G-E2", "g2", "rate", "Kany_hit", num(r["rate"]), chk="Kany_chk", n=int(r["numerator"]), d=int(r["denominator"]))
            elif r["numerator"]:
                add("G3/T45", "G-E2", "g2", "rate", rid + "_hit", num(r["rate"]), chk=rid + "_chk", n=int(r["numerator"]), d=int(r["denominator"]))
            elif r["mean"]:
                add("G3/T45", "G-E2", "g2", "mean", rid + "_val", num(r["mean"]))
    seen = set()
    for r in rows(os.path.join(g2p, "T02_real_band_fold0.csv")):
        key = (r.get("row") or r.get("set") or "", r.get("reading"))
        if r.get("stratum", r.get("group")) != "all" or r.get("numerator") in ("", None) or r.get("reading") not in ("S1", "S2", "S3", "S4a", "S4b", "S5"):
            continue
        if key in seen or "full" not in r.values():
            continue
        seen.add(key)
        add("G2/T02", "R-full", "g2", "rate", r["reading"] + "_hit", num(r["numerator"]) / max(1, num(r["denominator"])), chk=r["reading"] + "_chk",
            n=int(r["numerator"]), d=int(r["denominator"]))
    # 关3：全部用 0.2 版（g3old）复现，0.3 的差异另出表
    g3map = {"D-hard": "any_hard", "D-severe": "any_severe", "PEDADULT": "any_pedadult", "SEX": "any_sex", "warning": "any_warning", "N1": "any_n1"}
    for r in rows(os.path.join(q3, "Q03_rows.csv")):
        row = {"R-full": "R-full", "CARe-S1": "CARe-S1"}.get(r["row"])
        if not row:
            continue
        if r["metric"] in g3map:
            add("Q3/Q03", row, "g3old", "rate", g3map[r["metric"]], num(r["estimate"]), n=int(r["numerator"]), d=int(r["denominator"]))
        elif r["metric"] == "mentions_per_record":
            add("Q3/Q03", row, "g3old", "mean", "n_mentions", num(r["estimate"]))
        elif r["metric"] == "with_dose_per_mention":
            add("Q3/Q03", row, "g3old", "pooled", "n_with_dose", num(r["estimate"]), den="n_mentions", n=int(r["numerator"]), d=int(r["denominator"]))
    for r in rows(os.path.join(q3, "Q11_g2_arms.csv")):
        if r["arm"] not in arms or r["analysis_set"] != "main":
            continue
        if r["metric"] in g3map:
            add("Q3/Q11", r["arm"], "g3old", "rate", g3map[r["metric"]], num(r["estimate"]), n=int(r["numerator"]), d=int(r["denominator"]))
        elif r["metric"] == "mentions_per_record":
            add("Q3/Q11", r["arm"], "g3old", "mean", "n_mentions", num(r["estimate"]))
        elif r["metric"] == "with_dose_per_mention":
            add("Q3/Q11", r["arm"], "g3old", "pooled", "n_with_dose", num(r["estimate"]), den="n_mentions")
    wanted = {"iu_distinct": "iu_distinct", "NUM": "NUM", "anchor_n": "anchor_n", "chars": "chars", "empty_share": "empty_share", "DRUG": "DRUG", "EXAM": "EXAM"}
    for r in rows(os.path.join(q4b, "Q4b_01_rows.csv")):
        if r["row"] in ("R7", "R7prime", "CARe-S1") and r["metric"] in wanted:
            add("Q45b/Q4b_01", r["row"], "g4", "mean", wanted[r["metric"]], num(r["estimate"]))
    for r in rows(os.path.join(q4a, "Q4_03_coverage.csv")):
        if r["item"].startswith("L9") and r["version"] == "base" and r["row"] in ("R7", "CARe-S1"):
            add("Q45/Q4_03", r["row"], "g4", "rate", "L9", num(r["estimate"]), n=int(r["numerator"]), d=int(r["denominator"]))
    tag = {"course": "course", "hr": "hr", "discharge": "ds", "fp": "fp"}
    field = {"empty_share": "empty", "iu_per_kchar": "iu_kchar", "copy5_prev": "copy5"}
    for r in rows(os.path.join(q4a, "Q4_09_g2_arms.csv")):
        rname = {"R-core": "R", "G-D": "G-D", "G-E2": "G-E2"}.get(r["row"])
        if rname and r["metric"] in field and r["doc_type"] in tag:
            add("Q45/Q4_09", rname, "g4old", "mean", f"{tag[r['doc_type']]}_{field[r['metric']]}", num(r["estimate"]))  # 这张表当年用 g4-rules-0.1
    for r in rows(os.path.join(q4b, "Q5b_03_c2st.csv")):
        if r["comparison"] == "M1eq" and r["row"] == "CARe-S1":
            add("Q45b/Q5b_03", "CARe-S1", "c2", "row", "auc", num(r["auc_mean"]), tol=C2)
        if r["comparison"] == "M1eq-len" and r["row"] == "CARe-S1":
            add("Q45b/Q5b_03", "CARe-S1", "c2", "row", "auc_length", num(r["auc_mean"]), tol=C2)
    for r in rows(os.path.join(q4b, "Q5b_04_null.csv")):
        vals = list(r.values())
        if vals[0] == "False" and vals[1] == "MEAN":
            add("Q45b/Q5b_04", "R7", "c2", "row", "auc_null", num(vals[2]), tol=NULL)
    for r in rows(os.path.join(q6, "Q6_32_H63a.csv")):
        add("Q6/Q6_32", "CARe-S1", "g6", "row", "jaccard", num(r["J_S1"]))
        add("Q6/Q6_32", "CARe-S1", "g6", "row", "jaccard_ref", num(r["J_R7"]))
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    by = {}
    for r in out_rows:
        by[r["row"]] = by.get(r["row"], 0) + 1
    print(len(out_rows), by)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
