"""CPU 小实验 3：真实住院的画像，给 L3 规划器定分布。只出计数和分位数。

每次住院文书数；各类型字数分位；住院天数；手术比例；缺首次病程比例；admit_no 分布。
导出文件没有患者号，只有 admit_no，"每患者入院数"拿不到：admit_no>1 只是"这个人此前住过院"的下界，
影响患者级 MIA 的隐私单元，写进 GAPS。

python -B -m bench.profile_rows --path 导出文件 --out 目录
"""
import argparse
import collections
import csv
import os
import sys

from . import canon, paths
from .safe import guarded

Q = (0.05, 0.25, 0.5, 0.75, 0.95)


def quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {f"p{int(q * 100)}": None for q in Q}
    return {f"p{int(q * 100)}": xs[min(len(xs) - 1, int(q * len(xs)))] for q in Q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, paths.core_path("g2"))
    from ruler_g2 import build_real, core_set, surgery_day
    os.makedirs(a.out, exist_ok=True)
    n_docs, los, chars = [], [], collections.defaultdict(list)
    counts = collections.Counter()
    admit_no = collections.Counter()
    per_type = collections.Counter()
    gaps = []
    for x in canon.read_jsonl(a.path):
        line = canon.admission_line(x, "R")
        adm = build_real({"admission_id": line["id"], "docs": line["docs"]})
        counts["admissions"] += 1
        n_docs.append(len(adm.docs))
        days = [d.day for d in adm.docs if d.day is not None]
        if days:
            los.append(max(days) - min(days))
        else:
            counts["no_day_offset_any"] += 1
        counts["null_day_docs"] += sum(d.day is None for d in adm.docs)
        counts["documents"] += len(adm.docs)
        counts["has_surgery"] += surgery_day(adm) is not None
        counts["missing_fp"] += adm.fp is None
        counts["missing_hr"] += adm.hr is None
        counts["missing_ds"] += adm.ds is None
        counts["core_none"] += core_set(adm) is None
        kinds = {}
        for d in adm.docs:
            label = d.subtype or d.doc_type
            chars[label].append(len(d.text))
            per_type[label] += 1
            kinds[label] = kinds.get(label, 0) + 1
        for label in kinds:
            counts["adm_with:" + label] += 1
        admit_no[str(line["meta"].get("admit_no"))] += 1
    rows = [{"table": "counts", "key": k, "n": v, "share": v / counts["admissions"] if k != "documents" else None} for k, v in sorted(counts.items())]
    rows += [{"table": "admit_no", "key": k, "n": v, "share": v / counts["admissions"]} for k, v in sorted(admit_no.items(), key=lambda t: (len(t[0]), t[0]))]
    rows += [{"table": "n_docs", "key": "每住院文书数", "n": len(n_docs), **quantiles(n_docs)}]
    rows += [{"table": "los", "key": "末篇减首篇天数", "n": len(los), **quantiles(los)}]
    rows += [{"table": "chars", "key": k, "n": per_type[k], **quantiles(v)} for k, v in sorted(chars.items())]
    cols = ["table", "key", "n", "share"] + [f"p{int(q * 100)}" for q in Q]
    with open(os.path.join(a.out, "S0_profile.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    guarded(main)
