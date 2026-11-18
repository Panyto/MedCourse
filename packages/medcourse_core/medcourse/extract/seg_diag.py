"""切片诊断：真实文书切出来多少片、每片多长、各章节占多少。只出计数和分位数，不调模型，不出正文。

python -B -m medcourse.extract.seg_diag --path 导出文件 --out 目录
"""
import argparse
import csv
import json
import os

from ..safe import guarded
from .segments import segments

BANDS = (100, 200, 400, 600)


def pct(xs, q):
    if not xs:
        return None
    return xs[int(q * (len(xs) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by_type, by_section = {}, {}
    with open(a.path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for doc in json.loads(line).get("docs") or []:
                t = doc.get("doc_type") or "?"
                segs = segments(doc)
                g = by_type.setdefault(t, {"docs": 0, "chars": 0, "segs": [], "seg_chars": []})
                g["docs"] += 1
                g["chars"] += len(doc.get("text") or "")
                g["segs"].append(len(segs))
                for s in segs:
                    g["seg_chars"].append(len(s["text"]))
                    by_section.setdefault((t, s["section"]), []).append(len(s["text"]))
    rows = []
    for t, g in sorted(by_type.items()):
        chars, n = sorted(g["seg_chars"]), sorted(g["segs"])
        row = {"scope": "doc_type", "doc_type": t, "section": "*", "docs": g["docs"], "segments": len(chars), "mean_doc_chars": round(g["chars"] / max(1, g["docs"]), 1),
               "segs_p50": pct(n, 0.5), "segs_p95": pct(n, 0.95), "chars_p05": pct(chars, 0.05), "chars_p50": pct(chars, 0.5), "chars_p95": pct(chars, 0.95)}
        row.update({f"share_le_{b}": round(sum(c <= b for c in chars) / max(1, len(chars)), 4) for b in BANDS})
        rows.append(row)
    for (t, name), xs in sorted(by_section.items()):
        xs = sorted(xs)
        row = {"scope": "section", "doc_type": t, "section": name, "docs": None, "segments": len(xs), "mean_doc_chars": None, "segs_p50": None, "segs_p95": None,
               "chars_p05": pct(xs, 0.05), "chars_p50": pct(xs, 0.5), "chars_p95": pct(xs, 0.95)}
        row.update({f"share_le_{b}": round(sum(c <= b for c in xs) / len(xs), 4) for b in BANDS})
        rows.append(row)
    with open(os.path.join(a.out, "S2_seg_diag.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    guarded(main)
