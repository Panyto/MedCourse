"""分折封存：把 run 9 导出文件按 split_fold 数出每折住院数、文书数和住院号列表的 sha，写一张封存表。

只出计数和 sha，不出正文。fold 0 现在就能跑；批件到了之后用全量文件再跑一次，表里 fold 0 这一行必须和现在完全相同。

python -B -m bench.sealcheck --path 导出文件 --out 目录
"""
import argparse
import collections
import csv
import hashlib
import json
import os

from .canon import read_jsonl
from .safe import guarded

ROLES = {0: "开发", 1: "规则校准（只跑一次）", 8: "开发记分牌的真实行、模型选择", 9: "封存，S6 只开一次"}


def role(f):
    return ROLES.get(f, "抽取、统计、训练") if 2 <= f <= 7 or f in ROLES else "未定义"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids, docs = collections.defaultdict(list), collections.Counter()
    for x in read_jsonl(a.path):
        f = int(x["split_fold"])
        ids[f].append(str(x["admission_id"]))
        docs[f] += len(x.get("docs") or [])
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(ids):
        h = hashlib.sha256("\n".join(sorted(ids[f])).encode("utf-8")).hexdigest()
        rows.append({"fold": f, "role": role(f), "admissions": len(ids[f]), "documents": docs[f], "ids_sha256": h})
    with open(os.path.join(a.out, "S0_folds.csv"), "w", encoding="utf-8", newline="") as w:
        c = csv.DictWriter(w, fieldnames=["fold", "role", "admissions", "documents", "ids_sha256"])
        c.writeheader()
        c.writerows(rows)
    print(json.dumps([{k: r[k] for k in ("fold", "admissions", "documents")} for r in rows]))


if __name__ == "__main__":
    guarded(main)
