# -*- coding: utf-8 -*-
"""fold 2–7 S2 抽取第一段：只抽入院记录和出院记录（老板 11-10 批准分两段跑）。s2-stage1-0.1。

包在 medcourse.extract.run 外面，抽取代码一行不改：只把 run.work_items 换成只放行入院、出院记录切片的版本，参数和
medcourse.extract.run 完全相同。文书序号（doc_idx）、切片位置都还是原文件里的，所以：
- 纠偏（medcourse.extract.normalize --source 原文件）照常用；
- 第二段直接用 medcourse.extract.run、同一个原文件、同一个输出目录：done_segments.txt 里已有的入院、出院记录切片会跳过
  （skipped_done），只抽病程记录。

跑完在 counts.json 里补 stage1_* 几项：本段放行和跳过的文书、切片数，按文书类型。
"""
import argparse
import json
import os
import sys
from collections import Counter

from medcourse.extract import run as R

VERSION = "s2-stage1-0.1"
KEEP = ("hospital_record", "discharge_summary")
_ORIG = R.work_items
_SEEN = Counter()
_DOCS = set()


def work_items(lines):
    for rid, i, doc, seg in _ORIG(lines):
        kind = doc.get("doc_type") or "unknown"
        if kind in KEEP:
            _SEEN["stage1_segments_" + kind] += 1
            if (rid, i) not in _DOCS:
                _DOCS.add((rid, i))
                _SEEN["stage1_docs_" + kind] += 1
            yield rid, i, doc, seg
        else:
            _SEEN["stage1_skipped_segments_" + kind] += 1


def main():
    R.work_items = work_items
    R.main()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a, _ = ap.parse_known_args()
    path = os.path.join(a.out, "counts.json")
    with open(path, encoding="utf-8") as f:
        counts = json.load(f)
    counts.update(_SEEN)
    counts["stage1_version"] = VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=1, sort_keys=True)


if __name__ == "__main__":
    R.guarded(main)
