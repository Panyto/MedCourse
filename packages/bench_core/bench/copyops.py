"""CPU 小实验 2：相邻病程的拷贝操作分布，验证"真实病程是每日重述，不是逐日新写"这个假设。

相邻两篇病程（按 day_offset，同一住院），当前篇逐句归类，按字符占比：
  exact            规范化后和前一篇某句全同
  numeric_changed  数字归 0 后和前一篇某句相同（只改了数值）
  near             5-gram 被前一篇覆盖 ≥0.8
  new              其他
另记两个诊断量：exact_any_prior（全同的句子出现在任一更早病程）、persist3（连续三篇都有的句子）。
分组：文书细类、天数间隔（0、1、2、3+、空）、有无手术。只出比例和计数。

python -B -m bench.copyops --row 行名=路径[=view] … --out 目录
"""
import argparse
import csv
import os
import re
import sys

from . import canon, paths
from .safe import guarded, log

_SPLIT = re.compile(r"[。；;！？!?\n]+")
_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_DATE_TIME = re.compile(r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}日?(?:\s*\d{1,2}[:：时]\d{0,2}分?)?")
MIN_SENT = 4
OPS = ("exact", "numeric_changed", "near", "new")


def sentences(text):
    out = []
    for s in _SPLIT.split(text):
        s = _SPACE.sub("", s)
        if len(s) >= MIN_SENT:
            out.append(s)
    return out


def zero(s):
    return _DIGITS.sub("0", s)


def grams(s, n=5):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def coverage(s, pool, n=5):
    g = grams(s, n)
    return sum(1 for x in g if x in pool) / len(g) if g else 0.0


def classify_pair(prev_text, cur_text):
    """返回 {操作: 字符数} 和 exact 句子列表。"""
    prev = sentences(prev_text)
    pset, zset = set(prev), {zero(s) for s in prev}
    pool = set()
    for s in prev:
        pool |= grams(s)
    chars = {k: 0 for k in OPS}
    exact = []
    for s in sentences(cur_text):
        if s in pset:
            op = "exact"
            exact.append(s)
        elif zero(s) in zset:
            op = "numeric_changed"
        elif coverage(s, pool) >= 0.8:
            op = "near"
        else:
            op = "new"
        chars[op] += len(s)
    return chars, exact


def gap_label(a, b):
    if a is None or b is None:
        return "空"
    g = int(b) - int(a)
    return str(g) if g <= 2 else "3+"


def run_row(adms, subtype_of, surgery_of):
    """adms：Admission 列表。返回 [(分组类型, 分组值, 指标, 值…)] 的累加器。"""
    acc = {}

    def add(kind, value, chars, extra):
        k = (kind, value)
        a = acc.setdefault(k, {"pairs": 0, "total": 0, "any_prior": 0, "persist3": 0, **{o: 0 for o in OPS}})
        a["pairs"] += 1
        a["total"] += sum(chars.values())
        for o in OPS:
            a[o] += chars[o]
        a["any_prior"] += extra["any_prior"]
        a["persist3"] += extra["persist3"]

    per_adm = {}
    for adm in adms:
        notes = [d for d in adm.docs if d.doc_type == "progress_note"]
        seen = set()
        prev_exact = set()
        surg = "手术" if surgery_of(adm) is not None else "无手术"
        for a, b in zip(notes, notes[1:]):
            seen |= set(sentences(a.text))
            chars, exact = classify_pair(a.text, b.text)
            cur = sentences(b.text)
            extra = {"any_prior": sum(len(s) for s in cur if s in seen),
                     "persist3": sum(len(s) for s in exact if s in prev_exact)}
            prev_exact = set(exact)
            add("全部", "全部", chars, extra)
            add("文书细类", subtype_of(b), chars, extra)
            add("天数间隔", gap_label(a.day, b.day), chars, extra)
            add("手术", surg, chars, extra)
            t = per_adm.setdefault(adm.admission_id, {o: 0 for o in OPS})
            for o in OPS:
                t[o] += chars[o]
    return acc, per_adm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True, help="行名=路径[=view]")
    ap.add_argument("--filter-ids")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, paths.core_path("g2"))
    from ruler_g2 import surgery_day
    from .readers.g2 import build
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for spec in a.row:
        parts = spec.split("=")
        name, path, view = parts[0], parts[1], (parts[2] if len(parts) > 2 else "full")
        lines = canon.load_row({"row": name, "path": path, "view": view, "filter_ids": a.filter_ids})
        lines, dropped = canon.apply_core_view(lines)
        adms = [x for x in (build(y) for y in lines) if x is not None]
        acc, per_adm = run_row(adms, lambda d: d.subtype or "", surgery_day)
        log("copyops_row", row=name, admissions=len(adms), dropped=dropped)
        for (kind, value), v in sorted(acc.items()):
            total = v["total"] or 1
            rows.append({"row": name, "group": kind, "value": value, "pairs": v["pairs"], "chars": v["total"],
                         **{f"{o}_share": v[o] / total for o in OPS},
                         "any_prior_share": v["any_prior"] / total, "persist3_share": v["persist3"] / total})
    cols = ["row", "group", "value", "pairs", "chars"] + [f"{o}_share" for o in OPS] + ["any_prior_share", "persist3_share"]
    with open(os.path.join(a.out, "S0_copyops.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    guarded(main)
