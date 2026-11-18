"""重算旧账：S4b 手术日注入的重数。

问题：`g2_kit.mines.mine_s4b` 用 DATE_SURG 取"诊疗经过里第一个 日期+行…术"，这个日期可能是造影、胃镜这类操作的日期，
不是术后首程记的那台手术，这时候把它改 +2 天，S4b 不报是对的，不该算漏报。
判定：注入句里"行…"后面的手术名，和术后首程"手术方式/手术名称"的字符 2-gram 重合 ≥0.5 算同一台手术（改的是手术日）；
另外注入前日期已经等于术后首程的手术日，也算。其余算"改到了别的操作"。

输出：宿主数；两类各多少；两类各自的 S4b 召回（Clopper–Pearson）。只出计数。

python -B -m bench.s4b_recount --path 导出文件 --out 目录 [--filter-ids 文件]
"""
import argparse
import csv
import datetime as dt
import os
import random
import re
import sys

from . import canon, paths
from .stats_cp import cp
from .safe import guarded, log

_OP_NAME = re.compile(r"(?:手术方式|手术名称|拟施手术名称和方式|术式)\s*[:：]?\s*([^\n。；;]{2,60})")
_DATE = re.compile(r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")
_AFTER_XING = re.compile(r"行([^。；;\n，,]{2,40}术)")
SEED = 20261101
HIT_SLACK = 5


def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)}


def same_surgery(injected_clause, post_text):
    m = _AFTER_XING.search(injected_clause)
    n = _OP_NAME.search(post_text)
    if not m or not n:
        return None
    a, b = bigrams(m.group(1)), bigrams(n.group(1))
    if not a or not b:
        return None
    return len(a & b) / min(len(a), len(b)) >= 0.5


def post_date(post_text):
    m = re.search(r"手术(?:日期|时间)\s*[:：]?\s*(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})", post_text)
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def classify(orig_text, mine_rec, post_text):
    """(是手术日, 判定依据)。依据只有标签。mine_rec.start/end 是注入后的位置，这里用注入前原文的同一位置。"""
    s = mine_rec["start"]
    clause_end = re.search(r"[。；;\n]", orig_text[s:])
    clause = orig_text[s:s + clause_end.start()] if clause_end else orig_text[s:s + 80]
    m = _DATE.search(clause)
    pd = post_date(post_text)
    if m and pd:
        try:
            if dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) == pd:
                return True, "日期等于术后首程手术日"
        except ValueError:
            pass
    same = same_surgery(clause, post_text)
    if same is None:
        return None, "认不出手术名"
    return same, "手术名重合" if same else "手术名不重合"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter-ids")
    a = ap.parse_args()
    sys.path.insert(0, paths.core_path("g2"))
    from dataclasses import asdict
    from g2_kit.mines import mine_s4b
    from ruler_g2 import build_real, evaluate_admission, hit_span
    os.makedirs(a.out, exist_ok=True)
    keep = None
    if a.filter_ids:
        with open(a.filter_ids, encoding="utf-8") as f:
            keep = {x.strip() for x in f if x.strip()}
    tally = {}
    n_hosts = 0
    for x in canon.read_jsonl(a.path):
        line = canon.admission_line(x, "R")
        if keep is not None and line["id"] not in keep:
            continue
        adm = build_real({"admission_id": line["id"], "docs": line["docs"]})
        made = mine_s4b(adm, random.Random(f"{SEED}|{line['id']}"))
        if made is None:
            continue
        new, rec = made
        n_hosts += 1
        rec_d = asdict(rec) if hasattr(rec, "__dataclass_fields__") else rec._asdict()
        doc_idx, start, end = rec_d["doc_idx"], rec_d["start"], rec_d["end"]
        orig = next(d.text for d in adm.docs if d.idx == doc_idx)
        post = next((d for d in adm.docs if d.subtype == "术后首次病程"), None)
        is_surg, why = classify(orig, {"start": start}, post.text if post else "")
        detected = hit_span(evaluate_admission(new), "S4b", doc_idx, start, end, slack=HIT_SLACK)
        key = {True: "手术日", False: "别的操作", None: "认不出"}[is_surg]
        t = tally.setdefault(key, {"hosts": 0, "detected": 0})
        t["hosts"] += 1
        t["detected"] += bool(detected)
    log("s4b_recount", hosts=n_hosts, kinds=len(tally))
    rows = []
    for key in ("手术日", "别的操作", "认不出"):
        t = tally.get(key, {"hosts": 0, "detected": 0})
        lo, hi = cp(t["detected"], t["hosts"]) if t["hosts"] else (None, None)
        rows.append({"class": key, "hosts": t["hosts"], "detected": t["detected"], "recall": t["detected"] / t["hosts"] if t["hosts"] else None,
                     "cp_low": lo, "cp_high": hi})
    with open(os.path.join(a.out, "S0_s4b_recount.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "hosts", "detected", "recall", "cp_low", "cp_high"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    guarded(main)
