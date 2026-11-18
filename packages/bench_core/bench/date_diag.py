"""日期读数诊断（PREREG_S0_v2 第 5 节）：在跑记分牌之前，确认读到的确实是每篇自己的书写日。只出计数，不出日期和正文。

python -B -m bench.date_diag --row R-full=导出文件 --row G-E2=合成行 --real R-full --out 目录

三条验收只看 --real 指定的行，不过时退出码 3，停下把表交回总经理，不要调规则：
 D1 读得出：有日程的病程里读出日期的比例 ≥ 0.95
 D2 不是固定日期：相邻两篇日程不同时，日期也不同的比例 ≥ 0.95（读到入院日期这类固定值时会很低）
 D3 和日程一致：相邻两篇的日期差等于日程差的比例 ≥ 0.95
一致率接近 1 说明日程可能就是从这个日期推出来的，真实行的倒退率这时只说明读对了，不算独立证据，表里 tautology_flag 标出。
"""
import argparse
import csv
import os
import sys

from . import canon
from .readers import extra
from .safe import guarded, log

MIN_READ = MIN_MOVED = MIN_AGREE = 0.95
TAUTOLOGY = 0.99
BINS = ((0, 40), (40, 60), (60, 80), (80, 100), (100, 120))
COLS = ["row", "admissions", "notes", "read", "read_share", "any_date_120", "rule_first40", "rule_stamp", "rule_title", "rule_line",
        "pos_0_39", "pos_40_59", "pos_60_79", "pos_80_99", "pos_100_119", "adm_ge2", "adm_viol", "viol_rate", "pairs", "pairs_diff_day",
        "pairs_moved", "moved_share", "pairs_agree", "agree_share", "tautology_flag", "gate_row", "d1", "d2", "d3"]


def _share(a, b):
    return a / b if b else None


def tally(lines):
    c = {k: 0 for k in COLS if k not in ("row", "read_share", "viol_rate", "moved_share", "agree_share", "tautology_flag", "gate_row", "d1", "d2", "d3")}
    for x in lines:
        c["admissions"] += 1
        seq = [d for d in x["docs"] if d.get("doc_type") == "progress_note" and d.get("day_offset") is not None]
        seq.sort(key=lambda d: int(d["day_offset"]))
        got = []
        for d in seq:
            c["notes"] += 1
            head = d["text"][:extra.HEAD_MAX]
            c["any_date_120"] += extra._DATE.search(head) is not None
            when, rule = extra.head_date_rule(d["text"])
            if when is None:
                continue
            c["read"] += 1
            c["rule_" + rule] += 1
            at = extra._DATE.search(head)
            for lo, hi in BINS:
                if at and lo <= at.start() < hi:
                    c[f"pos_{lo}_{hi - 1}"] += 1
            got.append((int(d["day_offset"]), when))
        if len(got) < 2:
            continue
        c["adm_ge2"] += 1
        c["adm_viol"] += any(b[1] < a[1] for a, b in zip(got, got[1:]))
        for a, b in zip(got, got[1:]):
            c["pairs"] += 1
            days, step = (b[1] - a[1]).days, b[0] - a[0]
            c["pairs_agree"] += days == step
            if step:
                c["pairs_diff_day"] += 1
                c["pairs_moved"] += days != 0
    return c


def judge(c, name, real):
    c["row"], c["gate_row"] = name, name in real
    c["read_share"] = _share(c["read"], c["notes"])
    c["viol_rate"] = _share(c["adm_viol"], c["adm_ge2"])
    c["moved_share"] = _share(c["pairs_moved"], c["pairs_diff_day"])
    c["agree_share"] = _share(c["pairs_agree"], c["pairs"])
    c["d1"] = c["read_share"] is not None and c["read_share"] >= MIN_READ
    c["d2"] = c["moved_share"] is not None and c["moved_share"] >= MIN_MOVED
    c["d3"] = c["agree_share"] is not None and c["agree_share"] >= MIN_AGREE
    c["tautology_flag"] = c["agree_share"] is not None and c["agree_share"] >= TAUTOLOGY
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True, help="行名=路径")
    ap.add_argument("--real", default="", help="逗号分隔，要过 D1–D3 的真实行")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    real = {x for x in a.real.split(",") if x}
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for one in a.row:
        name, path = one.split("=", 1)
        lines = canon.load_row({"row": name, "path": path, "kind": "admission", "view": "full"})
        rows.append(judge(tally(lines), name, real))
        log("date_diag", row=name, notes=rows[-1]["notes"], read=rows[-1]["read"], adm_ge2=rows[-1]["adm_ge2"])
    with open(os.path.join(a.out, "S0_date_diag.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    if any(r["gate_row"] and not (r["d1"] and r["d2"] and r["d3"]) for r in rows):
        sys.exit(3)


if __name__ == "__main__":
    guarded(main)
