# -*- coding: utf-8 -*-
"""M1 正式记分表：PREREG M1 v1 第 4 节的 10 行线，第一组判 M1，第三组照同样的线读，和真实行（fold 8）并列。m1-table-0.1。

python -B -m gmtools.m1_table --board bench 的 S0_scoreboard.csv --mc-counts 第一组 counts.json --direct-counts 第三组 counts.json \
    --leak 第一组泄露表（scope,metric,…,count,denominator） --out M1_table.csv

只读计数表，不读正文。
"""
import argparse
import csv
import json

VERSION = "m1-table-0.1"
MC, DIRECT, REAL = "MC-8B", "DIRECT-8B", "R"
# (行, 读数, 比较, 线, 来源)。来源 board = 记分牌，counts = 运行计数，leak = 泄露表
LINES = [
    (1, "B_tfidf", "le", 0.40, "board"),
    (2, "B_exam", "ge", 6.5, "board"),
    (3, "B_dxchg", "le", 0.8, "board"),
    (4, "B_mentions", "ge", 7, "board"),
    (4, "B_dose", "ge", 0.20, "board"),
    (5, "A_g3_hard", "le", 0, "board"),
    (6, "A_g3_n1", "le", 0, "board"),
    (7, "B_ages", "ge", 45, "board"),
    (8, "calls_per_kchar", "le", 15, "counts"),
    (9, "B_chars", "ge", 768, "board"),
    (9, "discard_rate", "le", 0.05, "counts"),
    (10, "leak_O", "le", 0, "leak"),
    (10, "leak_A", "le", 0, "leak"),
    (10, "leak_B", "le", 0, "leak"),
]
NOT_COMPARED = {"B_ages": "不比（第三组用第一组的病人）", "discard_rate": "不比（第三组没有闸门）",
                "leak_O": "不适用（第三组没有时间线）", "leak_A": "不适用（第三组没有时间线）", "leak_B": "不适用（第三组没有时间线）",
                "calls_per_kchar": "不比（第三组不重写）"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_board(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lo, hi = _f(r.get("ci_lo")), _f(r.get("ci_hi"))
            if lo is None or hi is None:
                lo, hi = _f(r.get("cp_lo")), _f(r.get("cp_hi"))
            out[(r["row"], r["metric"])] = (_f(r.get("value")), lo, hi)
    return out


def read_leak(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("scope") == "admission" and r.get("metric") in ("O", "A", "B"):
                out["leak_" + r["metric"]] = _f(r["count"])
    return out


def ok(v, op, line):
    if v is None:
        return None
    return v <= line if op == "le" else v >= line


def overlap(a, b):
    if None in (a[1], a[2], b[1], b[2]):
        return None
    return not (a[2] < b[1] or b[2] < a[1])


def build(board, mc_counts, direct_counts, leak):
    rows = []
    for rule, metric, op, line, src in LINES:
        if src == "board":
            m, d, r = (board.get((x, metric), (None, None, None)) for x in (MC, DIRECT, REAL))
        elif src == "counts":
            m, d, r = (mc_counts.get(metric), None, None), (direct_counts.get(metric), None, None), (None, None, None)
        else:
            m, d, r = (leak.get(metric), None, None), (None, None, None), (None, None, None)
        pass_m, pass_d = ok(m[0], op, line), ok(d[0], op, line)
        if metric in NOT_COMPARED:
            verdict = NOT_COMPARED[metric]
        elif m[0] is None or d[0] is None:
            verdict = "缺数"
        else:
            sep = overlap(m, d) is False
            if r[0] is not None and line not in (0,):
                closer = "MC" if abs(m[0] - r[0]) < abs(d[0] - r[0]) else ("DIRECT" if abs(d[0] - r[0]) < abs(m[0] - r[0]) else "平")
            else:
                closer = "MC" if (pass_m and not pass_d) or (line == 0 and m[0] < d[0]) else ("DIRECT" if (pass_d and not pass_m) or (line == 0 and d[0] < m[0]) else "平")
            verdict = ("第一组胜" if closer == "MC" else "第三组胜" if closer == "DIRECT" else "平") if sep else "不分（区间重叠）"
        rows.append({"rule": rule, "metric": metric, "line": f"{op}_{line}",
                     "MC_8B": m[0], "MC_lo": m[1], "MC_hi": m[2], "MC_pass": pass_m,
                     "DIRECT_8B": d[0], "DIRECT_lo": d[1], "DIRECT_hi": d[2], "DIRECT_pass": pass_d,
                     "R_fold8": r[0], "R_lo": r[1], "R_hi": r[2], "R_pass": ok(r[0], op, line), "MC_vs_DIRECT": verdict})
    by_rule, d_rule = {}, {}
    for x in rows:
        by_rule.setdefault(x["rule"], []).append(x["MC_pass"])
        if x["metric"] not in NOT_COMPARED:
            d_rule.setdefault(x["rule"], []).append(x["DIRECT_pass"])
    m1 = all(all(v is True for v in vs) for vs in by_rule.values())
    return rows, {"m1_pass": m1, "rules_failed": sorted(k for k, vs in by_rule.items() if not all(v is True for v in vs)),
                  "direct_rules_passed": sorted(k for k, vs in d_rule.items() if all(v is True for v in vs)),
                  "direct_rules_compared": sorted(d_rule), "version": VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--mc-counts", required=True)
    ap.add_argument("--direct-counts", required=True)
    ap.add_argument("--leak", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.mc_counts, encoding="utf-8") as f:
        mc = json.load(f)
    with open(a.direct_counts, encoding="utf-8") as f:
        dc = json.load(f)
    rows, summary = build(read_board(a.board), mc, dc, read_leak(a.leak))
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(a.out[:-4] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
