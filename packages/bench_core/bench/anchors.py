"""基线复现：把跑分器的读数和归档定稿数字逐条对。容差写在 data/anchors.csv；分子分母有的话要求逐字相等。

python -B -m bench.anchors --out 跑分输出目录
产物 tables/S0_anchor_check.csv（每条一行）和 tables/S0_anchor_summary.csv；有不一致时退出码 3，不算崩溃，交给 REPORT 写明。
"""
import argparse
import csv
import os
import sys

from . import aggregate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_anchors(path=None):
    with open(path or os.path.join(HERE, "data", "anchors.csv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def within(got, want, abs_tol, rel_tol):
    return abs(got - want) <= abs_tol + rel_tol * abs(want)


def check(raw_dir, anchors):
    per, whole = {}, {}
    for reader in ("g1", "g2", "g3", "g3old", "g4", "g4old", "g6", "c2"):
        for r in aggregate.load_jsonl(os.path.join(raw_dir, f"all.{reader}.jsonl")):
            per.setdefault((reader, r["row"]), []).append(r)
        for r in aggregate.load_jsonl(os.path.join(raw_dir, f"all.{reader}.row.jsonl")):
            whole.setdefault((reader, r["row"]), {})[r["metric"]] = r
    out = []
    aggregate.B = 0     # 这里只要点估计，不算区间
    for a in anchors:
        m = {"id": a["anchor_id"], "how": a["how"], "field": a["field"], "chk": a["chk"] or None, "den": a["den"] or None}
        rec = per.get((a["reader"], a["row"]), [])
        res = aggregate.one_metric(m, rec, whole.get((a["reader"], a["row"]), {}))
        want = float(a["value"])
        got = res.get("value")
        row = {"anchor_id": a["anchor_id"], "source": a["source"], "row": a["row"], "reader": a["reader"], "field": a["field"], "want": want, "got": got,
               "want_num": a["numerator"], "want_den": a["denominator"], "got_num": res.get("numerator"), "got_den": res.get("denominator")}
        if got is None:
            row["status"] = "没有读数"
        else:
            ok = within(float(got), want, float(a["abs_tol"]), float(a["rel_tol"]))
            if a["numerator"] and res.get("numerator") is not None:
                ok = ok and int(float(a["numerator"])) == int(res["numerator"]) and int(float(a["denominator"])) == int(res["denominator"])
            row["diff"] = float(got) - want
            row["status"] = "一致" if ok else "不一致"
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors")
    a = ap.parse_args()
    res = check(os.path.join(a.out, "raw"), load_anchors(a.anchors))
    aggregate.write_csv(os.path.join(a.out, "tables", "S0_anchor_check.csv"), res,
                        ["anchor_id", "source", "row", "reader", "field", "want", "got", "diff", "want_num", "want_den", "got_num", "got_den", "status"])
    tally = {}
    for r in res:
        k = (r["source"], r["row"], r["status"])
        tally[k] = tally.get(k, 0) + 1
    aggregate.write_csv(os.path.join(a.out, "tables", "S0_anchor_summary.csv"),
                        [{"source": k[0], "row": k[1], "status": k[2], "n": v} for k, v in sorted(tally.items())], ["source", "row", "status", "n"])
    bad = sum(1 for r in res if r["status"] != "一致")
    print({"anchors": len(res), "not_matched": bad})
    if bad:
        sys.exit(3)


if __name__ == "__main__":
    main()
