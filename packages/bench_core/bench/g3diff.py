"""g3-rules 0.2 → 0.3 的改动差异：每行有多少住院的硬错误、N1、具体度变了。只读 raw 下的数字。

python -B -m bench.g3diff --out 跑分输出目录
产物 tables/S0_g3_diff.csv。真实行上"硬错误减少的住院"就是老板抽查 20 条的抽样框（住院号写进 work，不进包）。
"""
import argparse
import csv
import os

from .aggregate import load_jsonl

FLAGS = ("any_hard", "any_severe", "any_pedadult", "any_sex", "any_warning", "any_n1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--changed-ids", help="写出硬错误变化的 行,住院号 清单（放 work/，不进包）")
    a = ap.parse_args()
    raw = os.path.join(a.out, "raw")
    new = {(r["row"], r["id"]): r for r in load_jsonl(os.path.join(raw, "all.g3.jsonl"))}
    old = {(r["row"], r["id"]): r for r in load_jsonl(os.path.join(raw, "all.g3old.jsonl"))}
    rows, changed = {}, []
    for k, n in new.items():
        o = old.get(k)
        if o is None or "@" in k[0]:
            continue
        t = rows.setdefault(k[0], {"row": k[0], "records": 0, "mentions_old": 0, "mentions_new": 0, "dose_old": 0, "dose_new": 0})
        t["records"] += 1
        t["mentions_old"] += o["n_mentions"]
        t["mentions_new"] += n["n_mentions"]
        t["dose_old"] += o["n_with_dose"]
        t["dose_new"] += n["n_with_dose"]
        for f in FLAGS:
            t[f + "_old"] = t.get(f + "_old", 0) + bool(o[f])
            t[f + "_new"] = t.get(f + "_new", 0) + bool(n[f])
            t[f + "_removed"] = t.get(f + "_removed", 0) + (bool(o[f]) and not n[f])
            t[f + "_added"] = t.get(f + "_added", 0) + (bool(n[f]) and not o[f])
        if bool(o["any_hard"]) != bool(n["any_hard"]) or bool(o["any_n1"]) != bool(n["any_n1"]):
            changed.append((k[0], k[1]))
    cols = ["row", "records", "mentions_old", "mentions_new", "dose_old", "dose_new"]
    for f in FLAGS:
        cols += [f + s for s in ("_old", "_new", "_removed", "_added")]
    with open(os.path.join(a.out, "tables", "S0_g3_diff.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    if a.changed_ids:
        with open(a.changed_ids, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(f"{x},{y}\n" for x, y in sorted(changed))


if __name__ == "__main__":
    main()
