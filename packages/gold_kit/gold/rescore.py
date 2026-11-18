"""纠偏层的开发检查（PREREG S2G v2 第 2 节）：把 extract.normalize 套在 S2G v1 的 300 题上，用老板已有的标签重新计分。只出计数。

python -B -m gold.rescore --dir v1金标目录 --out tables/S2G_v1_rescore.csv

纠偏规则就是照着这 300 题写的，在这上面算出来的准确率偏乐观，不作判定。这一步只看两件事：
1. 误删：老板标"成立"的题，被纠偏分到出口（体征、七类以外、不是这个病人）的有几题。超过成立题的 5% 就停，规则太狠，不上 GPU。
2. 规则有没有按设计生效：分到出口的不成立题有几题；状态改了以后和老板改的对不对得上。
v1 的状态"排除"（rule_out）按老板当时的选项读作"已排除"（excluded）。

0.3（PREREG S2G v4）：开发集换成 S2G3 的 180 题（--legacy-v1 不加），v1 的 300 题加 --legacy-v1 只作描述。多出：
- 类型改对：不成立、原因是类型错的题，纠偏改成的正是老板选的那一类（type_fixed）；成立的题被改了类型算改坏（kept_type_changed_valid）。
- 逐条规则的状态翻转（S2G3 补充 2 的口径）：成立、类型没变、判了状态的题，纠偏前后对错变了的，记在这一题用到的每条规则下
  （up 错变对、down 对变错、side 错变另一种错；没有规则的记 <无规则>）。
- 各类改后读数（after_*，只作描述）：按纠偏后的类型分组，成立 = 原来成立且类型没变，或原来类型错而改成了老板选的那一类。
名称回退层（stratum=fallback）不算。
"""
import argparse
import collections
import csv
import json
import os

from medcourse.contract import is_core
from medcourse.extract.normalize import normalize_event
from medcourse.safe import guarded, log

from .serve import load_labels

DOC_EN = {"入院记录": "hospital_record", "出院记录": "discharge_summary", "病程记录": "progress_note"}
V1_TRUE = {"rule_out": "excluded"}
MAX_FALSE_DROP = 0.05


def rescore(items, labels, legacy=False):
    cnt = collections.Counter()
    for x in items:
        if x["part"] != "A" or x.get("stratum") == "fallback":
            continue
        lab = labels.get(x["item_id"]) or {}
        if lab.get("valid") not in ("yes", "no"):
            cnt["unlabeled"] += 1
            continue
        e = dict(x["event"])
        s, t = x["hl"][0]
        n = normalize_event(e, x["text"], s, t, DOC_EN.get(x.get("doc_type"), "progress_note"), x.get("section"))
        typ0, valid = e["event_type"], lab["valid"] == "yes"
        cnt[f"n:{typ0}"] += 1
        cnt[f"valid_v1:{typ0}"] += valid
        for r in n["norm"]:
            cnt[f"rule:{r}|{'valid' if valid else 'invalid'}"] += 1
        if not is_core(n):
            cnt[f"routed_out_{'valid' if valid else 'invalid'}:{typ0}"] += 1
            continue
        t1 = n["event_type"]
        fixed = not valid and lab.get("reason") == "wrong_type" and lab.get("type_true") == t1
        cnt[f"after_n:{t1}"] += 1
        cnt[f"after_valid:{t1}"] += (valid and t1 == typ0) or fixed
        cnt[f"kept:{typ0}"] += 1
        if t1 != typ0:
            # 类型改了：原来不成立、改成了老板选的那一类算改对；原来成立的算改坏
            cnt[f"kept_type_changed_{'valid' if valid else 'invalid'}:{typ0}->{t1}"] += 1
            cnt["type_fixed"] += fixed
            continue
        cnt[f"kept_same_type:{typ0}"] += 1
        cnt[f"kept_same_type_valid:{typ0}"] += valid
        if valid and lab.get("assertion") in ("ok", "bad"):
            gold = e.get("assertion") if lab["assertion"] == "ok" else lab.get("assertion_true")
            if legacy:
                gold = V1_TRUE.get(gold, gold)
            model = V1_TRUE.get(e.get("assertion"), e.get("assertion")) if legacy else e.get("assertion")
            after = n.get("assertion")
            cnt[f"assert_n:{typ0}"] += 1
            cnt[f"assert_ok_before:{typ0}"] += model == gold
            cnt[f"assert_ok_after:{typ0}"] += after == gold
            if after != model:
                kind = "up" if after == gold else "down" if model == gold else "side"
                for r in n["norm"] or ["<无规则>"]:
                    cnt[f"{kind}:{r}|{typ0}"] += 1
    rows = sorted(cnt.items())
    valid = sum(v for k, v in cnt.items() if k.startswith("valid_v1:"))
    dropped = sum(v for k, v in cnt.items() if k.startswith("routed_out_valid:"))
    share = dropped / valid if valid else 0.0
    rows += [("false_drop_share", round(share, 4)), ("false_drop_line", MAX_FALSE_DROP), ("false_drop_pass", int(share <= MAX_FALSE_DROP))]
    for g in ("all",):
        n_ = sum(v for k, v in cnt.items() if k.startswith("assert_n:"))
        b = sum(v for k, v in cnt.items() if k.startswith("assert_ok_before:"))
        a = sum(v for k, v in cnt.items() if k.startswith("assert_ok_after:"))
        rows += [(f"assert_rate_before:{g}", round(b / n_, 4) if n_ else ""), (f"assert_rate_after:{g}", round(a / n_, 4) if n_ else "")]
        kept = sum(v for k, v in cnt.items() if k.startswith("kept_same_type:"))
        kv = sum(v for k, v in cnt.items() if k.startswith("kept_same_type_valid:"))
        rows += [(f"valid_rate_kept_same_type:{g}", round(kv / kept, 4) if kept else "")]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legacy-v1", action="store_true")
    a = ap.parse_args()
    os.umask(0o077)
    with open(os.path.join(a.dir, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(x) for x in f if x.strip()]
    rows = rescore(items, load_labels(os.path.join(a.dir, "labels.jsonl")), legacy=a.legacy_v1)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    log("gold_rescore_done", rows=len(rows))


if __name__ == "__main__":
    guarded(main)
