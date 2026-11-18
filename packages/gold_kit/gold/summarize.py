"""金标汇总：items.jsonl ＋ labels.jsonl → 计数表和按预登记的判定。只有计数和类型名，可以进包。

python -B -m gold.summarize --dir 金标目录 --out tables/S2_gold_summary.csv

准确率（A 部分）：成立率（占全部题）、名称对（占成立）、数值对（占成立且有数值）、状态对（占成立），总体和按事件类型。
漏抽（B 部分）：召回 ≈ 抽到且成立的 ÷（抽到且成立的 ＋ 漏抽的）；"抽到且成立"用这一类在 A 部分的成立率折算。"3+" 按 3 计（偏保守）。
区间是 Wilson 95%。判定用点估计，线见 LINES，和预登记一致。

0.2（PREREG S2G v2）：按类型准入，总成立率只作描述（v1 的总成立率底线把单看过线的化验、诊断一起挡住了；总成立率随抽样比例变，
下游又是按类型用）。准入：这一类题数 ≥ 20、成立 ≥ 0.85、状态对 ≥ 0.85；化验、生命体征、用药还要数值对 ≥ 0.95（这一类带数值的题
不到 10 道时，用这三类合起来的数值准确率）。另出不成立的原因、"类型错"时模型类型 → 应该的类型。

0.3（PREREG S2G v4）：名称回退那一层（stratum=fallback）单算成立和状态（fallback:*），不进总体和按类型的读数，也不进准入。
"""
import argparse
import collections
import csv
import json
import math
import os

from medcourse.safe import guarded, log

from .serve import TYPES, load_labels

LINES = {"name": 0.90, "value": 0.95, "assertion": 0.85, "recall": 0.80}
TYPE_VALID, TYPE_ASSERT, MIN_N_TYPE = 0.85, 0.85, 20
VALUE_TYPES = ("lab", "vital", "medication")
MIN_VALUE_N = 10


def wilson(k, n, z=1.96):
    if n == 0:
        return "", ""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - r) / d, 4), round((c + r) / d, 4)


def summarize(items, labels):
    rows, cnt = [], collections.Counter()
    a_items = [x for x in items if x["part"] == "A" and x.get("stratum") != "fallback"]
    for x in items:
        if x["part"] == "A" and x.get("stratum") == "fallback":
            lab = labels.get(x["item_id"])
            if not lab or lab.get("valid") not in ("yes", "no"):
                cnt["A_unlabeled"] += 1
                continue
            cnt["fb_n"] += 1
            cnt[f"fb_n:{x['event']['event_type']}"] += 1
            if lab["valid"] == "yes":
                cnt["fb_valid"] += 1
                cnt[f"fb_valid:{x['event']['event_type']}"] += 1
                cnt["fb_assert_n"] += lab.get("assertion") in ("ok", "bad")
                cnt["fb_assert_ok"] += lab.get("assertion") == "ok"
            else:
                cnt[f"fb_invalid_reason:{lab.get('reason', '?')}"] += 1
    b_items = [x for x in items if x["part"] == "B"]
    for x in a_items:
        lab, t = labels.get(x["item_id"]), x["event"]["event_type"]
        if not lab or lab.get("valid") not in ("yes", "no"):
            cnt["A_unlabeled"] += 1
            continue
        if lab["valid"] != "yes":
            cnt[f"invalid_reason:{t}|{lab.get('reason', '?')}"] += 1
            if lab.get("reason") == "wrong_type":
                cnt[f"type_confusion:{t}->{lab.get('type_true', '?')}"] += 1
        for g in ("all", t):
            cnt[f"A_n:{g}"] += 1
            if lab["valid"] != "yes":
                continue
            if t in VALUE_TYPES and x["event"].get("value") is not None:
                cnt["value_n:valuetypes"] += lab.get("value") in ("ok", "bad") and g == t
                cnt["value_ok:valuetypes"] += lab.get("value") == "ok" and g == t
            cnt[f"valid:{g}"] += 1
            cnt[f"name_n:{g}"] += lab.get("name") in ("ok", "bad")
            cnt[f"name_ok:{g}"] += lab.get("name") == "ok"
            if x["event"].get("value") is not None:
                cnt[f"value_n:{g}"] += lab.get("value") in ("ok", "bad")
                cnt[f"value_ok:{g}"] += lab.get("value") == "ok"
            cnt[f"assert_n:{g}"] += lab.get("assertion") in ("ok", "bad")
            cnt[f"assert_ok:{g}"] += lab.get("assertion") == "ok"
            if g == "all" and lab.get("assertion") == "bad":
                cnt[f"assert_confusion:{x['event'].get('assertion')}->{lab.get('assertion_true', '?')}"] += 1
    for x in b_items:
        lab = labels.get(x["item_id"]) or {}
        if not all(f"miss_{t}" in lab for t in TYPES):
            cnt["B_unlabeled"] += 1
            continue
        cnt["B_n"] += 1
        for t in TYPES:
            m = 3 if lab[f"miss_{t}"] == "3+" else int(lab[f"miss_{t}"])
            cnt[f"missed:{t}"] += m
            cnt["missed:all"] += m
        for e in x["events"]:
            cnt[f"extracted:{e['event_type']}"] += 1
            cnt["extracted:all"] += 1

    def rate(name, k, n, line=None):
        lo, hi = wilson(k, n)
        r = round(k / n, 4) if n else ""
        rows.append({"metric": name, "k": k, "n": n, "rate": r, "lo": lo, "hi": hi, "line": line if line is not None else "",
                     "pass": "" if line is None or not n else int(k / n >= line)})
        return k / n if n else None

    res = {}
    for g in ["all"] + sorted(TYPES):
        overall = g == "all"
        res[(g, "valid")] = rate(f"valid:{g}", cnt[f"valid:{g}"], cnt[f"A_n:{g}"], None if overall else TYPE_VALID)
        rate(f"name:{g}", cnt[f"name_ok:{g}"], cnt[f"name_n:{g}"], LINES["name"] if overall else None)
        res[(g, "value")] = rate(f"value:{g}", cnt[f"value_ok:{g}"], cnt[f"value_n:{g}"], LINES["value"] if overall else None)
        res[(g, "assertion")] = rate(f"assertion:{g}", cnt[f"assert_ok:{g}"], cnt[f"assert_n:{g}"], LINES["assertion"] if overall else TYPE_ASSERT)
    # 召回：抽到的按这一类的成立率折算成"抽到且成立"
    for g in ["all"] + sorted(TYPES):
        p = res.get((g, "valid"))
        ext = cnt[f"extracted:{g}"] * (p if p is not None else 1.0)
        miss = cnt[f"missed:{g}"]
        r = ext / (ext + miss) if ext + miss else None
        rows.append({"metric": f"recall:{g}", "k": round(ext, 1), "n": round(ext + miss, 1), "rate": round(r, 4) if r is not None else "", "lo": "", "hi": "",
                     "line": LINES["recall"] if g == "all" else "", "pass": "" if g != "all" or r is None else int(r >= LINES["recall"])})
        res[(g, "recall")] = r
    res[("valuetypes", "value")] = rate("value:valuetypes", cnt["value_ok:valuetypes"], cnt["value_n:valuetypes"], None)
    rate("fallback:valid", cnt["fb_valid"], cnt["fb_n"], None)
    rate("fallback:assertion", cnt["fb_assert_ok"], cnt["fb_assert_n"], None)
    for k in ("A_unlabeled", "B_unlabeled", "B_n"):
        rows.append({"metric": k, "k": cnt[k], "n": "", "rate": "", "lo": "", "hi": "", "line": "", "pass": ""})
    for k, v in sorted(cnt.items()):
        if k.startswith(("assert_confusion:", "invalid_reason:", "type_confusion:", "fb_n:", "fb_valid:", "fb_invalid_reason:")):
            rows.append({"metric": k, "k": v, "n": "", "rate": "", "lo": "", "hi": "", "line": "", "pass": ""})
    # 判定（PREREG S2G v2 第 4 节）：按类型准入，没有总成立率底线
    labeled = cnt["A_n:all"] and not cnt["A_unlabeled"] and not cnt["B_unlabeled"]
    admitted = []
    for t in sorted(TYPES):
        n = cnt[f"A_n:{t}"]
        tv, ta = res[(t, "valid")], res[(t, "assertion")]
        ok = bool(labeled) and n >= MIN_N_TYPE and tv is not None and tv >= TYPE_VALID and ta is not None and ta >= TYPE_ASSERT
        if t in VALUE_TYPES:     # 数值要进分位数的类型，数值准确率也要过线
            vv = res[(t, "value")] if cnt[f"value_n:{t}"] >= MIN_VALUE_N else res[("valuetypes", "value")]
            ok = ok and (vv is None or vv >= LINES["value"])
        if ok:
            admitted.append(t)
        rows.append({"metric": f"admit:{t}", "k": int(ok), "n": n, "rate": "", "lo": "", "hi": "", "line": "", "pass": ""})
    verdict = "未标完" if not labeled else ("准入：" + "、".join(TYPES[t] for t in admitted) if admitted else "七类都不准入")
    rows.append({"metric": "verdict", "k": verdict, "n": "", "rate": "", "lo": "", "hi": "", "line": "", "pass": ""})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.umask(0o077)
    with open(os.path.join(a.dir, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(x) for x in f if x.strip()]
    rows = summarize(items, load_labels(os.path.join(a.dir, "labels.jsonl")))
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "k", "n", "rate", "lo", "hi", "line", "pass"])
        w.writeheader()
        w.writerows(rows)
    log("gold_summary_done", rows=len(rows))


if __name__ == "__main__":
    guarded(main)
