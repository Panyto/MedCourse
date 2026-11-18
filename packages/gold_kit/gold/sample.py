"""金标抽样：从一次抽取的产出（events.jsonl、done_segments.txt）和原文里按预登记规则抽题，写 items.jsonl。

python -B -m gold.sample --events 目录/events.jsonl --done 目录/done_segments.txt --source fold0.jsonl --out 金标目录 [--seed 20261103]

A 部分（量准确率）：抽事件，按事件类型分层，每类至少 MIN_PER_TYPE 条，其余按各类事件数的比例分。
B 部分（量漏抽）：抽初始切片，等概率，题面列出这一片抽到的全部事件。
items.jsonl 带原文切片，是真实派生材料，只能写在 MC_DERIVED_ROOT 下（合成数据加 --synthetic）。终端只出计数。

0.2（PREREG S2G v2）：只抽下游要用的事件（contract.is_core：七类核心类型、说的是这个病人）。体征、七类以外、"不是这个病人"
不进 A 部分；B 部分也只列这些事件，被错分到出口里的核心事件在 B 部分算漏抽。A 部分每类 20 条（等量，按类型准入要每类的准确率），
B 部分 40 个切片。读的是 extract.normalize 纠偏以后的 events_norm.jsonl。

0.3（PREREG S2G v4）：A 部分另抽一层"名称回退"（stratum=fallback，N_FB 条）：引文对不上、按名称找回来的七类事件（抽取 v0.3 的
ground="name"），单独量它们成不成立，只作描述，不进按类型的准入。先抽这一层，按类型抽时不重复抽到同一条。
"""
import argparse
import collections
import csv
import json
import os
import random

from medcourse.contract import is_core
from medcourse.extract.segments import segments
from medcourse.safe import SafeError, guarded, log

N_A, N_B, MIN_PER_TYPE = 140, 40, 20
N_FB = 20
SEED = 20261103
DERIVED_ROOT = os.environ.get("MC_DERIVED_ROOT", "/home/weify/pyt/product/derived/")
DOC_CN = {"hospital_record": "入院记录", "discharge_summary": "出院记录", "progress_note": "病程记录"}


def allocate(counts, total, floor):
    """分层名额：每类先给 min(floor, 该类数)，余下按各类剩余数的比例分（最大余数法），不超过各类总数。"""
    floor = min(floor, total // max(1, len(counts)))
    alloc = {t: min(floor, n) for t, n in counts.items()}
    left = total - sum(alloc.values())
    room = {t: counts[t] - alloc[t] for t in counts}
    pool = sum(room.values())
    if left > 0 and pool > 0:
        share = {t: left * room[t] / pool for t in counts}
        for t in counts:
            alloc[t] += min(room[t], int(share[t]))
        rest = sorted(counts, key=lambda t: (-(share[t] - int(share[t])), t))
        while sum(alloc.values()) < min(total, sum(counts.values())):
            moved = False
            for t in rest:
                if alloc[t] < counts[t] and sum(alloc.values()) < total:
                    alloc[t] += 1
                    moved = True
            if not moved:
                break
    return alloc


def build(events, done, source, seed=SEED, n_a=N_A, n_b=N_B, n_fb=N_FB):
    """返回 (items, 计数)。events：事件列表；done：初始切片键集合（住院号|文书序号|起点）；source：{住院号: 住院}。"""
    rng = random.Random(seed)
    events = [e for e in events if str(e["admission_id"]) in source]
    n_all = len(events)
    routed = collections.Counter((e.get("event_type"), e.get("assertion")) for e in events if not is_core(e))
    events = [e for e in events if is_core(e)]
    fb_all = [i for i, e in enumerate(events) if e.get("ground") == "name"]
    fb_pick = sorted(rng.sample(fb_all, min(n_fb, len(fb_all)))) if n_fb else []
    taken = set(fb_pick)
    by_type = collections.defaultdict(list)
    for i, e in enumerate(events):
        if i not in taken:
            by_type[e["event_type"]].append(i)
    alloc = allocate({t: len(v) for t, v in by_type.items()}, n_a, MIN_PER_TYPE)
    items = []
    picks = [(i, "fallback") for i in fb_pick] + [(i, "type") for t in sorted(by_type) for i in sorted(rng.sample(by_type[t], alloc[t]))]
    for i, stratum in picks:
        e = events[i]
        doc = source[str(e["admission_id"])]["docs"][e["doc_idx"]]
        a, b = e["seg_start"], e["seg_end"]
        items.append({"part": "A", "admission_id": str(e["admission_id"]), "doc_idx": e["doc_idx"], "doc_type": DOC_CN.get(doc.get("doc_type"), "病程记录"),
                      "section": e.get("section"), "text": doc["text"][a:b], "hl": [[e["ev_start"] - a, e["ev_end"] - a]],
                      "event": {k: e.get(k) for k in ("event_type", "name", "value", "unit", "assertion", "time_text")}, "stratum": stratum})
    # B：初始切片。切片位置用核心包同一个切法重算，和抽取时一致
    segs = []
    for key in sorted(done):
        rid, i, start = key.rsplit("|", 2)
        if rid not in source:
            continue
        doc = source[rid]["docs"][int(i)]
        seg = next((s for s in segments(doc) if s["start"] == int(start)), None)
        if seg:
            segs.append((rid, int(i), doc, seg))
    for rid, i, doc, seg in sorted(rng.sample(segs, min(n_b, len(segs))), key=lambda x: (x[0], x[1], x[3]["start"])):
        a, b = seg["start"], seg["end"]
        inside = [e for e in events if str(e["admission_id"]) == rid and e["doc_idx"] == i and a <= e["seg_start"] < b]
        items.append({"part": "B", "admission_id": rid, "doc_idx": i, "doc_type": DOC_CN.get(doc.get("doc_type"), "病程记录"), "section": seg["section"],
                      "text": seg["text"], "hl": [[e["ev_start"] - a, e["ev_end"] - a] for e in inside],
                      "events": [{k: e.get(k) for k in ("event_type", "name", "value", "unit", "assertion")} for e in inside]})
    a_items, b_items = [x for x in items if x["part"] == "A"], [x for x in items if x["part"] == "B"]
    rng.shuffle(a_items)
    rng.shuffle(b_items)
    items = a_items + b_items
    for n, x in enumerate(items):
        x["item_id"] = f"{x['part']}{n:03d}"
    counts = collections.Counter()
    counts["events_total"], counts["segments_total"] = len(events), len(segs)
    counts["events_before_core_filter"] = n_all
    for (t, a), n in routed.items():
        counts[f"routed_out:{t}|{a}"] = n
    for t, v in by_type.items():
        counts[f"events_total:{t}"] = len(v)
        counts[f"A_sampled:{t}"] = alloc[t]
    for x in b_items:
        counts[f"B_sampled:{x['doc_type']}"] += 1
    counts["A_sampled"], counts["B_sampled"] = len(a_items), len(b_items)
    counts["events_fallback_total"], counts["A_sampled_fallback"] = len(fb_all), len(fb_pick)
    for i in fb_all:
        counts[f"events_fallback:{events[i]['event_type']}"] += 1
    return items, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.umask(0o077)
    out = os.path.abspath(a.out)
    if not a.synthetic and not out.replace("\\", "/").startswith(DERIVED_ROOT):
        raise SafeError("out_not_in_derived_root")
    if os.path.exists(os.path.join(out, "items.jsonl")):
        raise SafeError("items_exist_do_not_resample")
    os.makedirs(out, exist_ok=True)
    with open(a.events, encoding="utf-8") as f:
        events = [json.loads(x) for x in f if x.strip()]
    with open(a.done, encoding="utf-8") as f:
        done = {x.strip() for x in f if x.strip()}
    rids = {k.rsplit("|", 2)[0] for k in done}
    source = {}
    with open(a.source, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                adm = json.loads(line)
                rid = str(adm.get("admission_id"))
                if rid in rids:
                    source[rid] = adm
    items, counts = build(events, done, source, a.seed)
    with open(os.path.join(out, "items.jsonl"), "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "sample_counts.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(sorted(counts.items()))
    log("gold_sample_done", items=len(items), A=counts["A_sampled"], B=counts["B_sampled"])


if __name__ == "__main__":
    guarded(main)
