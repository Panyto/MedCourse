"""抽取主程序。逐片抽两趟，引文校验，顶到上限或被截断的切片对半重抽，可断点续跑。

python -B -m medcourse.extract.run --path 导出文件 --base-url http://127.0.0.1:8102/v1 --model Qwen3-14B-AWQ --out 目录 \
    [--filter-ids 文件] [--limit-adm N] [--passes 2] [--concurrency 16] [--synthetic]

真实文书的产物只能放 MC_DERIVED_ROOT（默认 /home/weify/pyt/product/derived/）下；合成行加 --synthetic 才能放别处。
events.jsonl 含原话片段，是真实派生材料，不出 ys。counts.json 只有计数，可以进包。
v0.3：引文和名称都对不上、被丢掉的候选另存 events_dropped.jsonl（同样是真实派生材料，不出 ys），计数按类型进 counts.json。
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..contract import CONTRACT_VERSION, validate_event
from ..llm import EXTRACT, Client, LLMError
from ..safe import SafeError, guarded, log
from . import schema
from .segments import halve, segments
from .verify import dedupe, ground

DERIVED_ROOT = os.environ.get("MC_DERIVED_ROOT", "/home/weify/pyt/product/derived/")
MAX_SPLIT = 2
DOC_CN = {"hospital_record": "入院记录", "discharge_summary": "出院记录", "progress_note": "病程记录"}


def doc_kind(doc):
    return doc.get("doc_type_cn") or DOC_CN.get(doc.get("doc_type"), "病程记录")


def _add(tally, lock, **kw):
    with lock:
        for k, v in kw.items():
            tally[k] = tally.get(k, 0) + v


def _ask(client, messages, sch, max_tokens, seed, tally, lock):
    r = client.generate(messages=messages, max_tokens=max_tokens, sampling=EXTRACT, seed=seed, schema=sch)
    _add(tally, lock, requests=1, prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
         finish_length=int(r["finish"] == "length"), finish_stop=int(r["finish"] == "stop"))
    try:
        return json.loads(r["text"]), r["finish"]
    except ValueError:
        _add(tally, lock, parse_fail=1)
        return None, r["finish"]


def extract_segment(client, seg, kind, passes, tally, lock, depth=0, sink=None):
    """一片的全部事件。顶到上限或被截断就对半重抽，最多两层；切不开的留下并标 truncated。sink（v0.3）收被丢掉的候选。"""
    _add(tally, lock, segments=1, seg_chars=len(seg["text"]))
    data, finish = _ask(client, schema.pass1_messages(seg, kind), schema.PASS1_SCHEMA, schema.PASS1_MAX_TOKENS, 20261102, tally, lock)
    raw = (data or {}).get("events") if isinstance(data, dict) else None
    over = finish == "length" or (raw is not None and len(raw) >= schema.CAP)
    if over and depth < MAX_SPLIT:
        parts = halve(seg)
        if parts:
            _add(tally, lock, seg_split=1)
            return [e for p in parts for e in extract_segment(client, p, kind, passes, tally, lock, depth + 1, sink)]
    truncated = over
    if truncated:
        _add(tally, lock, seg_truncated=1)
    if raw is None:
        return []
    _add(tally, lock, events_raw=len(raw), seg_at_cap=int(len(raw) >= schema.CAP))
    local = {}
    grounded = [(c, ground(c, seg, local)) for c in raw]
    if sink is not None:
        sink.extend(dict(c, section=seg["section"], seg_start=seg["start"], seg_end=seg["end"]) for c, g in grounded if g is None and isinstance(c, dict))
    kept = dedupe([g for _, g in grounded if g is not None])
    _add(tally, lock, events_kept=len(kept), **local)
    for e in kept:
        e.update({"section": seg["section"], "seg_start": seg["start"], "seg_end": seg["end"], "truncated": truncated,
                  "assertion": None, "time_text": None})
    if passes >= 2 and kept:
        data2, _ = _ask(client, schema.pass2_messages(seg, kind, kept), schema.PASS2_SCHEMA, schema.PASS2_MAX_TOKENS, 20261103, tally, lock)
        items = (data2 or {}).get("items") if isinstance(data2, dict) else None
        seen = 0
        for it in items or []:
            i = it.get("id")
            if isinstance(i, int) and 0 <= i < len(kept):
                kept[i]["assertion"], kept[i]["time_text"], kept[i]["reason"] = it.get("assertion"), it.get("time_text"), it.get("reason")
                seen += 1
        _add(tally, lock, pass2_answered=seen, pass2_missing=len(kept) - seen)
    bad = [e for e in kept if validate_event(e)]
    if bad:
        _add(tally, lock, events_invalid=len(bad))
    return [e for e in kept if not validate_event(e)]


def work_items(lines):
    for adm in lines:
        rid = str(adm.get("admission_id") or adm.get("slot_id"))
        for i, doc in enumerate(adm.get("docs") or []):
            if not doc.get("doc_type") and doc.get("doc_type_cn"):
                doc = dict(doc, doc_type={"入院记录": "hospital_record", "出院记录": "discharge_summary"}.get(doc["doc_type_cn"], "progress_note"))
            for seg in segments(doc):
                yield rid, i, doc, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter-ids")
    ap.add_argument("--limit-adm", type=int)
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.umask(0o077)
    out = os.path.abspath(a.out)
    if not a.synthetic and not out.replace("\\", "/").startswith(DERIVED_ROOT):
        raise SafeError("out_not_in_derived_root")
    os.makedirs(out, exist_ok=True)
    keep = None
    if a.filter_ids:
        with open(a.filter_ids, encoding="utf-8") as f:
            keep = {x.strip() for x in f if x.strip()}
    lines = []
    with open(a.path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            adm = json.loads(line)
            if keep is None or str(adm.get("admission_id") or adm.get("slot_id")) in keep:
                lines.append(adm)
            if a.limit_adm and len(lines) >= a.limit_adm:
                break
    done_path, events_path = os.path.join(out, "done_segments.txt"), os.path.join(out, "events.jsonl")
    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done = {x.strip() for x in f if x.strip()}
    client = Client(a.base_url, a.model)
    tally, lock = {"admissions": len(lines), "skipped_done": 0, "llm_errors": 0}, threading.Lock()
    t0 = time.perf_counter()

    def one(item):
        rid, i, doc, seg = item
        key = f"{rid}|{i}|{seg['start']}"
        if key in done:
            _add(tally, lock, skipped_done=1)
            return
        sink = []
        try:
            events = extract_segment(client, seg, doc_kind(doc), a.passes, tally, lock, sink=sink)
        except LLMError as e:
            _add(tally, lock, llm_errors=1)
            log("llm_error", code=str(e), id=rid, doc=i)
            return
        with lock:
            with open(events_path, "a", encoding="utf-8") as f:
                for e in events:
                    f.write(json.dumps({"contract": CONTRACT_VERSION, "admission_id": rid, "doc_idx": i, "day_offset": doc.get("day_offset"),
                                        "doc_type": doc.get("doc_type"), **e}, ensure_ascii=False) + "\n")
            with open(os.path.join(out, "events_dropped.jsonl"), "a", encoding="utf-8") as f:
                for c in sink:
                    f.write(json.dumps({"contract": CONTRACT_VERSION, "admission_id": rid, "doc_idx": i, "doc_type": doc.get("doc_type"), **c},
                                       ensure_ascii=False) + "\n")
            with open(done_path, "a", encoding="utf-8") as f:
                f.write(key + "\n")

    with ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(one, work_items(lines)))
    tally["wall_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(out, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(tally, f, ensure_ascii=False, indent=1, sort_keys=True)
    log("extract_done", **{k: v for k, v in tally.items() if isinstance(v, (int, float))})
    if tally["llm_errors"]:
        raise SafeError("llm_errors", n=tally["llm_errors"])


if __name__ == "__main__":
    guarded(main)
