# -*- coding: utf-8 -*-
"""kb_v1_agentic：17 个病种逐个走 8B 两阶段提问检索，出知识表和有出处率。

模型服务只认 127.0.0.1（medcourse.llm.Client 写死）。8101 在 ys 上只绑 127.0.0.1，本机跑时先开 SSH 隧道：
    ssh -o BatchMode=yes -N -L 18101:127.0.0.1:8101 ys
    python -m route_b.build_kb_agentic --base-url http://127.0.0.1:18101/v1 --model Qwen3-8B ...
送给模型的只有公开路径和指导原则的文字。

产物：<out>/kb_<id>.json、<out>/trace_<id>.json（每个需求的检索词、段落、轮数，公开数据）、<out>/grounded_by_field.csv、<out>/run.json
"""
import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import VERSION, schema
from .agentic import GATE, SURGICAL_FIELDS, build_disease
from .chunks import make_chunks
from .compare_kb import LIST_FIELDS, item_counts
from .retrieve import BM25

GUIDE_ID = "guideline_abx2015"


def load_keys(path):
    cfg = json.load(open(path, encoding="utf-8"))
    return [e["key"] for e in cfg["exams"]], {e["key"]: e.get("cls") for e in cfg["exams"]}


def source_meta(src, rel):
    d, fn = os.path.split(rel)
    for m in json.load(open(os.path.join(src, d, "SOURCES.json"), encoding="utf-8")):
        if m["file"] == fn:
            return m
    raise KeyError(rel)


def merge(base, new, fields, kit):
    """增量重跑（老板 11-11）：只换 fields 这几个字段（和门控），其余原样沿用上一版。"""
    out = json.loads(json.dumps(base))
    for fld in fields:
        out["facts"].pop(fld, None)
    out["facts"].update({fld: fs for fld, fs in new["facts"].items() if fld in fields})
    out["missing"] = [m for m in base.get("missing", []) if m["field"] not in fields] +                      [m for m in new.get("missing", []) if m["field"] in fields]
    out["merged"] = {"base_kit": base.get("kit"), "rerun_kit": kit, "rerun_fields": sorted(fields), "rerun_seconds": new["seconds"]}
    out["kit"] = kit
    out["validate"] = schema.validate(out)
    return out


def run_one(client, d, src, guide_text, guide_chunks, keys, key_cls, rounds, k, extra, only_fields=None, base_facts=None):
    text = open(os.path.join(src, d["file"]), encoding="utf-8").read()
    doc_id = d["id"]
    index = BM25(make_chunks(doc_id, text) + guide_chunks)
    t0 = time.time()
    drop = re.sub(r"[（(].*?[）)]", "", d["name"])      # 检索词里去掉的病种名（不含括号里的"儿科""经皮肾镜碎石术"）
    facts, log, missing = build_disease(client, index, d["name"], doc_id, {doc_id: text, GUIDE_ID: guide_text}, keys, key_cls,
                                        rounds, k, extra, drop=drop, only_fields=only_fields, base_facts=base_facts)
    m = source_meta(src, d["file"])
    kb = {"meta": {"pathway": d["name"], "version": m.get("version") or "2009-2017县级版", "icd10": d["id"],
                   "issuer": m.get("issuer") or "国家卫生健康委（中华医学会网站转载）", "url": m["url"], "retrieved": m["retrieved"],
                   "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
          "facts": facts, "missing": missing, "kit": VERSION, "method": "agentic_8b", "seconds": round(time.time() - t0, 1)}
    kb["validate"] = schema.validate(kb)
    return kb, log


def _row(name, kbs, flds):
    fs = [f for kb in kbs for fld in flds for f in kb["facts"].get(fld, [])]
    lf = [f for kb in kbs for fld in flds if fld in LIST_FIELDS for f in kb["facts"].get(fld, [])]
    n_items, g_items = item_counts(lf, "v1") if lf else ("", "")
    cells = len(kbs) * len(flds)
    return {"field": name, "diseases_with_fact": sum(1 for kb in kbs for fld in flds if kb["facts"].get(fld)), "cells": cells,
            "cells_grounded": sum(1 for kb in kbs for fld in flds if any(f["grounded"] for f in kb["facts"].get(fld, []))),
            "facts": len(fs), "missing": sum(1 for kb in kbs for m in kb.get("missing", []) if m["field"] in flds),
            "grounded": sum(f["grounded"] for f in fs),
            "model_prior": sum(f["source"] == "model_prior" and not f["quote_failed"] for f in fs),
            "quote_failed": sum(f["quote_failed"] for f in fs),
            "not_applicable": sum(f["source"] == "not_applicable" for f in fs),   # 0.4.1：门控答"不做"记的"不适用"
            "items": n_items, "items_in_quote": g_items}


def summarize(kbs):
    """逐字段一行，再加 A、B 两组合计（老板 11-11：两组分开算，不混成一个分母）。
    cells = 病种数 × 字段数；cells_grounded = 有至少一条有出处事实的格子；missing = 没给出合格答案的需求。"""
    rows = [_row(fld, kbs, [fld]) for fld in schema.GROUP_A + schema.GROUP_B]
    rows.append(_row("GROUP_A_路径应有项", kbs, schema.GROUP_A))
    rows.append(_row("GROUP_B_追问补全项", kbs, schema.GROUP_B))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原文目录（含 nhc2019_text、cma_county、abx_guideline_2015）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--exams", required=True, help="评测台 exams.json（54 个检查键）")
    ap.add_argument("--guideline", default="abx_guideline_2015/抗菌药物临床应用指导原则_2015.txt")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--extra", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", default="", help="逗号分隔的病种 id，空为全部")
    ap.add_argument("--fields", default="", help="增量重跑：逗号分隔的字段，只问这些（手术相关字段自动带门控）")
    ap.add_argument("--base", default="", help="增量重跑：上一版 kb 目录，其余字段从这里合并")
    a = ap.parse_args()
    from medcourse.llm import Client
    client = Client(a.base_url, a.model, timeout=300)
    keys, key_cls = load_keys(a.exams)
    guide_text = open(os.path.join(a.src, a.guideline), encoding="utf-8").read()
    guide_chunks = make_chunks(GUIDE_ID, guide_text)
    ds = json.load(open(a.config, encoding="utf-8"))["diseases"]
    if a.only:
        ds = [d for d in ds if d["id"] in a.only.split(",")]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    fields = [x for x in a.fields.split(",") if x]
    if fields and not a.base:
        raise SystemExit("--fields 要和 --base 一起用")
    if fields and set(fields) & set(SURGICAL_FIELDS):
        fields = fields + [GATE] if GATE not in fields else fields

    def job(d):
        base = json.load(open(os.path.join(a.base, f"kb_{d['id']}.json"), encoding="utf-8")) if fields else None
        kb, log = run_one(client, d, a.src, guide_text, guide_chunks, keys, key_cls, a.rounds, a.k, a.extra,
                          only_fields=fields or None, base_facts=base["facts"] if base else None)
        if base:
            kb = merge(base, kb, fields, VERSION)
        json.dump(kb, open(os.path.join(a.out, f"kb_{d['id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(log, open(os.path.join(a.out, f"trace_{d['id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("done", d["id"], kb["seconds"], flush=True)
        return kb

    with ThreadPoolExecutor(a.concurrency) as ex:
        kbs = list(ex.map(job, ds))
    rows = summarize(kbs)
    with open(os.path.join(a.out, "grounded_by_field.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"kit": VERSION, "model": a.model, "rounds": a.rounds, "k": a.k, "extra": a.extra, "diseases": len(kbs),
               "incremental_fields": fields, "base": a.base,
               "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.out, "run.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("build_kb_agentic_done", len(kbs))


if __name__ == "__main__":
    main()
