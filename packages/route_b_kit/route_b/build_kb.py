# -*- coding: utf-8 -*-
"""【消融靶子】按病种清单用正则抽 kb_v0（不用模型），出有出处率汇总。路线 B 正式版见 build_kb_agentic.py。

python -m route_b.build_kb --src <原文目录> --config config/diseases_v0.json --exams <评测台 exams.json> --out <输出目录>
产物：<out>/kb_<id>.json（每病种一份）、<out>/grounded_by_field.csv、<out>/coverage.csv、<out>/versions.csv
"""
import argparse
import csv
import hashlib
import json
import os

from . import VERSION, schema
from .extract import extract

FIELDS = schema.PATHWAY_FIELDS + schema.GAP_FIELDS


def load_alias(path):
    cfg = json.load(open(path, encoding="utf-8"))
    return {a: e["key"] for e in cfg["exams"] for a in sorted(e["aliases"], key=len, reverse=True)}


def source_meta(src, rel):
    """从 SOURCES.json 找这份文件的出处。"""
    d, fn = os.path.split(rel)
    for m in json.load(open(os.path.join(src, d, "SOURCES.json"), encoding="utf-8")):
        if m["file"] == fn:
            return m
    raise KeyError(rel)


def build_one(src, rel, alias):
    text = open(os.path.join(src, rel), encoding="utf-8").read()
    facts, info = extract(text, alias)
    m = source_meta(src, rel)
    meta = {"pathway": os.path.splitext(os.path.basename(rel))[0] if rel.startswith("nhc") else m.get("title"),
            "version": m.get("version") or "2009-2017县级版", "icd10": info["icd10"] or "",
            "issuer": m.get("issuer") or "国家卫生健康委（中华医学会网站转载）", "url": m["url"], "retrieved": m["retrieved"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return {"meta": meta, "facts": facts, "parse": info, "kit": VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exams", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    alias = load_alias(a.exams)
    cfg = json.load(open(a.config, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    cov, versions = [], []
    tot = {f: [0, 0, 0] for f in FIELDS}   # 有这条的病种数、条数、有出处条数
    for d in cfg["diseases"]:
        kb = build_one(a.src, d["file"], alias)
        errs = schema.validate(kb)
        kb["validate"] = errs
        with open(os.path.join(a.out, f"kb_{d['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=1)
        row = {"id": d["id"], "name": d["name"], "kind": d["kind"], "version": kb["meta"]["version"], "icd10": kb["meta"]["icd10"],
               "sections": len(kb["parse"]["sections"]), "form": int(kb["parse"]["form"]), "validate_errors": len(errs)}
        for fld in FIELDS:
            fs = kb["facts"].get(fld, [])
            row[fld] = f"{sum(1 for x in fs if x['grounded'])}/{len(fs)}" if fs else ""
            tot[fld][0] += bool(fs)
            tot[fld][1] += len(fs)
            tot[fld][2] += sum(1 for x in fs if x["grounded"])
        cov.append(row)
        if d.get("county_alt"):
            alt = build_one(a.src, d["county_alt"], alias)
            g = lambda k, f: (k["facts"].get(f) or [{}])[0].get("value", "")  # noqa: E731
            versions.append({"id": d["id"], "field": "los_max", "v2019": g(kb, "los_max"), "county": g(alt, "los_max")})
            versions.append({"id": d["id"], "field": "surgery_day_max", "v2019": g(kb, "surgery_day_max"), "county": g(alt, "surgery_day_max")})
            versions.append({"id": d["id"], "field": "anesthesia", "v2019": g(kb, "anesthesia"), "county": g(alt, "anesthesia")})
    n = len(cfg["diseases"])
    with open(os.path.join(a.out, "coverage.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0]))
        w.writeheader()
        w.writerows(cov)
    with open(os.path.join(a.out, "grounded_by_field.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(("field", "group", "diseases_with_fact", "diseases", "facts", "grounded_facts"))
        for fld in FIELDS:
            w.writerow((fld, "pathway" if fld in schema.PATHWAY_FIELDS else "gap", tot[fld][0], n, tot[fld][1], tot[fld][2]))
    with open(os.path.join(a.out, "versions.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("id", "field", "v2019", "county"))
        w.writeheader()
        w.writerows(versions)
    print("build_kb_done", n)


if __name__ == "__main__":
    main()
