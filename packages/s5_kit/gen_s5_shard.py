# -*- coding: utf-8 -*-
"""Stage-1 GPU/template generation worker reading pre-planned journeys/specs."""
import argparse
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from s5_common import PATHWAY_SPECS, build_oracle_prompt, build_pathway_stats, setup_paths

setup_paths()

from medcourse import compiler, gate, kb, pipeline, planner, writer  # noqa: E402
from medcourse.contract import admission_line  # noqa: E402
from medcourse.llm import LLMError, WRITE  # noqa: E402
from medcourse.safe import SafeError, guarded, log  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stats_for_arm(arm, stats_a, kb_data):
    if arm in ("full", "vis_only", "oracle_leak"):
        return dict(stats_a, _kb=kb_data)
    if arm == "route_a_only":
        return dict(stats_a)
    if arm == "route_ab":
        return dict(stats_a, _kb=kb_data)
    if arm == "route_b":
        return kb.build_b_only_stats(kb_data)
    raise ValueError(f"Unknown arm: {arm}")


def _oracle_write_doc(j, spec, specs, stats, w, rng, tally):
    plans = compiler.compile_doc(j, spec, specs, stats, rng, ctx=None, epistemic=False)
    texts = {}
    for pl in plans:
        if pl["mode"] == "program":
            continue
        tally["sections"] = tally.get("sections", 0) + 1
        max_tokens = min(1600, int(pl["target_chars"] * 1.8) + 140)
        seed = planner.seed_of(j["journey_id"], spec["k"], pl["section"], "oracle") % (2 ** 31)
        prompt = build_oracle_prompt(j, spec, pl["section"], pl["target_chars"])
        if isinstance(w, writer.TemplateWriter):
            out = w.write(j, spec, pl, 0)
        else:
            r = w.client.generate(
                messages=[{"role": "system", "content": writer.SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                sampling=WRITE,
                seed=seed,
            )
            w.calls += 1
            w.prompt_tokens += r["prompt_tokens"]
            w.completion_tokens += r["completion_tokens"]
            w.seconds += r["seconds"]
            out = writer.clean(r["text"], pl["section"])
        sec_fails = gate.check_section(pl, out)
        fixable = [c for c in sec_fails if c in gate.PIPELINE_SAFE_FIX]
        if fixable:
            out = gate.repair_section(out, fixable, pl)
        texts[pl["section"]] = out
    return compiler.assemble(j, spec, specs, plans, texts)


def generate_from_plan(j, specs, stats_map, arm, w, seed, tally, save_first_try=False):
    jid = j["journey_id"]
    if arm in ("route_b", "route_a_only", "route_ab") and j.get("kb_pathway_key"):
        stats = stats_map[j["kb_pathway_key"]]
    else:
        stats = stats_map["_default"]

    rng = random.Random(planner.seed_of(seed, jid, "write"))
    if arm == "oracle_leak":
        docs = [
            dict(
                {
                    "doc_type": spec["doc_type"], "day_offset": spec["day"],
                    "text": _oracle_write_doc(j, spec, specs, stats, w, rng, tally),
                    "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"],
                },
                **({"time": spec["time"]} if spec.get("time") else {}),
            )
            for spec in specs
        ]
        return docs, [], None

    docs, first_docs, note_bodies, ctx = [], ([] if save_first_try else None), {}, {}
    for spec in specs:
        fbox = [] if save_first_try else None
        hf = []
        text, texts = pipeline.write_doc(
            j, spec, specs, stats, w, rng, tally,
            pipeline._prev_ctx(spec["k"], specs, note_bodies),
            ctx=ctx, first_doc_out=fbox, hard_fails_out=hf,
        )
        doc_rec = {
            "doc_type": spec["doc_type"], "day_offset": spec["day"],
            "text": text, "doc_type_cn": spec["doc_type_cn"],
            "k": spec["k"], "role": spec["role"],
            "_hard_sec_fails": sorted({c for _, c in hf}),
        }
        if spec.get("time"):
            doc_rec["time"] = spec["time"]
        docs.append(doc_rec)
        if save_first_try:
            fdoc_rec = {
                "doc_type": spec["doc_type"], "day_offset": spec["day"],
                "text": fbox[0], "doc_type_cn": spec["doc_type_cn"],
                "k": spec["k"], "role": spec["role"],
            }
            if spec.get("time"):
                fdoc_rec["time"] = spec["time"]
            first_docs.append(fdoc_rec)
        if spec["role"] == "HR":
            ctx = pipeline.hr_context(texts)
        src = pipeline._note_copy_source(spec["role"], texts)
        if src:
            note_bodies[spec["k"]] = src

    fails = gate.check_admission(j, docs)
    tally["first_pass"] = tally.get("first_pass", 0) + (not fails)
    for code in sorted({c for _, c in fails}):
        tally["first_fail_" + code] = tally.get("first_fail_" + code, 0) + 1
    for r in range(1, pipeline.MAX_DOC_REWRITES + 1):
        if not fails:
            break
        tally["adm_rewrites"] = tally.get("adm_rewrites", 0) + 1
        ks = {k for k, _ in fails if k is not None} or {d["k"] for d in docs}
        ks = pipeline.expand_rewrite_deps(ks, specs)
        for x in ks:
            note_bodies.pop(x, None)
        for k in sorted(ks):
            spec = specs[k]
            hf = []
            text, texts = pipeline.write_doc(
                j, spec, specs, stats, w, rng, tally,
                pipeline._prev_ctx(k, specs, note_bodies), r, ctx=ctx, hard_fails_out=hf,
            )
            docs[k] = dict(docs[k], text=text, _hard_sec_fails=sorted({c for _, c in hf}))
            if spec["role"] == "HR":
                ctx = pipeline.hr_context(texts)
            src = pipeline._note_copy_source(spec["role"], texts)
            if src:
                note_bodies[k] = src
        fails = gate.check_admission(j, docs)
    for d in docs:
        d.pop("_hard_sec_fails", None)
    if fails:
        return None, sorted({c for _, c in fails}), first_docs
    return docs, [], first_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--arm", required=True, choices=("full", "vis_only", "oracle_leak", "route_b", "route_a_only", "route_ab"))
    ap.add_argument("--stats", default="/home/weify/pyt/product/derived/m2_f27_v021/stats_v021.json")
    ap.add_argument("--writer", default="llm", choices=("llm", "template"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8104/v1")
    ap.add_argument("--model", default="Qwen3-32B-AWQ")
    ap.add_argument("--seed", type=int, default=20261210)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--save-first-try", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.umask(0o077)
    os.makedirs(a.out, exist_ok=True)
    kb_data = kb.load_default_kb()
    stats_a = None if a.arm == "route_b" else json.load(open(a.stats, encoding="utf-8"))
    stats_map = {"_default": _stats_for_arm(a.arm, stats_a, kb_data)}
    if a.arm in ("route_b", "route_a_only", "route_ab"):
        for row in PATHWAY_SPECS:
            kb_key = row[0]
            stats_map[kb_key] = build_pathway_stats(a.arm, kb_key, stats_a=stats_a, kb_data=kb_data)

    js = _read_jsonl(os.path.join(a.plan_dir, "journeys.jsonl"))
    specs_by_id = {x["slot_id"]: x["specs"] for x in _read_jsonl(os.path.join(a.plan_dir, "specs.jsonl"))}
    work = [(j, specs_by_id[j["journey_id"]]) for j in js]

    if a.writer == "template":
        w = writer.TemplateWriter()
    else:
        w = writer.LLMWriter(a.base_url, a.model, raw=False, num_placeholder=False, splice_copy=False)
    w.no_filler2 = True
    w.ablate_epistemic = (a.arm == "vis_only")

    tally, discards, lock = {"planned": len(work), "kept": 0, "discarded": 0, "llm_errors": 0}, {}, threading.Lock()
    t0 = time.perf_counter()
    save_ft = bool(a.save_first_try and a.arm != "oracle_leak")
    out_names = ("admissions", "journeys", "drafts") + (("first_try_admissions",) if save_ft else ())
    paths = {x: open(os.path.join(a.out, f"{x}.jsonl"), "w", encoding="utf-8") for x in out_names}

    def one(item):
        j, specs = item
        local = {}
        try:
            docs, fails, fdocs = generate_from_plan(j, specs, stats_map, a.arm, w, a.seed, local, save_first_try=save_ft)
        except LLMError as e:
            with lock:
                tally["llm_errors"] += 1
            log("llm_error", code=str(e), id=j["journey_id"])
            return
        with lock:
            for k, v in local.items():
                tally[k] = tally.get(k, 0) + v
            if docs is None:
                tally["discarded"] += 1
                stratum = (j["patient"]["primary_dx"]["code3"][:1] if j else "?") or "?"
                for code in fails:
                    discards[(stratum, code)] = discards.get((stratum, code), 0) + 1
                return
            tally["kept"] += 1
            tally["chars"] = tally.get("chars", 0) + sum(len(d["text"]) for d in docs)
            tally["docs"] = tally.get("docs", 0) + len(docs)
            paths["admissions"].write(json.dumps(admission_line(j, docs), ensure_ascii=False) + "\n")
            paths["journeys"].write(json.dumps(j, ensure_ascii=False) + "\n")
            paths["drafts"].write(json.dumps({"slot_id": j["journey_id"], "draft": planner.journey_to_draft(j)}, ensure_ascii=False) + "\n")
            if save_ft and fdocs is not None:
                paths["first_try_admissions"].write(json.dumps(admission_line(j, fdocs), ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(one, work))
    for f in paths.values():
        f.close()

    wall = time.perf_counter() - t0
    tally.update({
        "wall_s": round(wall, 1),
        "calls": w.calls,
        "prompt_tokens": w.prompt_tokens,
        "completion_tokens": w.completion_tokens,
        "calls_per_kchar": round(1000 * w.calls / tally["chars"], 3) if tally.get("chars") else None,
        "calls_per_adm": round(w.calls / max(1, tally["kept"]), 2),
        "first_pass_rate": round(tally.get("first_pass", 0) / max(1, tally["planned"]), 4),
        "discard_rate": round(tally["discarded"] / max(1, tally["planned"]), 4),
        "version": pipeline.VERSION,
        "s5_arm": a.arm,
        "seed": a.seed,
        "writer": a.writer,
        "ablate_epistemic": w.ablate_epistemic,
        "model": a.model if a.writer == "llm" else None,
    })
    with open(os.path.join(a.out, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(tally, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(os.path.join(a.out, "discard.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["chapter", "code", "n"])
        cw.writerows(sorted((s, c, n) for (s, c), n in discards.items()))
    log("s5_shard_done", arm=a.arm, kept=tally["kept"], discarded=tally["discarded"], calls=w.calls, wall_s=round(wall, 1))
    if tally["llm_errors"]:
        raise SafeError("llm_errors", n=tally["llm_errors"])


if __name__ == "__main__":
    guarded(main)
