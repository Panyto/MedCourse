# -*- coding: utf-8 -*-
"""Stage-1 & Stage-2 intrinsic structural, safety, oracle2-leakage, and diversity audit.

Emits strictly aggregate numerical metrics (zero raw patient or synthetic spans).
"""
import argparse
import collections
import csv
import json
import os
import re
import statistics

from s5_common import setup_paths
from t123_bench import EARLY_OP_RE, LATE_OP_RE, DOSE_RE, _completed_shard_paths, _is_discharge, _is_hr, _resolve_direct32b_paths

setup_paths()

from medcourse import gate, planner  # noqa: E402
from medcourse.synth_export import norm  # noqa: E402
from b7_rules import check_record  # noqa: E402
from g2_kit import oracle2  # noqa: E402
from q45_kit.realtext import care7_from_admission  # noqa: E402
from ruler_g1 import evaluate_document, extract_facts, is_first_progress, split_sections  # noqa: E402
from ruler_g2 import build_real, evaluate_admission  # noqa: E402
from ruler_g2.events import EXAM_RE, tone_of  # noqa: E402
from ruler_g3 import evaluate as g3_evaluate  # noqa: E402
from ruler_g3.adapters import from_admission  # noqa: E402

import numpy as np  # noqa: E402
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


_PAST_SKIP_RE = re.compile(r"既往|年前|曾|否认|入院前|外院")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_sample(adm_paths, journey_paths=None, max_stays=500, seed=20261201):
    adms = []
    for p in adm_paths:
        if p and os.path.exists(p):
            adms.extend(_read_jsonl(p))
    if max_stays and len(adms) > max_stays:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(adms), size=max_stays, replace=False)
        adms = [adms[i] for i in sorted(idx)]
    jmap = {}
    for jp in journey_paths or ():
        if jp and os.path.exists(jp):
            for j in _read_jsonl(jp):
                jmap[j["journey_id"]] = j
    return adms, jmap


def _oracle_docs(adm):
    out = []
    for idx, d in enumerate(adm["docs"]):
        role = d.get("role")
        if not role:
            dt = d.get("doc_type")
            if dt == "hospital_record":
                role = "HR"
            elif dt == "discharge_summary":
                role = "DS"
            else:
                role = "FP" if is_first_progress(d.get("text", "")) else "DAILY"
        out.append({
            "k": d.get("k", idx),
            "role": role,
            "day": int(d.get("day_offset") or 0),
            "text": d.get("text") or "",
        })
    return out


def _admit_dx_key(adm):
    hr = next((d for d in adm["docs"] if _is_hr(d)), None)
    if not hr:
        return None
    secs = {s.name: s.text for s in split_sections(hr["text"])}
    adx = secs.get("入院诊断", "").strip().splitlines()
    if not adx:
        return None
    first = re.sub(r"^\d+[\.、\s]*", "", adx[0]).strip()
    nrm = norm(first)[:16]
    return nrm if len(nrm) >= 2 else None


def _mean_nn_cos(texts):
    if not HAS_SKLEARN or len(texts) < 10:
        return None
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000, sublinear_tf=True)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return None
    if X.shape[1] == 0:
        return None
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, -1.0)
    return round(float(np.mean(np.max(sim, axis=1))), 4)


def audit_corpus(name, adm_paths, journey_paths=None, max_stays=500, seed=20261201):
    """Compute aggregate clinical safety, structural, oracle2, and diversity metrics for one corpus."""
    adms, jmap = _load_sample(adm_paths, journey_paths, max_stays=max_stays, seed=seed)
    n = len(adms)
    if n == 0:
        return {"arm": name, "n_stays": 0}

    doc_counts, char_counts, los_list = [], [], []
    surg_stays = dose_stays = 0
    g3_hard = n1_hits = g1_hits = b7_hits = g2_s_hits = k4_leaks = 0
    pe_hist_hits = pe_res_hits = d0_poststate_hits = late_op_elig = 0
    oracle_cnt = collections.Counter()
    oracle_n = 0
    dx_keys = set()
    stay_texts = []

    for idx, a in enumerate(adms):
        rid = str(a.get("admission_id") or f"{name}-{idx}")
        docs = a.get("docs") or []
        doc_counts.append(len(docs))
        full_txt = "\n".join(d.get("text") or "" for d in docs)
        char_counts.append(len(full_txt))
        stay_texts.append(norm(full_txt)[:4000])
        days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
        los_list.append(max(days) if days else 0)

        d01_txt = "\n".join(
            d.get("text") or "" for d in docs
            if d.get("day_offset") is not None and int(d["day_offset"]) <= 1 and not _is_discharge(d)
        )
        d2p_txt = "\n".join(
            d.get("text") or "" for d in docs
            if d.get("day_offset") is not None and int(d["day_offset"]) >= 2 and not _is_discharge(d)
        )
        has_surg = bool(EARLY_OP_RE.search(full_txt) or LATE_OP_RE.search(full_txt))
        surg_stays += int(has_surg)
        dose_stays += int(bool(DOSE_RE.search(full_txt)))

        dx_k = _admit_dx_key(a)
        if dx_k:
            dx_keys.add(dx_k)

        res_g3 = g3_evaluate(from_admission(a, rid))
        if res_g3.get("hits"):
            g3_hard += 1
        if res_g3.get("n1"):
            n1_hits += 1

        if any(
            c.pair_id in gate.G1_GATE
            for di, d in enumerate(docs)
            for c in evaluate_document(d.get("text") or "", rid, di, gate._doc_kind(d))["conflicts"]
        ):
            g1_hits += 1

        rec, _ = care7_from_admission(a, split_sections, extract_facts)
        if rec is not None and any(h.get("rule") != "AGE-UNPARSED" for h in check_record(rec)):
            b7_hits += 1

        rs = evaluate_admission(build_real(a))["readings"]
        if any(rs[k]["adm_conflict"] for k in gate.G2_GATE if k in rs):
            g2_s_hits += 1

        ds = next((d for d in reversed(docs) if _is_discharge(d)), None)
        k4_hit = bool(rs.get("K4", {}).get("adm_conflict"))
        if not k4_hit and ds:
            ds_secs = {s.name: s.text for s in split_sections(ds.get("text") or "")}
            adm_sec = ds_secs.get("入院情况", "")
            if adm_sec and gate.admission_leaks(adm_sec):
                k4_hit = True
        k4_leaks += int(k4_hit)

        hr = next((d for d in docs if _is_hr(d)), None)
        if hr:
            hr_secs = {s.name: s.text for s in split_sections(hr.get("text") or "")}
            pe_txt = "\n".join(hr_secs.get(sn, "") for sn in ("体格检查", "专科情况") if hr_secs.get(sn))
            if pe_txt:
                if any(w in pe_txt for w in gate._PE_HISTORY_WORDS):
                    pe_hist_hits += 1
                if any(tone_of(pe_txt, m.start(), m.end()) == "result" for m in EXAM_RE.finditer(pe_txt)):
                    pe_res_hits += 1

        if (max(days) if days else 0) >= 3 and LATE_OP_RE.search(d2p_txt) and not EARLY_OP_RE.search(d01_txt):
            late_op_elig += 1
            bad_d01 = False
            for cl in re.split(r"[，,。；;！？!?\n]", d01_txt):
                if cl and gate._BEFORE_OP_HARD.search(cl) and not _PAST_SKIP_RE.search(cl):
                    bad_d01 = True
                    break
            d0_poststate_hits += int(bad_d01)

        j = jmap.get(rid)
        if j is not None:
            oracle_n += 1
            draft = planner.journey_to_draft(j)
            _, o_sum = oracle2.oracle_leaks_v2(draft, None, _oracle_docs(a))
            for ok in ("O", "A", "B", "B_loose", "any"):
                oracle_cnt[ok] += int(bool(o_sum.get(ok)))

    row = {
        "arm": name,
        "n_stays": n,
        "mean_docs": round(float(np.mean(doc_counts)), 2),
        "mean_chars": round(float(np.mean(char_counts)), 1),
        "median_los": round(float(statistics.median(los_list)), 1) if los_list else 0.0,
        "surgical_stay_rate": round(surg_stays / n, 4),
        "dose_mention_rate": round(dose_stays / n, 4),
        "g3_hard_stay_rate": round(g3_hard / n, 4),
        "n1_stay_rate": round(n1_hits / n, 4),
        "g1_conflict_stay_rate": round(g1_hits / n, 4),
        "b7_fail_stay_rate": round(b7_hits / n, 4),
        "g2_s_conflict_rate": round(g2_s_hits / n, 4),
        "k4_adm_leak_rate": round(k4_leaks / n, 4),
        "pe_history_rate": round(pe_hist_hits / n, 4),
        "pe_result_rate": round(pe_res_hits / n, 4),
        "late_op_elig_n": late_op_elig,
        "day0_preop_poststate_leak_rate": round(d0_poststate_hits / late_op_elig, 4) if late_op_elig else None,
        "oracle_n": oracle_n,
        "oracle_O_rate": round(oracle_cnt["O"] / oracle_n, 4) if oracle_n else None,
        "oracle_A_rate": round(oracle_cnt["A"] / oracle_n, 4) if oracle_n else None,
        "oracle_B_strict_rate": round(oracle_cnt["B"] / oracle_n, 4) if oracle_n else None,
        "oracle_B_loose_rate": round(oracle_cnt["B_loose"] / oracle_n, 4) if oracle_n else None,
        "oracle_any_rate": round(oracle_cnt["any"] / oracle_n, 4) if oracle_n else None,
        "mean_nn_tfidf_cos": _mean_nn_cos(stay_texts),
        "unique_admit_dx_count": len(dx_keys),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--real-fold", default="/home/weify/pyt/data/deid_run9_fold1.jsonl")
    ap.add_argument("--direct32b-jsonl", default=None)
    ap.add_argument("--max-per-arm", type=int, default=500)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    fa, fj = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400)
    va, vj = _completed_shard_paths(args.s5_root, "s5_1", "vis_only_shard", min_kept=400)
    oa, oj = _completed_shard_paths(args.s5_root, "s5_1", "oracle_leak_shard", min_kept=400)
    d32 = _resolve_direct32b_paths(args.s5_root, args.direct32b_jsonl)

    specs = []
    if args.real_fold and os.path.exists(args.real_fold):
        specs.append(("Real-f1", [args.real_fold], None))
    if fa:
        specs.append(("Synth-Full", fa, fj))
    if va:
        specs.append(("Synth-VisibilityOnly", va, vj))
    if oa:
        specs.append(("Synth-OracleLeak", oa, oj))
    if d32:
        specs.append(("DIRECT-32B", d32, None))

    for label, sub in (("Route-B", "route_b"), ("Route-A-Only", "route_a_only"), ("Route-AB", "route_ab")):
        sdir = os.path.join(args.s5_root, "generated", "s5_2", sub)
        adm_p = os.path.join(sdir, "admissions.jsonl")
        j_p = os.path.join(sdir, "journeys.jsonl")
        if os.path.exists(adm_p):
            specs.append((label, [adm_p], [j_p] if os.path.exists(j_p) else None))

    rows = [audit_corpus(nm, apths, jpth, max_stays=args.max_per_arm) for nm, apths, jpth in specs]

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"arms": rows}, f, ensure_ascii=False, indent=2)

    if rows:
        cols = list(rows[0].keys())
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print("INTRINSIC_AUDIT_DONE", json.dumps({"arms": len(rows), "out_json": args.out_json}, ensure_ascii=False))


if __name__ == "__main__":
    main()
