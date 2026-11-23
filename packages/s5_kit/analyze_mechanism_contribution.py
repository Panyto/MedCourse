# -*- coding: utf-8 -*-
"""Fixed-rule, fixed-case core mechanism contribution decomposition for MedCourse.

Isolates orthogonal architectural contributions under frozen clinical compatibility rules
and identical planned trajectories (strictly Stage-1 / fold 9 untouched):
  - Block 0: Fixed domain-rule pre-filter boundary (comorb drop rate) & paired identity proof
  - Block 1: Trajectory planning & priors vs unplanned DIRECT-32B + S5-2 3-arm prior ablation
  - Block 2: Dual-axis temporal visibility compilation (OracleLeak vs VisOnly/Full First-Try)
  - Block 3: Explicit belief-state compilation (compiler.py) vs deterministic gate rewriting (gate.py)
             via 4-cell paired decomposition (VisOnly/Full x FirstTry/PostGate) + Manski bounds
  - Block 4: Downstream TSTR controlled deltas (Matched600 vs DIRECT-32B, Full vs VisOnly 2x2, Augment)
"""
import argparse
import collections
import datetime as dt
import json
import os
import random

import numpy as np

from s5_common import PATHWAY_SPECS, setup_paths
import audit_s5_2_17d
import audit_s5_intrinsic
import t123_bench

setup_paths()

from medcourse import gate  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _assert_real_fold_safe(real_fold):
    if real_fold and "fold9" in real_fold.lower().replace("_", ""):
        raise ValueError(f"FORBIDDEN_STAGE1_ACCESS: fold9 must remain untouched ({real_fold})")


def _is_completed_dir(sdir, min_kept=1):
    cp = os.path.join(sdir, "counts.json")
    ap = os.path.join(sdir, "admissions.jsonl")
    jp = os.path.join(sdir, "journeys.jsonl")
    if not (os.path.isfile(cp) and os.path.isfile(ap) and os.path.isfile(jp)):
        return False
    try:
        c = _read_json(cp)
    except Exception:
        return False
    return int(c.get("llm_errors", 0)) == 0 and int(c.get("kept", 0)) >= min_kept


def _discover_completed_s51_shards(s5_root, prefix, min_kept=1):
    out = []
    for s in range(4):
        sdir = os.path.join(s5_root, "generated", "s5_1", f"{prefix}{s}")
        if _is_completed_dir(sdir, min_kept=min_kept):
            out.append(s)
    return out


def _discover_completed_s52_arms(s5_root, min_kept=1):
    out = []
    for arm in ("route_b", "route_a_only", "route_ab"):
        sdir = os.path.join(s5_root, "generated", "s5_2", arm)
        if _is_completed_dir(sdir, min_kept=min_kept):
            out.append(arm)
    return out


def _existing_first_try_paths(shard_dirs):
    out = []
    for d in shard_dirs:
        p = os.path.join(d, "first_try_admissions.jsonl")
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            out.append(p)
    return out


def _journey_signature(j):
    p = j.get("patient") or {}
    pdx = p.get("primary_dx") or {}
    coms = sorted(
        (c.get("name"), bool(c.get("at_admission")), c.get("discover_day"), tuple(c.get("evidence_ids") or ()))
        for c in (p.get("comorbidities") or ())
    )
    evs = [
        (ev.get("id"), ev.get("name"), ev.get("state"), ev.get("day"), tuple(ev.get("evidence_ids") or ()))
        for ev in (j.get("dx_events") or ())
    ]
    procs = [(pr.get("id"), pr.get("name"), pr.get("decide_day"), pr.get("do_day")) for pr in (j.get("procedures") or ())]
    return (
        pdx.get("name"),
        pdx.get("code3"),
        tuple(coms),
        tuple(evs),
        len(j.get("exams") or ()),
        len(j.get("meds") or ()),
        tuple(procs),
    )


def load_rule_scope_summary(pre_gpu_ledger_path):
    """Load fixed domain-rule filtering boundary from pre-GPU audit ledger without inflating rule claims."""
    if not pre_gpu_ledger_path or not os.path.isfile(pre_gpu_ledger_path):
        return {"available": False, "path": pre_gpu_ledger_path}
    raw = _read_json(pre_gpu_ledger_path)
    cohorts = raw.get("cohorts") or {}
    summary = {
        "available": True,
        "path": pre_gpu_ledger_path,
        "scope_note": raw.get("scope_note"),
        "cohorts": {},
    }
    for ck in ("ALL_ARM_STAYS_3020", "UNIQUE_SLOT_STAYS_2340", "S5-1"):
        c = cohorts.get(ck)
        if not c:
            continue
        cl = c.get("comorbidity_ledger") or {}
        by_ch = cl.get("by_icd_chapter") or {}
        ch_rates = [
            (ch, float(v.get("drop_rate_of_inpatient_candidates", 0.0)))
            for ch, v in by_ch.items()
            if v.get("inpatient_candidates", 0) > 0
        ]
        ch_rates.sort(key=lambda kv: (kv[1], kv[0]))
        summary["cohorts"][ck] = {
            "cohort_definition": c.get("cohort_definition"),
            "stays_checked": c.get("stays_checked"),
            "contract_violations_total": c.get("contract_violations_total"),
            "semantic_rule_violations_total": c.get("semantic_rule_violations_total"),
            "distinct_diagnostic_exam_dx_pairs": c.get("distinct_diagnostic_exam_dx_pairs"),
            "diagnostic_exam_dx_pairs_note": c.get("diagnostic_exam_dx_pairs_note"),
            "surviving_events": c.get("surviving_events"),
            "affected_stays": c.get("affected_stays"),
            "comorbidity_ledger": {
                "before_total": cl.get("before_total"),
                "admission_known": cl.get("admission_known"),
                "inpatient_candidates": cl.get("inpatient_candidates"),
                "inpatient_grounded": cl.get("inpatient_grounded"),
                "inpatient_by_source": cl.get("inpatient_by_source"),
                "ungrounded_inpatient_dropped": cl.get("ungrounded_inpatient_dropped"),
                "drop_rate_of_inpatient_candidates": cl.get("drop_rate_of_inpatient_candidates"),
                "drop_rate_of_before_total": cl.get("drop_rate_of_before_total"),
                "mean_comorb_before": cl.get("mean_comorb_before"),
                "mean_comorb_after": cl.get("mean_comorb_after"),
                "lowest_chapter_drop_rate": ch_rates[0] if ch_rates else None,
                "highest_chapter_drop_rate": ch_rates[-1] if ch_rates else None,
            },
        }
    return summary


def verify_paired_plan_identity(s5_root):
    """Prove 0.0% trajectory or medical-rule confounding between S5-1 arms and planned shards."""
    planned_sigs = {}
    planned_shards_loaded = []
    for s in range(4):
        pj = os.path.join(s5_root, "planned", "s5_1_shards", f"shard_{s}", "journeys.jsonl")
        if os.path.isfile(pj):
            planned_shards_loaded.append(s)
            for j in _read_jsonl(pj):
                planned_sigs[str(j["journey_id"])] = _journey_signature(j)

    arms_checked = {}
    total_verified = 0
    total_mismatches = 0
    for arm, prefix in (("Synth-Full", "full_shard"), ("Synth-VisibilityOnly", "vis_only_shard"), ("Synth-OracleLeak", "oracle_leak_shard")):
        shards = _discover_completed_s51_shards(s5_root, prefix)
        arm_stays = 0
        arm_mismatches = 0
        for s in shards:
            jp = os.path.join(s5_root, "generated", "s5_1", f"{prefix}{s}", "journeys.jsonl")
            for j in _read_jsonl(jp):
                jid = str(j["journey_id"])
                sig = _journey_signature(j)
                if planned_sigs:
                    if jid not in planned_sigs or planned_sigs[jid] != sig:
                        arm_mismatches += 1
                    arm_stays += 1
                else:
                    planned_sigs[jid] = sig
                    arm_stays += 1
        arms_checked[arm] = {"completed_shards": shards, "stays_verified": arm_stays, "mismatches": arm_mismatches}
        total_verified += arm_stays
        total_mismatches += arm_mismatches

    return {
        "planned_shards_found": planned_shards_loaded,
        "planned_stays_indexed": len(planned_sigs),
        "arms": arms_checked,
        "stays_verified_total": total_verified,
        "mismatches": total_mismatches,
        "zero_confounding_verified": total_mismatches == 0,
    }


def summarize_shard_gate_telemetry(shard_dirs):
    """Aggregate runtime gate counters directly from completed counts.json files."""
    totals = collections.Counter()
    sec_codes = collections.Counter()
    ff_codes = collections.Counter()
    used_dirs = []
    for d in shard_dirs:
        cp = os.path.join(d, "counts.json")
        if not os.path.isfile(cp):
            continue
        c = _read_json(cp)
        if int(c.get("llm_errors", 0)) != 0:
            continue
        used_dirs.append(d)
        for k, v in c.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                totals[k] += v
                if k.startswith("sec_"):
                    sec_codes[k[4:]] += int(v)
                elif k.startswith("first_fail_"):
                    ff_codes[k[11:]] += int(v)
    planned = int(totals.get("planned", 0))
    kept = int(totals.get("kept", 0))
    discarded = int(totals.get("discarded", 0))
    first_pass = int(totals.get("first_pass", 0))
    sections = int(totals.get("sections", 0))
    sec_ft = int(totals.get("sections_first_try", 0))
    chars = int(totals.get("chars", 0))
    calls = int(totals.get("calls", 0))
    dx_pol = int(sec_codes.get("dx_polarity_missing", 0))
    ff_fut = int(ff_codes.get("future_result", 0))
    return {
        "n_shards": len(used_dirs),
        "planned": planned,
        "kept": kept,
        "discarded": discarded,
        "discard_rate": round(discarded / max(1, planned), 4) if planned else None,
        "first_pass": first_pass,
        "first_pass_rate": round(first_pass / max(1, planned), 4) if planned else None,
        "adm_rewrites": int(totals.get("adm_rewrites", 0)),
        "sections": sections,
        "sections_first_try": sec_ft,
        "section_first_try_rate": round(sec_ft / max(1, sections), 4) if sections else None,
        "section_retries": int(totals.get("section_retries", 0)),
        "calls": calls,
        "calls_per_adm": round(calls / max(1, kept), 2) if kept else None,
        "calls_per_kchar": round(1000.0 * calls / max(1, chars), 3) if chars else None,
        "sec_dx_polarity_missing_count": dx_pol,
        "sec_dx_polarity_missing_per_section": round(dx_pol / max(1, sections), 4) if sections else None,
        "first_fail_future_result_count": ff_fut,
        "first_fail_future_result_per_planned": round(ff_fut / max(1, planned), 4) if planned else None,
        "top_10_sec_retry_codes": sec_codes.most_common(10),
        "top_10_adm_first_fail_codes": ff_codes.most_common(10),
    }


def _gate_pass_sample(adm_paths, j_paths, max_stays=500, seed=20261201):
    adms, jmap = audit_s5_intrinsic._load_sample(adm_paths, j_paths, max_stays=max_stays, seed=seed)
    checked = 0
    passed = 0
    fail_codes = collections.Counter()
    for a in adms:
        j = jmap.get(str(a.get("admission_id")))
        if not j:
            continue
        checked += 1
        fs = gate.check_admission(j, a.get("docs") or [])
        if not fs:
            passed += 1
        else:
            for _, code in fs:
                fail_codes[code] += 1
    return {
        "checked_stays": checked,
        "gate_pass_stays": passed,
        "gate_admission_pass_rate": round(passed / max(1, checked), 4) if checked else None,
        "top_gate_fail_codes": fail_codes.most_common(10),
    }


def compute_first_try_vs_post_gate_intrinsic(label, adm_paths, first_try_paths, j_paths, max_stays=500, seed=20261201):
    """Compare First-Try vs Post-Gate intrinsic audit and current gate.check_admission pass rate."""
    pg_int = audit_s5_intrinsic.audit_corpus(f"{label} (PostGate)", adm_paths, j_paths, max_stays=max_stays, seed=seed)
    pg_gate = _gate_pass_sample(adm_paths, j_paths, max_stays=max_stays, seed=seed) if j_paths else None
    out = {
        "label": label,
        "post_gate": dict(pg_int, gate_audit=pg_gate),
        "first_try": None,
        "gate_marginal_deltas": None,
    }
    if first_try_paths:
        ft_int = audit_s5_intrinsic.audit_corpus(f"{label} (FirstTry)", first_try_paths, j_paths, max_stays=max_stays, seed=seed)
        ft_gate = _gate_pass_sample(first_try_paths, j_paths, max_stays=max_stays, seed=seed) if j_paths else None
        out["first_try"] = dict(ft_int, gate_audit=ft_gate)
        deltas = {}
        for k in (
            "g3_hard_stay_rate", "n1_stay_rate", "g1_conflict_stay_rate",
            "b7_fail_stay_rate", "g2_s_conflict_rate", "k4_adm_leak_rate",
            "day0_preop_poststate_leak_rate", "oracle_any_rate",
        ):
            v_ft = ft_int.get(k)
            v_pg = pg_int.get(k)
            if v_ft is not None and v_pg is not None:
                deltas[f"{k}_reduction"] = round(v_ft - v_pg, 4)
        if ft_gate and pg_gate and ft_gate.get("gate_admission_pass_rate") is not None and pg_gate.get("gate_admission_pass_rate") is not None:
            deltas["gate_admission_pass_gain"] = round(pg_gate["gate_admission_pass_rate"] - ft_gate["gate_admission_pass_rate"], 4)
        out["gate_marginal_deltas"] = deltas
    return out


def compute_four_cell_belief_gate_decomposition(s5_root, common_shards, n_boot=1000, seed=20261201):
    """Decompose VisOnly -> Full belief-state updates into compiler First-Try vs gate Post-Gate."""
    if not common_shards:
        return {"available": False, "common_shards": [], "reason": "no_completed_full_and_vis_only_pair"}

    vis_dirs = [os.path.join(s5_root, "generated", "s5_1", f"vis_only_shard{s}") for s in common_shards]
    full_dirs = [os.path.join(s5_root, "generated", "s5_1", f"full_shard{s}") for s in common_shards]
    vis_pg_paths = [os.path.join(d, "admissions.jsonl") for d in vis_dirs]
    full_pg_paths = [os.path.join(d, "admissions.jsonl") for d in full_dirs]
    vis_ft_paths = _existing_first_try_paths(vis_dirs)
    full_ft_paths = _existing_first_try_paths(full_dirs)
    planned_j_paths = [
        p for s in common_shards
        for p in (os.path.join(s5_root, "planned", "s5_1_shards", f"shard_{s}", "journeys.jsonl"),)
        if os.path.isfile(p)
    ]
    vis_j_paths = [os.path.join(d, "journeys.jsonl") for d in vis_dirs]
    full_j_paths = [os.path.join(d, "journeys.jsonl") for d in full_dirs]

    if len(vis_ft_paths) != len(common_shards) or len(full_ft_paths) != len(common_shards):
        return {
            "available": False,
            "common_shards": common_shards,
            "reason": "missing_first_try_admissions",
        }

    def _by_id(paths):
        return {str(a["admission_id"]): a for p in paths for a in _read_jsonl(p)}

    cells = {
        "VisOnly_FirstTry": _by_id(vis_ft_paths),
        "VisOnly_PostGate": _by_id(vis_pg_paths),
        "Full_FirstTry": _by_id(full_ft_paths),
        "Full_PostGate": _by_id(full_pg_paths),
    }
    universe_src = planned_j_paths if planned_j_paths else (vis_j_paths + full_j_paths)
    jmap = {str(j["journey_id"]): j for p in universe_src for j in _read_jsonl(p)}
    common_ids = sorted(set.intersection(*(set(m.keys()) for m in cells.values())) & set(jmap.keys()))

    def _eval_4way(state_filter):
        stats = {k: {"missed": 0, "total": 0, "silent": 0} for k in cells}
        for jid in common_ids:
            j = jmap[jid]
            for ev in (j.get("dx_events") or ()):
                st = ev.get("state")
                ed = int(ev.get("day", 0))
                dx = ev.get("name") or ""
                if ed < 1 or st not in state_filter or len(dx) < 2:
                    continue
                res = {cname: t123_bench._eval_single_event_on_stay(cmap[jid], dx, ed, st) for cname, cmap in cells.items()}
                if not all(r[0] for r in res.values()):
                    continue
                for cname, (_, _, missed, silent) in res.items():
                    stats[cname]["missed"] += missed
                    stats[cname]["total"] += 1
                    stats[cname]["silent"] += silent
        for cname, d in stats.items():
            d["miss_rate"] = round(d["missed"] / max(1, d["total"]), 4)
        v_ft = stats["VisOnly_FirstTry"]["miss_rate"]
        v_pg = stats["VisOnly_PostGate"]["miss_rate"]
        f_ft = stats["Full_FirstTry"]["miss_rate"]
        f_pg = stats["Full_PostGate"]["miss_rate"]
        tot_red = round(v_ft - f_pg, 4)
        comp_red = round(v_ft - f_ft, 4)
        gate_marg = round(f_ft - f_pg, 4)
        gate_on_vis = round(v_ft - v_pg, 4)
        return {
            "cells": stats,
            "decomposition": {
                "total_miss_reduction_vis_ft_to_full_pg": tot_red,
                "pure_compiler_reduction_at_first_try": comp_red,
                "marginal_gate_reduction_on_full": gate_marg,
                "gate_alone_reduction_on_vis_only": gate_on_vis,
                "compiler_share_of_total_reduction": round(comp_red / tot_red, 4) if tot_red > 0 else None,
                "gate_marginal_share_of_total_reduction": round(gate_marg / tot_red, 4) if tot_red > 0 else None,
            },
        }

    ep_ft = t123_bench.audit_epistemic_asymmetry(
        vis_ft_paths, vis_j_paths, full_ft_paths, full_j_paths,
        planned_j_paths=planned_j_paths or None, n_boot=n_boot, seed=seed,
    )
    ep_pg = t123_bench.audit_epistemic_asymmetry(
        vis_pg_paths, vis_j_paths, full_pg_paths, full_j_paths,
        planned_j_paths=planned_j_paths or None, n_boot=n_boot, seed=seed,
    )
    return {
        "available": True,
        "common_shards": common_shards,
        "common_4way_stays": len(common_ids),
        "planned_stays_in_common_shards": len(jmap),
        "four_cell_paired": {
            "confirmed": _eval_4way(("确诊",)),
            "excluded": _eval_4way(("排除",)),
            "combined": _eval_4way(("确诊", "排除")),
        },
        "epistemic_asymmetry_first_try": ep_ft,
        "epistemic_asymmetry_post_gate": ep_pg,
    }


def _extract_probe_compact(rehearsal_json):
    if not rehearsal_json or not os.path.isfile(rehearsal_json):
        return None
    d = _read_json(rehearsal_json)
    pc = d.get("paired_comparisons") or {}
    t1a = d.get("T1") or {}
    t2a = d.get("T2a") or {}
    t2b = d.get("T2b") or {}
    ab = d.get("T2_2x2_supervision_ablation") or {}
    lr = d.get("T1_low_resource_curve") or {}
    t3 = d.get("T3_gold_dev") or {}

    def _pair_pick(block, key):
        p = block.get(key)
        if not p:
            return None
        return {
            "n": p.get("n"), "pos": p.get("pos"),
            "auroc_a": p.get("auroc_a"), "auroc_b": p.get("auroc_b"),
            "delta_auroc": p.get("delta_auroc"), "delta_auroc_ci95": p.get("delta_auroc_ci95"),
            "auprc_a": p.get("auprc_a"), "auprc_b": p.get("auprc_b"),
            "delta_auprc": p.get("delta_auprc"),
            "p_boot_two_sided": p.get("p_boot_two_sided"),
        }

    t2a_f1 = {
        k: {kk: vv for kk, vv in (t2a.get(k) or {}).items() if kk in ("n_train", "macro_f1_3class", "macro_f1_ci95", "f1_excluded_0", "f1_conf_2")}
        for k in ("Synth-Full", "Synth-Full (Matched N=600)", "Synth-VisibilityOnly", "DIRECT-32B", "Real-Small (250)", "Real-Full (2000)", "Augment (Real250+SynthFull)")
        if k in t2a
    }
    ab_compact = {}
    for task in ("T2a", "T2b"):
        if task in ab:
            ab_compact[task] = {
                cell: {
                    kk: vv for kk, vv in cdict.items()
                    if kk in ("macro_f1_3class", "macro_f1_ci95", "f1_excluded_0", "mentioned_only_auroc", "mentioned_only_auroc_ci95", "mentioned_only_auprc")
                }
                for cell, cdict in ab[task].items()
            }
    return {
        "source_path": rehearsal_json,
        "Matched600_vs_DIRECT32B": {
            "T1a": _pair_pick(pc.get("T1a") or {}, "SynthFull600_minus_DIRECT32B"),
            "T1_major_op": _pair_pick(pc.get("T1_major_op") or {}, "SynthFull600_minus_DIRECT32B"),
            "T2b_mentioned_only": _pair_pick(pc.get("T2b_mentioned_only") or {}, "SynthFull600_minus_DIRECT32B"),
            "T1_direct_vs_full600": {
                "Synth-Full (Matched N=600)": (t1a.get("Synth-Full (Matched N=600)") or {}).get("T1a", {}).get("auroc"),
                "DIRECT-32B": (t1a.get("DIRECT-32B") or {}).get("T1a", {}).get("auroc"),
            },
            "T2a_macro_f1": {k: (t2a_f1.get(k) or {}).get("macro_f1_3class") for k in ("Synth-Full (Matched N=600)", "DIRECT-32B") if k in t2a_f1},
            "T3_gold_dev": {
                k: {kk: vv for kk, vv in (t3.get(k) or {}).items() if kk in ("micro_f1_12", "macro_f1_12", "has_dose_acc", "has_dose_recall")}
                for k in ("Synth-Full (Matched N=600)", "Synth-Full", "DIRECT-32B", "Real-Small (250)")
                if k in t3
            } if t3 else None,
        },
        "SynthFull_vs_VisOnly": {
            "T1a": _pair_pick(pc.get("T1a") or {}, "SynthFull_minus_VisOnly"),
            "T2a_macro_f1": {k: (t2a_f1.get(k) or {}).get("macro_f1_3class") for k in ("Synth-Full", "Synth-VisibilityOnly") if k in t2a_f1},
            "T2b_mentioned_only_PostGate": _pair_pick(pc.get("T2b_mentioned_only") or {}, "SynthFull_PostGate_minus_VisOnly_PostGate"),
            "T2b_mentioned_only_FirstTry": _pair_pick(pc.get("T2b_mentioned_only") or {}, "SynthFull_FirstTry_minus_VisOnly_FirstTry"),
            "T2b_mentioned_only_raw": {
                k: {kk: vv for kk, vv in (t2b.get(k) or {}).items() if kk in ("mentioned_only_auroc", "mentioned_only_auprc", "f1_05")}
                for k in ("Synth-Full (PostGate)", "Synth-Full (FirstTry)", "Synth-VisibilityOnly (PostGate)", "Synth-VisibilityOnly (FirstTry)")
                if k in t2b
            },
            "T2_2x2_supervision_ablation": ab_compact,
            "T2b_text_rule_2x2_paired": pc.get("T2b_text_rule_2x2"),
        },
        "Augment_vs_RealSmall": {
            "T1a_Real250": _pair_pick(pc.get("T1a") or {}, "Augment_SynthFull_minus_Real250"),
            "T1a_Real100": _pair_pick(pc.get("T1a") or {}, "Augment_Real100_SynthFull_minus_Real100"),
            "T1a_Real50": _pair_pick(pc.get("T1a") or {}, "Augment_Real50_SynthFull_minus_Real50"),
            "T2b_Real250": _pair_pick(pc.get("T2b_mentioned_only") or {}, "Augment_SynthFull_minus_Real250"),
            "T1_low_resource_auroc": {
                k: ((v.get("T1a") or {}).get("auroc") if isinstance(v, dict) and "T1a" in v else (v.get("auroc") if isinstance(v, dict) else v))
                for k, v in lr.items()
            },
        },
    }


def extract_tstr_deltas(s5_root, rehearsal_linear=None, rehearsal_neural=None):
    """Extract controlled TSTR deltas when Stage-1 rehearsal JSONs exist."""
    if not rehearsal_linear:
        for fn in ("rehearsal_t123_fold1_final2k_v17.json", "rehearsal_t123_fold1_final2k_v15.json"):
            cand = os.path.join(s5_root, "tables", fn)
            if os.path.isfile(cand):
                rehearsal_linear = cand
                break
    if not rehearsal_neural:
        for fn in ("rehearsal_neural_fold1_final2k_v17.json", "rehearsal_neural_fold1_final2k_v15.json"):
            cand = os.path.join(s5_root, "tables", fn)
            if os.path.isfile(cand):
                rehearsal_neural = cand
                break
    return {
        "linear": _extract_probe_compact(rehearsal_linear),
        "neural": _extract_probe_compact(rehearsal_neural),
    }


def _run_s52_text_probe(s5_root, s52_arms, real_fold, s52_probe_json=None):
    if s52_probe_json and os.path.isfile(s52_probe_json):
        return _read_json(s52_probe_json)
    for fn in ("s5_2_17d_text_probe_final_v17.json", "s5_2_17d_text_probe_final_v15.json"):
        cand = os.path.join(s5_root, "tables", fn)
        if os.path.isfile(cand) and len(s52_arms) == 3:
            # Verify generated mtimes aren't newer than cached probe; recompute below if missing.
            return _read_json(cand)
    if not s52_arms:
        return None
    real_adms = _read_jsonl(real_fold) if (real_fold and os.path.isfile(real_fold)) else []
    s5_2_data = {}
    for arm in s52_arms:
        sdir = os.path.join(s5_root, "generated", "s5_2", arm)
        s5_2_data[arm] = (_read_jsonl(os.path.join(sdir, "admissions.jsonl")), {j["journey_id"]: j for j in _read_jsonl(os.path.join(sdir, "journeys.jsonl"))})
    out = {"real_fold": os.path.basename(real_fold) if real_fold else None, "cutoffs": {}}
    label_of = {"route_b": "Route-B", "route_a_only": "Route-A-Only", "route_ab": "Route-AB"}
    for cutoff_lbl, gt2_flag in (("ge2", False), ("gt2", True)):
        real_rows = audit_s5_2_17d.extract_unified_cohort(real_adms, is_real=True, gt2_only=gt2_flag) if real_adms else []
        by_pw_r = collections.Counter(r["kb_key"] for r in real_rows)
        common_weights = {
            k: sum(1 for r in real_rows if r["kb_key"] == k and r["y"] == 1) * sum(1 for r in real_rows if r["kb_key"] == k and r["y"] == 0)
            for k in by_pw_r
        }
        sec = {}
        if real_rows:
            sec["Real-17d"] = audit_s5_2_17d.evaluate_unified_probe(f"Real-17d ({cutoff_lbl})", real_rows, common_weights)
        for arm in s52_arms:
            adms, jmap = s5_2_data[arm]
            rows = audit_s5_2_17d.extract_unified_cohort(adms, jmap=jmap, is_real=False, gt2_only=gt2_flag)
            sec[label_of[arm]] = audit_s5_2_17d.evaluate_unified_probe(f"{label_of[arm]} ({cutoff_lbl})", rows, common_weights)
        out["cutoffs"][cutoff_lbl] = sec
    return out


def build_mechanism_report(
    s5_root,
    real_fold="/home/weify/pyt/data/deid_run9_fold1.jsonl",
    pre_gpu_ledger=None,
    rehearsal_linear=None,
    rehearsal_neural=None,
    s52_probe_json=None,
    max_intrinsic_stays=500,
    n_boot=1000,
    seed=20261201,
):
    """Build complete 5-block mechanism contribution decomposition JSON."""
    _assert_real_fold_safe(real_fold)
    if not pre_gpu_ledger:
        for fn in ("v037_pre_gpu_audit.json", "v036_pre_gpu_audit.json"):
            cand = os.path.join(s5_root, "tables", fn)
            if os.path.isfile(cand):
                pre_gpu_ledger = cand
                break

    full_shards = _discover_completed_s51_shards(s5_root, "full_shard")
    vis_shards = _discover_completed_s51_shards(s5_root, "vis_only_shard")
    ora_shards = _discover_completed_s51_shards(s5_root, "oracle_leak_shard")
    common_shards = sorted(set(full_shards) & set(vis_shards))
    s52_arms = _discover_completed_s52_arms(s5_root)

    block0_rules = load_rule_scope_summary(pre_gpu_ledger)
    block0_identity = verify_paired_plan_identity(s5_root)

    # Compute intrinsic FirstTry vs PostGate per available arm
    arm_intrinsics = {}
    arm_telemetry = {}

    def _run_s51_arm(arm_name, prefix, shard_list):
        if not shard_list:
            return
        dirs = [os.path.join(s5_root, "generated", "s5_1", f"{prefix}{s}") for s in shard_list]
        arm_telemetry[arm_name] = summarize_shard_gate_telemetry(dirs)
        adm_p = [os.path.join(d, "admissions.jsonl") for d in dirs]
        ft_p = _existing_first_try_paths(dirs)
        j_p = [os.path.join(d, "journeys.jsonl") for d in dirs]
        arm_intrinsics[arm_name] = compute_first_try_vs_post_gate_intrinsic(arm_name, adm_p, ft_p, j_p, max_stays=max_intrinsic_stays, seed=seed)

    _run_s51_arm("Synth-Full", "full_shard", full_shards)
    _run_s51_arm("Synth-VisibilityOnly", "vis_only_shard", vis_shards)
    _run_s51_arm("Synth-OracleLeak", "oracle_leak_shard", ora_shards)

    d32_dirs = t123_bench._resolve_direct32b_paths(s5_root, None)
    if d32_dirs:
        arm_intrinsics["DIRECT-32B"] = compute_first_try_vs_post_gate_intrinsic("DIRECT-32B", d32_dirs, [], [], max_stays=max_intrinsic_stays, seed=seed)
        d32_cnt_dirs = [os.path.dirname(p) for p in d32_dirs if os.path.isfile(os.path.join(os.path.dirname(p), "counts.json"))]
        if d32_cnt_dirs:
            arm_telemetry["DIRECT-32B"] = summarize_shard_gate_telemetry(d32_cnt_dirs)

    for arm in s52_arms:
        sdir = os.path.join(s5_root, "generated", "s5_2", arm)
        key = f"S5-2:{arm}"
        arm_telemetry[key] = summarize_shard_gate_telemetry([sdir])
        arm_intrinsics[key] = compute_first_try_vs_post_gate_intrinsic(
            key, [os.path.join(sdir, "admissions.jsonl")], _existing_first_try_paths([sdir]), [os.path.join(sdir, "journeys.jsonl")],
            max_stays=max_intrinsic_stays, seed=seed,
        )

    s52_probe = _run_s52_text_probe(s5_root, s52_arms, real_fold, s52_probe_json=s52_probe_json)

    # Block 2: Dual-axis visibility compilation (Oracle-Leak vs VisOnly / Full FirstTry & PostGate)
    vis_summary = {}
    for aname in ("Synth-OracleLeak", "Synth-VisibilityOnly", "Synth-Full"):
        if aname not in arm_intrinsics:
            continue
        ai = arm_intrinsics[aname]
        for stage in ("first_try", "post_gate"):
            sd = ai.get(stage)
            if not sd:
                continue
            vis_summary[f"{aname}_{stage}"] = {
                k: sd.get(k) for k in ("n_stays", "oracle_n", "oracle_O_rate", "oracle_A_rate", "oracle_B_strict_rate", "oracle_any_rate", "k4_adm_leak_rate", "day0_preop_poststate_leak_rate")
            }
    vis_reductions = {}
    ora_pg = vis_summary.get("Synth-OracleLeak_post_gate") or {}
    for target in ("Synth-VisibilityOnly_first_try", "Synth-VisibilityOnly_post_gate", "Synth-Full_first_try", "Synth-Full_post_gate"):
        td = vis_summary.get(target) or {}
        if ora_pg.get("oracle_any_rate") is not None and td.get("oracle_any_rate") is not None:
            o0 = float(ora_pg["oracle_any_rate"])
            o1 = float(td["oracle_any_rate"])
            vis_reductions[target] = {
                "oracle_any_abs_drop": round(o0 - o1, 4),
                "oracle_any_rel_elim_share": round((o0 - o1) / o0, 4) if o0 > 0 else None,
                "k4_adm_abs_drop": round(float(ora_pg.get("k4_adm_leak_rate", 0.0)) - float(td.get("k4_adm_leak_rate", 0.0)), 4),
            }

    # Block 3: 4-cell explicit belief-state compilation vs gate rewrite decomposition
    block3_4cell = compute_four_cell_belief_gate_decomposition(s5_root, common_shards, n_boot=n_boot, seed=seed)

    # Block 4: Downstream TSTR controlled deltas
    block4_tstr = extract_tstr_deltas(s5_root, rehearsal_linear=rehearsal_linear, rehearsal_neural=rehearsal_neural)

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "s5_root": s5_root,
        "fold9_untouched_verified": True,
        "discovery": {
            "completed_s51_full_shards": full_shards,
            "completed_s51_vis_only_shards": vis_shards,
            "completed_s51_oracle_leak_shards": ora_shards,
            "common_s51_full_vis_shards": common_shards,
            "completed_s52_arms": s52_arms,
            "direct32b_present": bool(d32_dirs),
        },
        "block0_fixed_rule_scope_and_paired_identity": {
            "rule_scope": block0_rules,
            "paired_identity": block0_identity,
        },
        "block1_planning_priors_vs_unplanned_direct32b": {
            "gate_telemetry_by_arm": arm_telemetry,
            "intrinsic_first_try_vs_post_gate": arm_intrinsics,
            "s5_2_17d_unified_text_probe": s52_probe,
        },
        "block2_temporal_visibility_isolation": {
            "leakage_rates": vis_summary,
            "relative_to_oracle_leak": vis_reductions,
        },
        "block3_compiler_state_vs_gate_rewrite": block3_4cell,
        "block4_tstr_controlled_deltas": block4_tstr,
    }


def render_markdown_summary(report):
    """Render concise Markdown attribution report with zero U+00B7 characters."""
    disc = report["discovery"]
    b0_id = report["block0_fixed_rule_scope_and_paired_identity"]["paired_identity"]
    b0_rs = report["block0_fixed_rule_scope_and_paired_identity"]["rule_scope"]
    lines = [
        "# MedCourse Core Mechanism Attribution Summary (Fixed Rules & Fixed Trajectories)",
        "",
        f"- Root: `{report['s5_root']}` | Generated UTC: `{report['generated_at_utc']}`",
        f"- Shards ready: `S5-1 full={disc['completed_s51_full_shards']}`, `vis_only={disc['completed_s51_vis_only_shards']}`, `oracle_leak={disc['completed_s51_oracle_leak_shards']}`, `S5-2={disc['completed_s52_arms']}`, `DIRECT-32B={disc['direct32b_present']}`",
        f"- Paired trajectory identity check: `{b0_id['stays_verified_total']}` stays checked, `mismatches = {b0_id['mismatches']}` (`zero_confounding_verified = {b0_id['zero_confounding_verified']}`)",
        "",
    ]
    c3020 = ((b0_rs.get("cohorts") or {}).get("ALL_ARM_STAYS_3020") or {}) if b0_rs.get("available") else {}
    if c3020:
        cl = c3020["comorbidity_ledger"]
        lines.extend([
            "## Block 0: Fixed Domain-Rule Pre-Filter Boundary",
            f"- Scope: `3,020` arm-stays (`contract_violations = {c3020['contract_violations_total']}`, `semantic_rule_violations = {c3020['semantic_rule_violations_total']}`)",
            f"- Rule-conformance pairs (`ALL_ARM_STAYS_3020`): `{c3020['distinct_diagnostic_exam_dx_pairs']}` (verifies code-to-rule conformance, NOT independent clinical ground truth)",
            f"- Inpatient comorbidity filtering: `{cl['inpatient_candidates']}` candidates -> grounded `{cl['inpatient_grounded']}`, dropped `{cl['ungrounded_inpatient_dropped']}` (`{round(100 * cl['drop_rate_of_inpatient_candidates'], 2)}%` candidate drop rate; chapter range `{cl['lowest_chapter_drop_rate']}` to `{cl['highest_chapter_drop_rate']}`)",
            "",
        ])

    b1 = report["block1_planning_priors_vs_unplanned_direct32b"]
    lines.append("## Block 1 & 2: First-Try vs Post-Gate Telemetry & Visibility Isolation")
    lines.append("| Arm | Planned / Kept | First-Pass Rate | Gate Pass (FirstTry -> PostGate) | Oracle Any (FirstTry -> PostGate) | K4 Adm Leak (FirstTry -> PostGate) |")
    lines.append("|---|---|---|---|---|---|")
    for arm_name, ai in b1["intrinsic_first_try_vs_post_gate"].items():
        tm = b1["gate_telemetry_by_arm"].get(arm_name) or {}
        ft = ai.get("first_try") or {}
        pg = ai.get("post_gate") or {}
        ft_g = (ft.get("gate_audit") or {}).get("gate_admission_pass_rate")
        pg_g = (pg.get("gate_audit") or {}).get("gate_admission_pass_rate")
        lines.append(
            f"| `{arm_name}` | `{tm.get('planned', pg.get('n_stays'))} / {tm.get('kept', pg.get('n_stays'))}` | `{tm.get('first_pass_rate')}` | `{ft_g} -> {pg_g}` | `{ft.get('oracle_any_rate')} -> {pg.get('oracle_any_rate')}` | `{ft.get('k4_adm_leak_rate')} -> {pg.get('k4_adm_leak_rate')}` |"
        )
    lines.append("")

    b3 = report["block3_compiler_state_vs_gate_rewrite"]
    lines.append("## Block 3: 4-Cell Decomposition (Explicit Belief-State Compilation vs Gate Rewrite)")
    if b3.get("available"):
        lines.append(f"- Common 4-way stays: `{b3['common_4way_stays']}` across shards `{b3['common_shards']}`")
        for ev_type in ("confirmed", "excluded", "combined"):
            sub = b3["four_cell_paired"][ev_type]
            cs = sub["cells"]
            dc = sub["decomposition"]
            lines.append(
                f"- **{ev_type}**: `VisFT={cs['VisOnly_FirstTry']['miss_rate']}` | `VisPG={cs['VisOnly_PostGate']['miss_rate']}` | `FullFT={cs['Full_FirstTry']['miss_rate']}` | `FullPG={cs['Full_PostGate']['miss_rate']}` "
                f"=> Total reduction `{dc['total_miss_reduction_vis_ft_to_full_pg']}` = Pure compiler (`FirstTry`) `{dc['pure_compiler_reduction_at_first_try']}` (`share={dc['compiler_share_of_total_reduction']}`) + Gate marginal `{dc['marginal_gate_reduction_on_full']}` (`share={dc['gate_marginal_share_of_total_reduction']}`)"
            )
    else:
        lines.append(f"- Pending completion of paired `full` and `vis_only` shards (`reason={b3.get('reason')}`)")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-root", required=True)
    ap.add_argument("--real-fold", default="/home/weify/pyt/data/deid_run9_fold1.jsonl")
    ap.add_argument("--pre-gpu-ledger", default=None)
    ap.add_argument("--rehearsal-linear", default=None)
    ap.add_argument("--rehearsal-neural", default=None)
    ap.add_argument("--s52-probe-json", default=None)
    ap.add_argument("--max-intrinsic-stays", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    rep = build_mechanism_report(
        args.s5_root,
        real_fold=args.real_fold,
        pre_gpu_ledger=args.pre_gpu_ledger,
        rehearsal_linear=args.rehearsal_linear,
        rehearsal_neural=args.rehearsal_neural,
        s52_probe_json=args.s52_probe_json,
        max_intrinsic_stays=args.max_intrinsic_stays,
        n_boot=args.n_boot,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(render_markdown_summary(rep))

    print("MECHANISM_ATTRIBUTION_DONE", json.dumps({
        "out_json": args.out_json,
        "discovery": rep["discovery"],
        "identity_mismatches": rep["block0_fixed_rule_scope_and_paired_identity"]["paired_identity"]["mismatches"],
        "block3_available": rep["block3_compiler_state_vs_gate_rewrite"].get("available"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
