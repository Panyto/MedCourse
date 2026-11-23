# -*- coding: utf-8 -*-
"""Committed, single-source reproducible audit ledger builder for medcourse_core v0.3.7.

Produces two JSON ledgers from the exact planned/generated artifacts on disk:
1) v037_pre_gpu_audit.json:
   - Contract & closed capability / semantic validation across S5-1 (2,000),
     S5-2 (1,020 arm-stays = 340 unique slots x 3 arms), ALL_ARM_STAYS (3,020),
     and UNIQUE_SLOT_STAYS (2,340: S5-1 2,000 + S5-2 route_ab 340, plus per-arm totals).
   - Full comorbidity reduction breakdown by ICD-10 chapter and primary diagnosis.
2) v037_item_num_parser_audit.json:
   - Audits generated admissions directly via gate.audit_item_numbers(pl, sec_text),
     emitting mutually exclusive counts (parsed_value_tokens, qualitative_or_coord_only_mentions,
     unparsed_mentions) and extracting unparsed_30_samples strictly from gate.audit_item_numbers.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import re

from s5_common import setup_paths

setup_paths()

from medcourse import compiler, contract, gate, kb, planner  # noqa: E402
from medcourse.rules import setup  # noqa: E402

setup(("BENCH_G3", "BENCH_G2"))
from ruler_g1 import split_sections  # noqa: E402


_ENDOSCOPY_KEYS = frozenset({"胃镜", "肠镜", "胶囊内镜", "ERCP", "超声内镜", "支气管镜", "喉镜", "膀胱镜"})


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_comorb_agg():
    return {
        "stays": 0,
        "before_total": 0,
        "admission_known": 0,
        "inpatient_candidates": 0,
        "inpatient_grounded": 0,
        "inpatient_by_source": {"procedure": 0, "lab": 0, "incidental": 0},
        "ungrounded_inpatient_dropped": 0,
        "after_total": 0,
    }


def _merge_comorb_agg(dst, stays, lg):
    dst["stays"] += stays
    dst["before_total"] += int(lg.get("before_total", 0))
    dst["admission_known"] += int(lg.get("admission_known", 0))
    dst["inpatient_candidates"] += int(lg.get("inpatient_candidates", 0))
    dst["inpatient_grounded"] += int(lg.get("inpatient_grounded", 0))
    dst["ungrounded_inpatient_dropped"] += int(lg.get("ungrounded_inpatient_dropped", 0))
    dst["after_total"] += int(lg.get("after_total", 0))
    bys = lg.get("inpatient_by_source") or {}
    for sk in ("procedure", "lab", "incidental"):
        dst["inpatient_by_source"][sk] += int(bys.get(sk, 0))


def _finalize_comorb_agg(d):
    out = dict(d)
    st = max(1, d["stays"])
    out["drop_rate_of_inpatient_candidates"] = (
        round(d["ungrounded_inpatient_dropped"] / d["inpatient_candidates"], 4)
        if d["inpatient_candidates"] else 0.0
    )
    out["drop_rate_of_before_total"] = (
        round(d["ungrounded_inpatient_dropped"] / d["before_total"], 4)
        if d["before_total"] else 0.0
    )
    out["mean_comorb_before"] = round(d["before_total"] / st, 4)
    out["mean_comorb_after"] = round(d["after_total"] / st, 4)
    return out


def _empty_stay_ledger(j):
    coms = (j.get("patient") or {}).get("comorbidities") or []
    adm_n = sum(1 for c in coms if c.get("at_admission"))
    inp_n = len(coms) - adm_n
    return {
        "before_total": len(coms),
        "admission_known": adm_n,
        "inpatient_candidates": inp_n,
        "inpatient_grounded": inp_n,
        "inpatient_by_source": {"procedure": 0, "lab": 0, "incidental": 0},
        "ungrounded_inpatient_dropped": 0,
        "dropped_names": [],
        "after_total": len(coms),
    }


def _proc_supports_comorb(p, proc, cname):
    """Verify procedure actually grounds an inpatient status/post-op comorbidity."""
    if not proc or proc.get("do_day", -1) < 0 or not cname:
        return False
    aname = proc.get("name") or ""
    sex = p.get("sex") or ""
    if planner._has_any(cname, ("产后", "产褥")):
        return sex == "女" and planner._has_any(aname, planner._DELIVERY_PROC_WORDS)
    if planner._has_any(cname, ("流产后", "清宫术后", "刮宫术后")):
        return sex == "女" and planner._has_any(aname, planner._ABORTION_PROC_WORDS)
    if "术后" in cname:
        stem = cname.split("术后")[0]
        c_orgs = planner.dx_organs(cname, "")
        a_orgs = planner.dx_organs(aname, "")
        return (len(stem) >= 2 and stem[:2] in aname) or planner.same_organ(aname, cname) or bool(c_orgs & a_orgs)
    return False


def _exam_supports_excl(ex, diff_name, pdx, code3, primary_organs):
    """Independently verify negative exam against _STRICT_NEGATIVE_EXCL_RULES and organ site."""
    if not ex or not diff_name:
        return False
    if (
        ex.get("result_day", -1) < 1
        or ex.get("recheck")
        or ex.get("items")
        or ex.get("diagnostic")
        or ex.get("abnormal")
        or ex.get("incidental_findings")
    ):
        return False
    d_orgs = set(planner.dx_organs(diff_name, code3 if diff_name == pdx else "") or primary_organs)
    if not d_orgs or not planner.site_ok(ex["key"], ex.get("region"), d_orgs):
        return False
    e_key = "心脏超声" if planner._is_cardiac_us(ex["key"], ex.get("region")) else ex["key"]
    return any(
        e_key in m_keys
        and bool(set(m_orgs) & d_orgs)
        and planner.site_ok(ex["key"], ex.get("region"), set(m_orgs))
        and planner._has_any(diff_name, d_kws)
        for m_keys, m_orgs, d_kws in planner._STRICT_NEGATIVE_EXCL_RULES
    )


def _check_stay_semantics(j):
    """Verify planned journey against closed v0.3.7 capability rules without trusting pre-stamped flags."""
    sem_viols = []
    p = j["patient"]
    pdx = p["primary_dx"]["name"]
    code3 = p["primary_dx"].get("code3") or ""
    tkind = planner.treatment_kind(pdx, code3)
    under = (p["primary_dx"].get("underlying") or planner.underlying_dx(pdx, [c["name"] for c in p.get("comorbidities") or ()])) if tkind else None
    calc_organs = set(planner.dx_organs(under or pdx, code3))
    organs = calc_organs or set(p["primary_dx"].get("organs") or ())
    fname = (f"{under}既往治疗改变" if under else None) if tkind else planner.finding_label(pdx, code3)
    cap_dx = (f"{under}既往治疗改变" if under else None) if tkind else fname
    procs = j.get("procedures") or []
    ex_by_id = {e["id"]: e for e in j.get("exams") or ()}
    pr_by_id = {a["id"]: a for a in procs}
    excl_eids = {
        eid
        for ev in (j.get("dx_events") or ())
        if ev.get("state") == "排除"
        for eid in (ev.get("evidence_ids") or ())
    }

    diag_pairs = []
    for e in j.get("exams") or ():
        for inc_nm in (e.get("incidental_findings") or ()):
            if e.get("items") or not e.get("abnormal") or e["id"] in excl_eids or not planner._incidental_capable(e, inc_nm):
                sem_viols.append(f"invalid_incidental_finding:{e['key']}:{e.get('region') or ''}:{inc_nm}")
        if not (e.get("diagnostic") or e.get("can_confirm")):
            continue
        diag_pairs.append((e["key"], e.get("region") or "", e["kind"], bool(e.get("can_confirm")), e.get("finding") or "", pdx))
        if e["kind"] == "检验":
            sem_viols.append(f"lab_marked_diagnostic:{e['key']}")
            continue
        if e["key"] in ("病理", "活检"):
            path_dx = under if tkind else fname
            if not (path_dx and planner._pathology_can_confirm(e, path_dx, organs, procs)):
                sem_viols.append(f"invalid_pathology_pos:{e['key']}:{path_dx}")
            continue
        if not cap_dx:
            sem_viols.append(f"missing_cap_dx:{e['key']}")
            continue
        exp_cap = planner._exam_pos_capability(e["key"], e["kind"], e.get("region"), cap_dx, organs)
        if exp_cap is None:
            if e["key"] in _ENDOSCOPY_KEYS:
                sem_viols.append(f"endoscopy_subsite_mismatch:{e['key']}:{cap_dx}")
            elif planner._has_any(cap_dx, ("腰椎", "颈椎", "胸椎", "脊柱")) and e.get("region") == "腹":
                sem_viols.append(f"spine_abdomen_mismatch:{e['key']}:{cap_dx}")
            elif "痔" in cap_dx and e.get("region") == "血管":
                sem_viols.append(f"vascular_us_hemorrhoid:{e['key']}:{cap_dx}")
            else:
                sem_viols.append(f"disallowed_exam_dx_pair:{e['key']}:{e.get('region') or ''}:{cap_dx}")
        elif e.get("can_confirm") and exp_cap != "confirm_finding":
            sem_viols.append(f"unconfirmed_cap_elevated:{e['key']}:{cap_dx}:{exp_cap}")

    for c in p.get("comorbidities") or ():
        if c.get("at_admission"):
            continue
        cname = c.get("name") or ""
        c_eids = c.get("evidence_ids") or []
        if not c_eids:
            sem_viols.append(f"inpatient_comorb_missing_evidence:{cname}")
            continue
        ok_com = False
        for eid in c_eids:
            if eid in pr_by_id:
                ok_com = _proc_supports_comorb(p, pr_by_id[eid], cname)
            elif eid in ex_by_id:
                ex = ex_by_id[eid]
                if ex.get("result_day", -1) >= 1:
                    if ex.get("items"):
                        ok_com = planner._lab_supports_dx(ex, cname)
                    else:
                        ok_com = (
                            cname in (ex.get("incidental_findings") or ())
                            and eid not in excl_eids
                            and planner._incidental_capable(ex, cname)
                        )
            if ok_com:
                break
        if not ok_com:
            sem_viols.append(f"inpatient_comorb_unsupported_evidence:{cname}")

    is_status = planner._has_any(pdx, planner._ADMISSION_STATUS_DX_WORDS)
    for ev in j.get("dx_events") or ():
        nm, st = ev.get("name"), ev.get("state")
        if nm == pdx and st == "疑似" and (is_status or tkind):
            sem_viols.append(f"status_or_tx_dx_suspected:{nm}")
        if ev.get("day", 0) >= 1 or ev.get("newly_discovered"):
            eids = ev.get("evidence_ids") or []
            if not eids:
                sem_viols.append(f"dx_event_missing_evidence_id:{nm}")
                continue
            ok_ev = False
            for eid in eids:
                if st == "排除":
                    ok_ev = (eid in ex_by_id) and _exam_supports_excl(ex_by_id[eid], nm, pdx, code3, organs)
                elif st == "确诊" and nm == pdx and not ev.get("newly_discovered"):
                    if eid in ex_by_id:
                        ex = ex_by_id[eid]
                        if ex.get("result_day", -1) >= 1 and not ex.get("prior") and not ex.get("recheck") and not tkind and not is_status:
                            if ex.get("items"):
                                ok_ev = planner._lab_supports_dx(ex, pdx)
                            elif ex["key"] in ("病理", "活检"):
                                ok_ev = (
                                    bool(fname)
                                    and bool(ex.get("diagnostic"))
                                    and bool(ex.get("can_confirm"))
                                    and planner._pathology_can_confirm(ex, fname, organs, procs)
                                )
                            else:
                                ok_ev = (
                                    bool(fname)
                                    and bool(ex.get("diagnostic"))
                                    and bool(ex.get("can_confirm"))
                                    and planner._exam_pos_capability(ex["key"], ex["kind"], ex.get("region"), fname, organs) == "confirm_finding"
                                )
                elif st == "确诊" and (ev.get("newly_discovered") or nm != pdx):
                    if eid in ex_by_id:
                        ex = ex_by_id[eid]
                        if ex.get("result_day", -1) >= 1:
                            if ex.get("items"):
                                ok_ev = planner._lab_supports_dx(ex, nm)
                            else:
                                ok_ev = (
                                    nm in (ex.get("incidental_findings") or ())
                                    and eid not in excl_eids
                                    and planner._incidental_capable(ex, nm)
                                )
                    elif eid in pr_by_id:
                        ok_ev = _proc_supports_comorb(p, pr_by_id[eid], nm)
                if ok_ev:
                    break
            if not ok_ev:
                sem_viols.append(f"dx_event_unsupported_evidence:{nm}:{st}")

    return sem_viols, diag_pairs


def audit_planned_cohort(cohort_name, pairs_list, cohort_note=""):
    """Audit a list of (journey, specs) tuples."""
    contract_viols = collections.Counter()
    sem_viols = collections.Counter()
    surviving_events = collections.Counter()
    affected_stays = collections.Counter()
    diag_pairs = collections.Counter()
    dropped_names = collections.Counter()
    grounded_names = collections.Counter()
    chapter_agg = {}
    dx_agg = {}
    agg = _new_comorb_agg()

    for j, specs in pairs_list:
        for e in contract.validate_journey(j):
            contract_viols[e] += 1
        if specs is not None:
            for e in contract.validate_docspecs(specs, j["los"]):
                contract_viols[e] += 1
        svs, dpairs = _check_stay_semantics(j)
        for v in svs:
            sem_viols[v] += 1
        for dp in dpairs:
            diag_pairs[dp] += 1

        pdx = j["patient"]["primary_dx"]["name"]
        code3 = j["patient"]["primary_dx"].get("code3") or "?"
        ch = code3[:1] if code3 else "?"
        lg = j.get("comorb_ledger") or _empty_stay_ledger(j)
        _merge_comorb_agg(agg, 1, lg)
        _merge_comorb_agg(chapter_agg.setdefault(ch, _new_comorb_agg()), 1, lg)
        _merge_comorb_agg(dx_agg.setdefault(pdx, _new_comorb_agg()), 1, lg)

        for dn in lg.get("dropped_names") or ():
            dropped_names[dn] += 1

        evs = j.get("dx_events") or []
        has_p_conf = any(ev.get("name") == pdx and ev.get("state") == "确诊" and ev.get("day", 0) >= 1 for ev in evs)
        has_excl = any(ev.get("state") == "排除" for ev in evs)
        inpat_coms = [c for c in (j["patient"].get("comorbidities") or ()) if not c.get("at_admission")]
        if planner._has_any(pdx, planner._ADMISSION_STATUS_DX_WORDS) and any(
            ev.get("name") == pdx and ev.get("state") == "确诊" and ev.get("day", 0) == 0 for ev in evs
        ):
            affected_stays["status_dx_kept_at_admission"] += 1
        if has_p_conf:
            affected_stays["primary_confirm"] += 1
        if has_excl:
            affected_stays["diff_excl"] += 1
        if inpat_coms:
            affected_stays["inpatient_comorb"] += 1
        for c in inpat_coms:
            grounded_names[c["name"]] += 1

        for ev in evs:
            if ev.get("name") == pdx and ev.get("state") == "确诊" and ev.get("day", 0) >= 1:
                surviving_events["primary_confirm"] += 1
            elif ev.get("state") == "排除":
                surviving_events["diff_excl"] += 1
            elif ev.get("newly_discovered"):
                surviving_events["inpatient_comorb"] += 1

    ch_table = {ch: _finalize_comorb_agg(v) for ch, v in sorted(chapter_agg.items())}
    dx_sorted = sorted(dx_agg.items(), key=lambda kv: (-kv[1]["before_total"], -kv[1]["stays"], kv[0]))
    dx_table = {dx: _finalize_comorb_agg(v) for dx, v in dx_sorted[:40]}

    comorb_summary = _finalize_comorb_agg(agg)
    comorb_summary["top_20_dropped_names"] = dropped_names.most_common(20)
    comorb_summary["top_20_grounded_inpatient_names"] = grounded_names.most_common(20)
    comorb_summary["by_icd_chapter"] = ch_table
    comorb_summary["top_40_primary_dx"] = dx_table

    return {
        "cohort": cohort_name,
        "cohort_definition": cohort_note,
        "stays_checked": len(pairs_list),
        "contract_versions": sorted({j.get("contract") for j, _ in pairs_list}),
        "contract_violations_total": sum(contract_viols.values()),
        "contract_violations": dict(contract_viols),
        "semantic_rule_violations_total": sum(sem_viols.values()),
        "semantic_rule_violations": dict(sem_viols),
        "surviving_events": dict(surviving_events),
        "affected_stays": dict(affected_stays),
        "comorbidity_ledger": comorb_summary,
        "distinct_diagnostic_exam_dx_pairs": len(diag_pairs),
        "diagnostic_exam_dx_pairs_note": (
            "Count of distinct (exam_key, region, kind, can_confirm, finding, primary_dx) combinations "
            "conforming to current v0.3.7 capability rule tables (_exam_pos_capability / _pathology_can_confirm); "
            "verifies code-to-rule conformance, NOT independent clinical validation."
        ),
        "top_50_diagnostic_exam_dx_pairs": [
            {"key": k, "region": reg, "kind": knd, "can_confirm": cc, "finding": fnd, "primary_dx": dx, "count": cnt}
            for (k, reg, knd, cc, fnd, dx), cnt in diag_pairs.most_common(50)
        ],
    }


def _reconstruct_specs(docs):
    specs = []
    for i, d in enumerate(docs):
        specs.append({
            "k": d.get("k", i),
            "day": int(d["day_offset"]),
            "role": d.get("role") or "",
            "doc_type": d.get("doc_type") or "",
            "doc_type_cn": d.get("doc_type_cn") or "",
            "time": d.get("time") or "",
        })
    return specs


def _extract_doc_sections(text, plans):
    """Extract section texts matching compiler.compile_doc() across all 7 doc roles (HR/FP/SUP/DAILY/PRE/POST/DS)."""
    if len(plans) == 1 and not plans[0].get("title"):
        parts = (text or "").split("\n", 1)
        body = parts[1] if len(parts) == 2 else text or ""
        return {plans[0]["section"]: body}
    titles = [pl["title"] for pl in plans if pl.get("title")]
    title_pat = "|".join(re.escape(t) for t in titles)
    rx = re.compile(rf"(?m)^({title_pat})：")
    ms = list(rx.finditer(text or ""))
    out = {}
    title_to_sec = {pl["title"]: pl["section"] for pl in plans if pl.get("title")}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        sec_name = title_to_sec.get(m.group(1), m.group(1))
        out[sec_name] = (text or "")[m.end():end].strip()
    if len(out) < sum(1 for pl in plans if pl.get("mode") == "llm"):
        for sec in split_sections(text or ""):
            out.setdefault(sec.name, sec.text)
    return out


def audit_generated_parser(corpus_dir, stats_path):
    """Run gate.audit_item_numbers directly over every LLM section with lab_item_roles across all 7 roles."""
    adms = _read_jsonl(os.path.join(corpus_dir, "admissions.jsonl"))
    js = {j["journey_id"]: j for j in _read_jsonl(os.path.join(corpus_dir, "journeys.jsonl"))}
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    stays = 0
    docs_total = 0
    llm_sections_with_lab_roles = 0
    parsed_value_tokens = 0
    qualitative_only_mentions = 0
    unparsed_mentions = 0
    section_flag_counts = collections.Counter()
    hits_by_kind_and_item = collections.Counter()
    unparsed_samples = []
    flagged_samples = []

    for a in adms:
        jid = a["admission_id"]
        j = js.get(jid)
        if not j:
            continue
        stays += 1
        docs = a.get("docs") or []
        docs_total += len(docs)
        specs = _reconstruct_specs(docs)
        rng = random.Random(planner.seed_of(20261210, jid))
        for spec, d in zip(specs, docs):
            plans = compiler.compile_doc(j, spec, specs, stats, rng, ctx=None, epistemic=True)
            plans_by_sec = {pl["section"]: pl for pl in plans if pl.get("mode") == "llm" and pl.get("lab_item_roles")}
            if not plans_by_sec:
                continue
            secs_map = _extract_doc_sections(d["text"], plans)
            for sname, pl in plans_by_sec.items():
                sec_text = secs_map.get(sname)
                if sec_text is None:
                    continue
                llm_sections_with_lab_roles += 1
                res = gate.audit_item_numbers(pl, sec_text)
                parsed_value_tokens += res["parsed"]
                qualitative_only_mentions += res["qualitative_only"]
                unparsed_mentions += res["unparsed"]
                for fc in res["fails"]:
                    section_flag_counts[fc] += 1
                for h in res["hits"]:
                    kind, ik = h[0], h[1]
                    hits_by_kind_and_item[f"{kind}|{ik}"] += 1
                    if len(flagged_samples) < 30:
                        flagged_samples.append({
                            "jid": jid, "role": d.get("role"), "section": sname,
                            "kind": kind, "item": ik, "observed": h[2], "unit": h[3], "expected_latest": h[4],
                        })
                for ud in res["unparsed_details"]:
                    if len(unparsed_samples) < 30:
                        unparsed_samples.append({
                            "jid": jid, "role": d.get("role"), "section": sname,
                            "item": ud["item"], "seg": ud["seg"], "reason": ud["reason"],
                        })

    tot_pu = parsed_value_tokens + unparsed_mentions
    tot_all = tot_pu + qualitative_only_mentions
    return {
        "corpus": corpus_dir,
        "stays": stays,
        "docs_total": docs_total,
        "llm_sections_with_lab_roles": llm_sections_with_lab_roles,
        "counting_definition": {
            "parsed_value_tokens": "Lab item mentions paired with >=1 extractable lab numeric token or validated 1-to-1 ordered '分别' clause in gate.audit_item_numbers.",
            "qualitative_or_coord_only_mentions": "Lab item mentions immediately followed by qualitative wording (e.g. '未见异常/升高') or coordination before the next named lab item ('白蛋白、肌酐...').",
            "unparsed_mentions": "Lab item mentions where gate.audit_item_numbers could neither pair a numeric token nor classify the span as benign qualitative/coordinated mention (including ambiguous '分别' clauses).",
        },
        "parsed_value_tokens": parsed_value_tokens,
        "qualitative_or_coord_only_mentions": qualitative_only_mentions,
        "unparsed_mentions": unparsed_mentions,
        "unparsed_share_of_parsed_plus_unparsed": round(unparsed_mentions / tot_pu, 4) if tot_pu else 0.0,
        "unparsed_share_of_all_mentions": round(unparsed_mentions / tot_all, 4) if tot_all else 0.0,
        "section_flag_counts": dict(section_flag_counts),
        "hits_by_kind_and_item": hits_by_kind_and_item.most_common(50),
        "unparsed_30_samples": unparsed_samples,
        "flagged_hit_samples": flagged_samples,
    }


def _load_plan_pairs(plan_dir):
    js = _read_jsonl(os.path.join(plan_dir, "journeys.jsonl"))
    sp_map = {r["slot_id"]: r["specs"] for r in _read_jsonl(os.path.join(plan_dir, "specs.jsonl"))}
    return [(j, sp_map.get(j["journey_id"])) for j in js]


def build_pre_gpu_ledger(s5_root):
    s5_1 = []
    for s_idx in range(4):
        pdir = os.path.join(s5_root, "planned", "s5_1_shards", f"shard_{s_idx}")
        if os.path.isdir(pdir):
            s5_1.extend(_load_plan_pairs(pdir))
    s5_2_by_arm = {}
    s5_2_all = []
    for arm in ("route_b", "route_a_only", "route_ab"):
        pdir = os.path.join(s5_root, "planned", "s5_2_17d", arm)
        if os.path.isdir(pdir):
            pairs = _load_plan_pairs(pdir)
            s5_2_by_arm[arm] = pairs
            s5_2_all.extend(pairs)
    unique_slots = s5_1 + (s5_2_by_arm.get("route_ab") or [])
    all_arms = s5_1 + s5_2_all
    unique_note = (
        "Representative 2,340-stay single-arm slice combining S5-1 (2,000) + S5-2 route_ab (340). "
        "Because S5-2's three arms (route_b, route_a_only, route_ab) sample trajectories independently under slot=None, "
        "event counts in UNIQUE_SLOT_STAYS_2340 reflect S5-1 + route_ab only and must NOT be interpreted as events shared across all three S5-2 arms."
    )
    return {
        "scope_note": (
            "Re-verifies planned trajectories against medcourse_core v0.3.7 contract and capability rule tables "
            "(reusing generator capability predicates for zero-trust code-conformance checks; NOT independent clinical ground-truth validation)."
        ),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohorts": {
            "S5-1": audit_planned_cohort("S5-1", s5_1, "S5-1 paired counterfactual stays (2,000 trajectories shared across Full/VisOnly/OracleLeak)."),
            "S5-2_ALL_ARMS": audit_planned_cohort("S5-2_ALL_ARMS", s5_2_all, "All three S5-2 arms summed (340 x 3 = 1,020 independently sampled arm-stays)."),
            "S5-2_ROUTE_B": audit_planned_cohort("S5-2_ROUTE_B", s5_2_by_arm.get("route_b") or [], "S5-2 route_b (340 stays, --kb-only)."),
            "S5-2_ROUTE_A_ONLY": audit_planned_cohort("S5-2_ROUTE_A_ONLY", s5_2_by_arm.get("route_a_only") or [], "S5-2 route_a_only (340 stays)."),
            "S5-2_ROUTE_AB": audit_planned_cohort("S5-2_ROUTE_AB", s5_2_by_arm.get("route_ab") or [], "S5-2 route_ab (340 stays, route A+B)."),
            "UNIQUE_SLOT_STAYS_2340": audit_planned_cohort("UNIQUE_SLOT_STAYS_2340", unique_slots, unique_note),
            "ALL_ARM_STAYS_3020": audit_planned_cohort("ALL_ARM_STAYS_3020", all_arms, "All S5-1 (2,000) + all three S5-2 arms (1,020) = 3,020 arm-stays."),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r2")
    ap.add_argument("--parser-corpus", default="/home/weify/pyt/product/s5_r1/generated/s5_1/full_shard0")
    ap.add_argument("--stats", default="/home/weify/pyt/product/derived/m2_f27_v021/stats_v021.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-parser", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.s5_root, "tables")
    os.makedirs(out_dir, exist_ok=True)
    pre_path = os.path.join(out_dir, "v037_pre_gpu_audit.json")
    par_path = os.path.join(out_dir, "v037_item_num_parser_audit.json")

    pre = build_pre_gpu_ledger(args.s5_root)
    with open(pre_path, "w", encoding="utf-8") as f:
        json.dump(pre, f, ensure_ascii=False, indent=2)

    all_c = pre["cohorts"]["ALL_ARM_STAYS_3020"]
    print("V037_PRE_GPU_AUDIT", json.dumps({
        "sha256": _sha256_file(pre_path),
        "stays_all": all_c["stays_checked"],
        "stays_unique": pre["cohorts"]["UNIQUE_SLOT_STAYS_2340"]["stays_checked"],
        "contract_violations_total": all_c["contract_violations_total"],
        "semantic_rule_violations_total": all_c["semantic_rule_violations_total"],
        "surviving_events": all_c["surviving_events"],
        "affected_stays": all_c["affected_stays"],
    }, ensure_ascii=False))

    if not args.skip_parser:
        par = audit_generated_parser(args.parser_corpus, args.stats)
        with open(par_path, "w", encoding="utf-8") as f:
            json.dump(par, f, ensure_ascii=False, indent=2)
        print("V037_PARSER_AUDIT", json.dumps({
            "sha256": _sha256_file(par_path),
            "stays": par["stays"],
            "parsed_value_tokens": par["parsed_value_tokens"],
            "qualitative_or_coord_only_mentions": par["qualitative_or_coord_only_mentions"],
            "unparsed_mentions": par["unparsed_mentions"],
            "unparsed_share_of_parsed_plus_unparsed": par["unparsed_share_of_parsed_plus_unparsed"],
            "section_flag_counts": par["section_flag_counts"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
