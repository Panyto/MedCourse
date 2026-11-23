# -*- coding: utf-8 -*-
"""S6-2 60-case blinded clinical evaluation packager and statistical scorer.

Builds 15 prototypes x 4 arms (Real, MedCourse-Full, MedCourse-VisOnly, DIRECT-32B) = 60 stays,
strips source-identifying headers/placeholders into a uniform neutral layout, writes dual-rater
CSV sheets + SEALED_KEY_DO_NOT_OPEN.json, and scores Krippendorff's alpha, Cohen's kappa, and
paired Wilcoxon signed-rank tests with Holm-Bonferroni correction.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import random
import re

from s5_common import setup_paths
from t123_bench import EARLY_OP_RE, LATE_OP_RE, _completed_shard_paths, _is_discharge, _is_hr, _resolve_direct32b_paths

setup_paths()

from ruler_g1 import split_sections  # noqa: E402


STRATA = (
    ("internal_med", 4),
    ("surgery", 4),
    ("acute_critical", 4),
    ("special_pop", 3),
)

Q_COLS = (
    "q1_temporal_causal_1to5",
    "q2_epistemic_update_1to5",
    "q3_clinical_commonsense_1to5",
    "q4_medication_dosing_1to5",
    "q5_overall_usability_1to5",
)

_DEID_NOISE_RE = re.compile(
    r"\[(?:本院|患者|床号|住院号|姓名|医院)\]|(?:第\s*\d+\s*次\s*)?入\s+院\s+记\s+录|"
    r"(?:姓名|床号|住院号|入院日期|出院日期)[:：][^\n]{0,30}"
)
_AGE_NUM_RE = re.compile(r"(\d{1,3})\s*岁")


def assert_blind_fold_allowed(real_fold, allow_stage2_fold9=False):
    if real_fold and "fold9" in str(real_fold).lower() and not allow_stage2_fold9:
        raise ValueError("STAGE1_FORBIDS_FOLD9_BLIND_PACK")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_stratum(adm):
    """Assign admission to one of the 4 clinical strata for balanced 15-prototype sampling."""
    docs = adm.get("docs") or []
    txt = "\n".join(d.get("text") or "" for d in docs)
    meta = adm.get("meta") or {}
    dept = str(meta.get("dept") or "")
    age = meta.get("age")
    if age is None:
        m = _AGE_NUM_RE.search(txt[:400])
        age = int(m.group(1)) if m else 50

    if (isinstance(age, (int, float)) and age < 14) or any(
        w in (dept + txt[:600]) for w in ("儿科", "产科", "妊娠", "剖宫产", "肿瘤", "化疗", "Z51")
    ):
        return "special_pop"
    if any(w in txt[:1200] for w in ("心肌梗死", "心力衰竭", "脑梗死", "急性胰腺炎", "呼吸衰竭", "重症", "ICU", "休克")):
        return "acute_critical"
    if EARLY_OP_RE.search(txt) or LATE_OP_RE.search(txt) or "外科" in dept:
        return "surgery"
    return "internal_med"


def _doc_label(doc):
    if _is_hr(doc):
        return "入院记录"
    if _is_discharge(doc):
        return "出院记录"
    role = (doc.get("role") or "").upper()
    return {
        "FP": "首次病程记录",
        "SUP": "上级医师查房记录",
        "PRE": "术前小结",
        "POST": "术后首次病程记录",
    }.get(role, doc.get("doc_type_cn") or "日常病程记录")


_ABS_DATE_RE = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?")


def _parse_base_date(adm):
    """Extract earliest valid calendar date in admission docs / metadata as Day-0 anchor."""
    meta_d = (adm.get("meta") or {}).get("admit_date") or ""
    candidates = []
    for txt in [meta_d] + [(d.get("text") or "")[:600] for d in (adm.get("docs") or [])[:2]]:
        for m in _ABS_DATE_RE.finditer(txt):
            try:
                candidates.append(dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
    return candidates[0] if candidates else None


def normalize_for_blind(adm, blind_id):
    """Strip identifying headers/de-ID tags, preserve None day_offset as unknown, and keep relative date deltas."""
    lines = [f"【盲评病历编号：{blind_id}】"]
    base_date = _parse_base_date(adm)
    orig_docs = list(adm.get("docs") or [])

    last_known_day = -1
    order_pairs = []
    for orig_idx, d in enumerate(orig_docs):
        raw_dy = d.get("day_offset")
        if raw_dy is not None:
            last_known_day = int(raw_dy)
            sort_dy = int(raw_dy)
        else:
            sort_dy = max(0, last_known_day)
        order_pairs.append((sort_dy, orig_idx, d))
    order_pairs.sort(key=lambda x: (x[0], int(x[2].get("k") if x[2].get("k") is not None else x[1])))

    def _rel_date_repl(m):
        if base_date is None:
            return "[日期已隐去]"
        try:
            cur = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            delta = (cur - base_date).days
            if -30 <= delta <= 90:
                return f"[入院第+{delta}天]" if delta >= 0 else f"[入院前{abs(delta)}天]"
        except ValueError:
            pass
        return "[日期已隐去]"

    for idx, (_, _, d) in enumerate(order_pairs):
        raw_dy = d.get("day_offset")
        day_lbl = f"住院第 {int(raw_dy)} 天" if raw_dy is not None else "住院日：日期未记载"
        lines.append(f"--- 文书 {idx + 1}（{day_lbl} | {_doc_label(d)}）---")
        raw = (d.get("text") or "").replace("\r\n", "\n")
        cleaned_lines = []
        for ln in raw.splitlines():
            ln = _DEID_NOISE_RE.sub("", ln)
            ln = _ABS_DATE_RE.sub(_rel_date_repl, ln)
            ln = re.sub(r"[ \t]{2,}", " ", ln).strip()
            if ln:
                cleaned_lines.append(ln)
        lines.append("\n".join(cleaned_lines))
    return "\n\n".join(lines)


def _select_matched_15(real_adms, full_adms, vis_adms, direct_adms, seed=20261240):
    rng = random.Random(seed)
    vis_by_id = {str(a.get("admission_id")): a for a in vis_adms}
    paired_full = [a for a in full_adms if str(a.get("admission_id")) in vis_by_id]

    def _bucket(adms):
        b = collections.defaultdict(list)
        for a in adms:
            b[classify_stratum(a)].append(a)
        for k in b:
            rng.shuffle(b[k])
        return b

    b_real = _bucket(real_adms)
    b_full = _bucket(paired_full)
    b_dir = _bucket(direct_adms)

    used_ids = {"Real": set(), "MedCourse-Full": set(), "DIRECT": set()}
    prototypes = []
    proto_idx = 0
    for stratum, need in STRATA:
        for _ in range(need):
            def _pick(bucket, fallback_all, tag):
                cand_list = bucket.get(stratum) or []
                for a in cand_list:
                    aid = str(a.get("admission_id"))
                    if aid not in used_ids[tag]:
                        used_ids[tag].add(aid)
                        return a
                for a in fallback_all:
                    aid = str(a.get("admission_id"))
                    if aid not in used_ids[tag]:
                        used_ids[tag].add(aid)
                        return a
                raise RuntimeError(f"Insufficient admissions for stratum={stratum} arm={tag}")

            a_real = _pick(b_real, real_adms, "Real")
            a_full = _pick(b_full, paired_full, "MedCourse-Full")
            a_vis = vis_by_id[str(a_full.get("admission_id"))]
            a_dir = _pick(b_dir, direct_adms, "DIRECT")
            prototypes.append({
                "proto_idx": proto_idx,
                "stratum": stratum,
                "arms": {
                    "Real": a_real,
                    "MedCourse-Full": a_full,
                    "MedCourse-VisOnly": a_vis,
                    "DIRECT-32B": a_dir,
                },
            })
            proto_idx += 1
    return prototypes


def build_blind_package(real_adms, full_adms, vis_adms, direct_adms, out_dir, seed=20261240):
    """Create 60 blinded packets, dual-rater CSV sheets (with guess probe), and sealed key JSON."""
    protos = _select_matched_15(real_adms, full_adms, vis_adms, direct_adms, seed=seed)
    rng = random.Random(seed)
    items = []
    for pr in protos:
        for arm_name, adm in pr["arms"].items():
            act_st = classify_stratum(adm)
            items.append({
                "proto_idx": pr["proto_idx"],
                "stratum": pr["stratum"],
                "actual_stratum": act_st,
                "used_fallback_stratum": act_st != pr["stratum"],
                "arm": arm_name,
                "source_admission_id": str(adm.get("admission_id")),
                "adm": adm,
            })
    rng.shuffle(items)

    os.makedirs(out_dir, exist_ok=True)
    key_rows = []
    packet_blocks = []
    sheet_rows = []
    for idx, it in enumerate(items):
        bid = f"B{idx + 1:02d}"
        norm_txt = normalize_for_blind(it["adm"], bid)
        packet_blocks.append(norm_txt)
        key_rows.append({
            "blind_id": bid,
            "proto_idx": it["proto_idx"],
            "target_stratum": it["stratum"],
            "actual_stratum": it["actual_stratum"],
            "used_fallback_stratum": it["used_fallback_stratum"],
            "arm": it["arm"],
            "source_admission_id": it["source_admission_id"],
        })
        sheet_rows.append({
            "blind_id": bid,
            "stratum_hint": it["actual_stratum"],
            **{q: "" for q in Q_COLS},
            "hard_defect_0or1": "",
            "guess_is_synthetic_0or1": "",
            "defect_note": "",
        })

    with open(os.path.join(out_dir, "blind_packets.md"), "w", encoding="utf-8") as f:
        f.write("\n\n" + ("=" * 60 + "\n\n").join(packet_blocks) + "\n")

    fields = ["blind_id", "stratum_hint", *Q_COLS, "hard_defect_0or1", "guess_is_synthetic_0or1", "defect_note"]
    for rater in ("rater_A_sheet.csv", "rater_B_sheet.csv"):
        with open(os.path.join(out_dir, rater), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sheet_rows:
                w.writerow(r)

    key_obj = {"seed": seed, "total_cases": len(key_rows), "mapping": key_rows}
    with open(os.path.join(out_dir, "SEALED_KEY_DO_NOT_OPEN.json"), "w", encoding="utf-8") as f:
        json.dump(key_obj, f, ensure_ascii=False, indent=2)
    return key_obj


def krippendorff_alpha_interval(rater_pairs):
    """Compute interval Krippendorff's alpha over paired numeric ratings [(a_i, b_i), ...]."""
    valid = [(float(a), float(b)) for a, b in rater_pairs if a is not None and b is not None]
    n = len(valid)
    if n < 2:
        return None
    do_num = sum((a - b) ** 2 for a, b in valid) / n
    pooled = [v for pair in valid for v in pair]
    mean_v = sum(pooled) / len(pooled)
    de_num = 2.0 * sum((v - mean_v) ** 2 for v in pooled) / (2 * n - 1)
    if de_num <= 1e-12:
        return 1.0
    return round(1.0 - do_num / de_num, 4)


def cohens_kappa_binary(pairs):
    """Compute Cohen's kappa over paired 0/1 binary ratings [(a_i, b_i), ...]."""
    valid = [(int(a), int(b)) for a, b in pairs if a is not None and b is not None]
    n = len(valid)
    if n == 0:
        return None
    po = sum(1 for a, b in valid if a == b) / n
    pa1 = sum(a for a, _ in valid) / n
    pb1 = sum(b for _, b in valid) / n
    pe = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if 1.0 - pe <= 1e-12:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def _paired_wilcoxon_p(xs, ys):
    """Two-sided paired Wilcoxon signed-rank normal-approx p-value (exact tie handling)."""
    diffs = [float(x) - float(y) for x, y in zip(xs, ys) if abs(float(x) - float(y)) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 1.0
    abs_d = sorted((abs(d), i, d) for i, d in enumerate(diffs))
    ranks = [0.0] * n
    tie_adj = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(abs_d[j][0] - abs_d[i][0]) <= 1e-12:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        t = j - i
        if t > 1:
            tie_adj += t * (t * t - 1)
        i = j
    w_plus = sum(ranks[k] for k in range(n) if abs_d[k][2] > 0)
    mu = n * (n + 1) / 4.0
    var = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_adj) / 24.0
    if var <= 1e-12:
        return 1.0
    z = (w_plus - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(min(1.0, max(0.0, p)))


def _holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values preserving input order."""
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda kv: kv[1])
    adj = [1.0] * m
    prev_max = 0.0
    for rank, (orig_i, p) in enumerate(indexed):
        val = min(1.0, (m - rank) * p)
        prev_max = max(prev_max, val)
        adj[orig_i] = round(prev_max, 6)
    return adj


def _clopper_pearson_ci(k, n, alpha=0.05):
    if n <= 0:
        return [0.0, 1.0]

    def _binom_sf(p, k_lo):
        s = 0.0
        for i in range(k_lo, n + 1):
            lp = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            lp += i * math.log(max(p, 1e-15)) + (n - i) * math.log(max(1.0 - p, 1e-15))
            s += math.exp(lp)
        return s

    def _binom_cdf(p, k_hi):
        s = 0.0
        for i in range(0, k_hi + 1):
            lp = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            lp += i * math.log(max(p, 1e-15)) + (n - i) * math.log(max(1.0 - p, 1e-15))
            s += math.exp(lp)
        return s

    lo = 0.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (a + b)
            if _binom_sf(mid, k) < alpha / 2.0:
                a = mid
            else:
                b = mid
        lo = 0.5 * (a + b)
    hi = 1.0
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (a + b)
            if _binom_cdf(mid, k) < alpha / 2.0:
                b = mid
            else:
                a = mid
        hi = 0.5 * (a + b)
    return [round(lo, 4), round(hi, 4)]


def _van_elteren_p(xs_by_st, ys_by_st):
    """Stratified van Elteren two-sided rank-sum test across clinical strata (exact tie handling)."""
    num = 0.0
    var = 0.0
    strata = sorted(set(xs_by_st) & set(ys_by_st))
    for st in strata:
        xh = [float(v) for v in xs_by_st[st]]
        yh = [float(v) for v in ys_by_st[st]]
        n1, n2 = len(xh), len(yh)
        if n1 == 0 or n2 == 0:
            continue
        N = n1 + n2
        pooled = sorted([(v, 1) for v in xh] + [(v, 0) for v in yh], key=lambda t: t[0])
        ranks_x_sum = 0.0
        t_corr = 0.0
        i = 0
        while i < N:
            j = i + 1
            while j < N and abs(pooled[j][0] - pooled[i][0]) <= 1e-12:
                j += 1
            avg_r = 0.5 * ((i + 1) + j)
            for k in range(i, j):
                if pooled[k][1] == 1:
                    ranks_x_sum += avg_r
            t = j - i
            if t > 1:
                t_corr += t * (t * t - 1)
            i = j
        e_h = n1 * (N + 1) / 2.0
        v_h = n1 * n2 * (N * (N * N - 1) - t_corr) / (12.0 * N * max(1, N - 1))
        w_h = 1.0 / (N + 1)
        num += w_h * (ranks_x_sum - e_h)
        var += (w_h ** 2) * v_h
    if var <= 1e-12:
        return 1.0
    z = num / math.sqrt(var)
    return float(min(1.0, max(0.0, math.erfc(abs(z) / math.sqrt(2.0)))))


def score_blind_sheets(sheet_a_csv, sheet_b_csv, key_json, out_json=None):
    """Score dual-blind clinician sheets using paired Wilcoxon (Full vs VisOnly) and van Elteren (cross-arm)."""
    def _read_sheet(p):
        out = {}
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["blind_id"]] = row
        return out

    ra = _read_sheet(sheet_a_csv)
    rb = _read_sheet(sheet_b_csv)
    with open(key_json, encoding="utf-8") as f:
        key = json.load(f)

    alpha = {}
    for q in Q_COLS:
        pairs = []
        for m in key["mapping"]:
            bid = m["blind_id"]
            va = ra.get(bid, {}).get(q, "")
            vb = rb.get(bid, {}).get(q, "")
            if str(va).strip() != "" and str(vb).strip() != "":
                pairs.append((float(va), float(vb)))
        alpha[q] = krippendorff_alpha_interval(pairs)

    hd_pairs = []
    guess_correct, guess_total = 0, 0
    for m in key["mapping"]:
        bid = m["blind_id"]
        is_syn_true = 0 if m["arm"] == "Real" else 1
        ha = ra.get(bid, {}).get("hard_defect_0or1", "")
        hb = rb.get(bid, {}).get("hard_defect_0or1", "")
        if str(ha).strip() != "" and str(hb).strip() != "":
            hd_pairs.append((int(float(ha)), int(float(hb))))
        for sh in (ra.get(bid, {}), rb.get(bid, {})):
            g = sh.get("guess_is_synthetic_0or1", "")
            if str(g).strip() != "":
                guess_total += 1
                if int(float(g)) == is_syn_true:
                    guess_correct += 1
    kappa_hd = cohens_kappa_binary(hd_pairs)

    by_arm_proto = collections.defaultdict(dict)
    by_arm_vals = collections.defaultdict(lambda: collections.defaultdict(list))
    by_arm_stratum_vals = collections.defaultdict(lambda: collections.defaultdict(list))
    by_arm_hd = collections.defaultdict(list)
    for m in key["mapping"]:
        bid = m["blind_id"]
        arm = m["arm"]
        pidx = m["proto_idx"]
        act_st = m.get("actual_stratum") or m.get("stratum") or "all"
        for q in Q_COLS:
            vals = []
            for s in (ra.get(bid, {}), rb.get(bid, {})):
                v = s.get(q, "")
                if str(v).strip() != "":
                    vals.append(float(v))
            if vals:
                avg_v = sum(vals) / len(vals)
                by_arm_vals[arm][q].append(avg_v)
                by_arm_proto[(arm, q)][pidx] = avg_v
                by_arm_stratum_vals[(arm, q)][act_st].append(avg_v)
        ha = ra.get(bid, {}).get("hard_defect_0or1", "")
        hb = rb.get(bid, {}).get("hard_defect_0or1", "")
        hds = [int(float(x)) for x in (ha, hb) if str(x).strip() != ""]
        if hds:
            by_arm_hd[arm].append(1 if any(hds) else 0)

    arm_summary = {}
    for arm in sorted({m["arm"] for m in key["mapping"]}):
        hd_k = int(sum(by_arm_hd[arm]))
        hd_n = len(by_arm_hd[arm])
        arm_summary[arm] = {
            "n": len(by_arm_vals[arm][Q_COLS[0]]),
            "means": {q: round(sum(vs) / len(vs), 3) if vs else None for q, vs in by_arm_vals[arm].items()},
            "hard_defect_any_rater_hits": f"{hd_k}/{hd_n}",
            "hard_defect_any_rater_rate": round(hd_k / max(1, hd_n), 4),
            "hard_defect_any_rater_ci95": _clopper_pearson_ci(hd_k, hd_n) if hd_n else None,
        }

    comparators = ("Real", "MedCourse-VisOnly", "DIRECT-32B")
    stat_tests = {}
    for q in Q_COLS:
        raw_ps = []
        comp_stats = []
        for comp in comparators:
            if comp == "MedCourse-VisOnly":
                common_p = sorted(set(by_arm_proto[("MedCourse-Full", q)]) & set(by_arm_proto[(comp, q)]))
                xs = [by_arm_proto[("MedCourse-Full", q)][i] for i in common_p]
                ys = [by_arm_proto[(comp, q)][i] for i in common_p]
                p = _paired_wilcoxon_p(xs, ys)
                diff = round(sum(xs) / max(1, len(xs)) - sum(ys) / max(1, len(ys)), 3)
                test_nm, n_obs = "paired_wilcoxon", len(common_p)
            else:
                xs_map = by_arm_stratum_vals[("MedCourse-Full", q)]
                ys_map = by_arm_stratum_vals[(comp, q)]
                p = _van_elteren_p(xs_map, ys_map)
                xs_all = by_arm_vals["MedCourse-Full"][q]
                ys_all = by_arm_vals[comp][q]
                diff = round(sum(xs_all) / max(1, len(xs_all)) - sum(ys_all) / max(1, len(ys_all)), 3)
                test_nm, n_obs = "stratified_van_elteren", len(xs_all) + len(ys_all)
            raw_ps.append(p)
            comp_stats.append((comp, test_nm, n_obs, diff, round(p, 6)))
        adj = _holm_adjust(raw_ps)
        stat_tests[q] = {
            comp: {"test": tnm, "n": nobs, "diff_full_minus_comp": df_, "raw_p": rp_, "holm_p": hp_}
            for (comp, tnm, nobs, df_, rp_), hp_ in zip(comp_stats, adj)
        }

    report = {
        "agreement": {
            "krippendorff_alpha": alpha,
            "cohens_kappa_hard_defect": kappa_hd,
            "source_guess_accuracy": round(guess_correct / guess_total, 4) if guess_total else None,
        },
        "arms": arm_summary,
        "comparisons_holm": stat_tests,
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _make_dry_run_fixture():
    strata_texts = {
        "internal_med": "心内科一区 2型糖尿病 肺炎 口服降糖",
        "surgery": "普通外科一区 拟行腹腔镜胆囊切除术 术后首次病程 今日在全麻下行腹腔镜下胆囊切除术",
        "acute_critical": "心内科二区 急性心肌梗死 重症 心力衰竭",
        "special_pop": "儿科一区 5岁 小儿支气管肺炎 咳嗽发热",
    }
    arms = {"Real": [], "MedCourse-Full": [], "MedCourse-VisOnly": [], "DIRECT-32B": []}
    for st, desc in strata_texts.items():
        for i in range(6):
            gid = f"SHARED-{st}-{i}"
            for arm_k in arms:
                aid = gid if arm_k in ("MedCourse-Full", "MedCourse-VisOnly") else f"{arm_k}-{st}-{i}"
                docs = [
                    {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "k": 0,
                     "text": f"[本院]\n姓名：[患者]科室：普通外科一区年龄:52岁 床号：[床号]住院号：[住院号]\n主诉：{desc}\n现病史：发作1天。\n入院诊断：{desc}"},
                    {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "k": 1,
                     "text": f"病例特点：52岁，{desc}。\n拟诊讨论：待查。"},
                    {"doc_type": "discharge_summary", "role": "DS", "day_offset": 4, "k": 2,
                     "text": "入院情况：平稳。\n诊疗经过：规范治疗好转。\n出院诊断：明确诊断。"},
                ]
                arms[arm_k].append({"admission_id": aid, "docs": docs, "meta": {"age": 5 if st == "special_pop" else 52}})
    return arms["Real"], arms["MedCourse-Full"], arms["MedCourse-VisOnly"], arms["DIRECT-32B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--real-fold", default="/home/weify/pyt/data/deid_run9_fold8.jsonl")
    ap.add_argument("--direct32b-jsonl", default=None)
    ap.add_argument("--allow-stage2-fold9", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20261240)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.dry_run:
        r, f_adms, v_adms, d_adms = _make_dry_run_fixture()
    else:
        assert_blind_fold_allowed(args.real_fold, allow_stage2_fold9=args.allow_stage2_fold9)
        fa, _ = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400)
        va, _ = _completed_shard_paths(args.s5_root, "s5_1", "vis_only_shard", min_kept=400)
        da = _resolve_direct32b_paths(args.s5_root, args.direct32b_jsonl)
        r = _read_jsonl(args.real_fold)
        f_adms = [a for p in fa for a in _read_jsonl(p)]
        v_adms = [a for p in va for a in _read_jsonl(p)]
        d_adms = [a for p in da for a in _read_jsonl(p)]

    key_obj = build_blind_package(r, f_adms, v_adms, d_adms, args.out_dir, seed=args.seed)
    print("S6_BLIND_PACK_DONE", json.dumps({"out_dir": args.out_dir, "total": key_obj["total_cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
