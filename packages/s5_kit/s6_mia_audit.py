# -*- coding: utf-8 -*-
"""S6-1 stratified empirical Membership Inference Audit (MIA) with positive controls.

Stage-1 calibration strictly uses folds 0-8 (fold 8 non-members) and hard-refuses
any file path referencing fold 9. Stage-2 lock freezes exact calibrated score weights
and runs once against held-out fold 9 non-members.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import pickle
import re

from s5_common import setup_paths
from t123_bench import _completed_shard_paths, _is_discharge, _is_hr, _resolve_direct32b_paths

setup_paths()

from medcourse.synth_export import lines_of, norm, pattern_hits, rare_span_counts  # noqa: E402
from ruler_g1 import split_sections  # noqa: E402

import numpy as np  # noqa: E402
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
    from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


_DEPT_RE = re.compile(r"科室[:：]\s*([^\s年龄床住]{2,12})")
_SEX_RE = re.compile(r"性别[:：]\s*(男|女)|(男|女)性")
_AGE_RE = re.compile(r"(?:年龄[:：]\s*)?(\d{1,3})\s*岁")


def assert_no_fold9(paths, mode):
    """Hard guard refusing fold 9 inputs during Stage-1 calibration."""
    if mode != "stage1_cal":
        return
    for p in paths or ():
        if p and "fold9" in str(p).lower():
            raise ValueError("STAGE1_FORBIDS_FOLD9")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_profile(adm):
    """Extract normalized text, writer lines, disease prefix, and QI tuple without saving raw strings."""
    docs = adm.get("docs") or []
    hr = next((d for d in docs if _is_hr(d)), docs[0] if docs else None)
    hr_txt = (hr or {}).get("text") or ""
    meta = adm.get("meta") or {}

    dm = _DEPT_RE.search(hr_txt)
    dept = meta.get("dept") or (dm.group(1) if dm else "UNK")
    sm = _SEX_RE.search(hr_txt)
    sex = meta.get("sex") or ((sm.group(1) or sm.group(2)) if sm else "UNK")
    am = _AGE_RE.search(hr_txt)
    age = meta.get("age") if meta.get("age") is not None else (int(am.group(1)) if am else -1)

    days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
    los = max(days) if days else 0
    los_bin = "short" if los <= 3 else ("mid" if los <= 7 else "long")

    dx4 = "UNK"
    if hr_txt:
        secs = {s.name: s.text for s in split_sections(hr_txt)}
        adx = secs.get("入院诊断", "").strip().splitlines()
        if adx:
            c = norm(re.sub(r"^\d+[\.、\s]*", "", adx[0]).strip())
            if len(c) >= 2:
                dx4 = c[:6]

    wlines = [ln for kind, ln in lines_of(adm) if kind == "writer" and len(ln) >= 12]
    stay_norm = "".join(norm(d.get("text") or "") for d in docs)
    return {
        "aid": str(adm.get("admission_id") or ""),
        "dept": dept,
        "sex": sex,
        "age": age,
        "los": los,
        "los_bin": los_bin,
        "dx4": dx4,
        "qi": (dept, sex, age, dx4[:4], los),
        "writer_lines": wlines,
        "stay_norm": stay_norm[:8000],
    }


def _match_non_members(member_profiles, pool_profiles, seed=20261201):
    """Stratified 1:1 non-member selection matching (dx4[:3], los_bin) with match diagnostics."""
    rng = np.random.RandomState(seed)
    by_stratum = collections.defaultdict(list)
    for idx, p in enumerate(pool_profiles):
        by_stratum[(p["dx4"][:3], p["los_bin"])].append(idx)
    by_los = collections.defaultdict(list)
    for idx, p in enumerate(pool_profiles):
        by_los[p["los_bin"]].append(idx)

    for k in by_stratum:
        rng.shuffle(by_stratum[k])
    for k in by_los:
        rng.shuffle(by_los[k])

    used = set()
    chosen = []
    exact_hits = los_hits = rand_hits = 0
    for mp in member_profiles:
        picked = None
        for idx in by_stratum.get((mp["dx4"][:3], mp["los_bin"]), ()):
            if idx not in used:
                picked = idx
                exact_hits += 1
                break
        if picked is None:
            for idx in by_los.get(mp["los_bin"], ()):
                if idx not in used:
                    picked = idx
                    los_hits += 1
                    break
        if picked is None:
            rem = [i for i in range(len(pool_profiles)) if i not in used]
            if rem:
                picked = int(rem[rng.randint(0, len(rem))])
                rand_hits += 1
        if picked is not None:
            used.add(picked)
            chosen.append(pool_profiles[picked])
    tot = max(1, len(chosen))
    diag = {
        "n_matched": len(chosen),
        "exact_dx3_los_rate": round(exact_hits / tot, 4),
        "los_fallback_rate": round(los_hits / tot, 4),
        "random_fallback_rate": round(rand_hits / tot, 4),
    }
    return chosen, diag


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


def _empirical_1pct_tpr(y_true, s_vals):
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, thrs = roc_curve(y_true, s_vals)
    idx_1 = np.where(fpr <= 0.01)[0]
    if not idx_1.size:
        return 0.0
    thr_1 = float(thrs[idx_1[-1]])
    return float(np.sum((y_true == 1) & (s_vals >= thr_1))) / max(1, int(np.sum(y_true == 1)))


def _boot_auc_ci(y_true, scores, n_boot=2000, seed=20261201):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "ci95": None, "tpr_at_1pct_fpr": None}
    auc = float(roc_auc_score(y, s))
    fpr, tpr, thrs = roc_curve(y, s)
    idx_1 = np.where(fpr <= 0.01)[0]
    thr_1 = float(thrs[idx_1[-1]]) if len(idx_1) else float("inf")
    n_pos = int(np.sum(y == 1))
    hits_1 = int(np.sum((y == 1) & (s >= thr_1))) if idx_1.size else 0
    tpr_1 = hits_1 / max(1, n_pos)

    tpr_1_neg = _empirical_1pct_tpr(y, -s) or 0.0

    rng = np.random.RandomState(seed)
    boots, boot_tprs_pos, boot_tprs_neg = [], [], []
    for _ in range(n_boot):
        bi = rng.choice(len(y), size=len(y), replace=True)
        yb, sb = y[bi], s[bi]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, sb))
        tp_b = _empirical_1pct_tpr(yb, sb)
        tn_b = _empirical_1pct_tpr(yb, -sb)
        if tp_b is not None:
            boot_tprs_pos.append(tp_b)
        if tn_b is not None:
            boot_tprs_neg.append(tn_b)
    ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if boots else None
    return {
        "roc_auc": round(auc, 4),
        "auc_pos_direction": round(auc, 4),
        "auc_neg_direction": round(1.0 - auc, 4),
        "inverted_auc": round(1.0 - auc, 4),
        "two_sided_dev_from_05": round(abs(auc - 0.5), 4),
        "attack_polarity_scope": "both_directions_evaluated (higher_score_is_member_pos_and_lower_score_is_member_neg)",
        "ci95": ci,
        "tpr_at_1pct_fpr": round(tpr_1, 4),
        "tpr_at_1pct_fpr_neg_dir": round(float(tpr_1_neg), 4),
        "tpr_1pct_hits": f"{hits_1}/{n_pos}",
        "tpr_1pct_cp_ci95": _clopper_pearson_ci(hits_1, n_pos),
        "empirical_tpr_1pct_pos_joint_boot_ci95": [
            round(float(np.percentile(boot_tprs_pos, 2.5)), 4),
            round(float(np.percentile(boot_tprs_pos, 97.5)), 4),
        ] if boot_tprs_pos else None,
        "empirical_tpr_1pct_neg_joint_boot_ci95": [
            round(float(np.percentile(boot_tprs_neg, 2.5)), 4),
            round(float(np.percentile(boot_tprs_neg, 97.5)), 4),
        ] if boot_tprs_neg else None,
    }


def _freeze_upper_1pct_thr(scores_non):
    """Conservative threshold so at most floor(0.01 * N) Stage-1 non-members satisfy s >= thr."""
    s = np.asarray(scores_non, dtype=float)
    n = len(s)
    if n == 0:
        return float("inf")
    k_allow = int(math.floor(0.01 * n))
    sd = np.sort(s)[::-1]
    if k_allow <= 0 or sd[0] == sd[-1]:
        return float(sd[0]) + 1e-12
    thr = float(sd[k_allow - 1])
    if int(np.sum(s >= thr)) > k_allow:
        strictly_hi = s[s > sd[k_allow]]
        thr = float(np.min(strictly_hi)) if strictly_hi.size else (float(sd[0]) + 1e-12)
    return thr


def _freeze_lower_1pct_thr(scores_non):
    """Symmetric conservative lower-tail threshold so at most floor(0.01 * N) satisfy s <= thr."""
    return float(-_freeze_upper_1pct_thr(-np.asarray(scores_non, dtype=float)))


def eval_frozen_bidirectional_thresholds(y_true, scores, thr_pos, thr_neg):
    """Evaluate fixed Stage-1 thresholds cleanly on independent Stage-2 (or split-half) samples."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    def _one_dir(hit_arr, thr_val):
        tp_k = int(np.sum(hit_arr[pos_mask])) if n_pos else 0
        fp_k = int(np.sum(hit_arr[neg_mask])) if n_neg else 0
        return {
            "thr_frozen": round(float(thr_val), 6),
            "realized_fpr": round(fp_k / max(1, n_neg), 4),
            "fpr_hits": f"{fp_k}/{n_neg}",
            "fpr_cp_ci95": _clopper_pearson_ci(fp_k, n_neg, alpha=0.05),
            "fpr_bonferroni_ci975": _clopper_pearson_ci(fp_k, n_neg, alpha=0.025),
            "realized_tpr": round(tp_k / max(1, n_pos), 4),
            "tpr_hits": f"{tp_k}/{n_pos}",
            "tpr_cp_ci95": _clopper_pearson_ci(tp_k, n_pos, alpha=0.05),
            "tpr_bonferroni_ci975": _clopper_pearson_ci(tp_k, n_pos, alpha=0.025),
        }

    return {
        "pos_direction_higher_is_member": _one_dir(s >= float(thr_pos), thr_pos),
        "neg_direction_lower_is_member": _one_dir(s <= float(thr_neg), thr_neg),
        "multiplicity_rule": "Bidirectional fixed-threshold checks use two-sided Bonferroni alpha=0.025 per direction (family-wise 0.05).",
    }


def _row_max_top2_cos(X_q, X_s):
    if X_q.shape[0] == 0 or X_s.shape[0] == 0:
        z = np.zeros(X_q.shape[0], dtype=float)
        return z, z
    top1 = np.zeros(X_q.shape[0], dtype=float)
    top2 = np.zeros(X_q.shape[0], dtype=float)
    step = 400
    for i in range(0, X_q.shape[0], step):
        block = (X_q[i:i + step] @ X_s.T).toarray()
        if block.shape[1] == 1:
            top1[i:i + step] = block[:, 0]
            top2[i:i + step] = block[:, 0]
        else:
            part = np.partition(block, -2, axis=1)
            top1[i:i + step] = part[:, -1]
            top2[i:i + step] = part[:, -2]
    ratio = top1 / (top2 + 1e-4)
    return top1, ratio


def _grams_from_lines(lines, g=30):
    out = set()
    for ln in lines:
        if len(ln) >= g:
            for i in range(len(ln) - g + 1):
                out.add(ln[i:i + g])
    return out


def _rare_30gram_query_scores(member_profiles, non_member_profiles, synth_profiles):
    """Per-real-record count of rare real 30-grams present in Synth writer text."""
    all_real = member_profiles + non_member_profiles
    df = collections.Counter()
    per_q_grams = []
    for p in all_real:
        gs = _grams_from_lines(p["writer_lines"], 30)
        per_q_grams.append(gs)
        for gram in gs:
            df[gram] += 1

    synth_grams = set()
    for sp in synth_profiles:
        synth_grams |= _grams_from_lines(sp["writer_lines"], 30)

    scores = []
    for gs in per_q_grams:
        hits = sum(1 for gram in gs if 1 <= df[gram] <= 10 and gram in synth_grams)
        scores.append(float(hits))
    return np.array(scores, dtype=float)


def fit_frozen_arm_scorer(synth_profiles, cal_member_profiles, cal_non_member_profiles):
    """Fit and freeze TF-IDF vectorizers, reference 30-gram dictionary, and bidirectional 1% FPR thresholds."""
    stay_s = [p["stay_norm"] for p in synth_profiles]
    vec_stay = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=50000, sublinear_tf=True)
    X_s = vec_stay.fit_transform(stay_s)

    synth_lines = [ln for sp in synth_profiles for ln in sp["writer_lines"]][:25000]
    vec_ln = None
    XL_s = None
    if synth_lines:
        vec_ln = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2 if len(synth_lines) > 50 else 1,
            max_features=40000,
            sublinear_tf=True,
        )
        try:
            XL_s = vec_ln.fit_transform(synth_lines)
        except ValueError:
            vec_ln, XL_s = None, None

    df_ref = collections.Counter()
    for p in cal_member_profiles + cal_non_member_profiles:
        for gram in _grams_from_lines(p["writer_lines"], 30):
            df_ref[gram] += 1
    synth_grams = set()
    for sp in synth_profiles:
        synth_grams |= _grams_from_lines(sp["writer_lines"], 30)
    frozen_rare_synth_grams = frozenset(g for g in synth_grams if 1 <= df_ref[g] <= 10)

    scorer = {
        "vec_stay": vec_stay,
        "X_s": X_s,
        "vec_ln": vec_ln,
        "XL_s": XL_s,
        "frozen_rare_synth_grams": frozen_rare_synth_grams,
        "thresholds": {},
    }
    if cal_non_member_profiles:
        non_scores = score_profiles_with_frozen_scorer(scorer, cal_non_member_profiles)
        for atk, arr in non_scores.items():
            scorer["thresholds"][atk] = {
                "thr_pos_1pct": _freeze_upper_1pct_thr(arr),
                "thr_neg_1pct": _freeze_lower_1pct_thr(arr),
                "cal_non_member_n": len(arr),
            }
    return scorer


def score_profiles_with_frozen_scorer(scorer, query_profiles):
    """Score arbitrary query profiles without mutating vectorizers, reference DF, or score scales."""
    stay_q = [p["stay_norm"] for p in query_profiles]
    X_q = scorer["vec_stay"].transform(stay_q)
    stay_cos, stay_ratio = _row_max_top2_cos(X_q, scorer["X_s"])

    line_cos = np.zeros(len(query_profiles), dtype=float)
    vec_ln, XL_s = scorer["vec_ln"], scorer["XL_s"]
    if vec_ln is not None and XL_s is not None:
        for qi, p in enumerate(query_profiles):
            ql = p["writer_lines"]
            if not ql:
                continue
            sims = (vec_ln.transform(ql) @ XL_s.T).toarray()
            line_cos[qi] = float(np.max(sims)) if sims.size else 0.0

    rare_set = scorer["frozen_rare_synth_grams"]
    rare_q_hits = np.array(
        [float(sum(1 for g in _grams_from_lines(p["writer_lines"], 30) if g in rare_set)) for p in query_profiles],
        dtype=float,
    )
    combined = stay_cos + 0.5 * line_cos + 0.25 * np.log1p(rare_q_hits)
    return {
        "combined": combined,
        "stay_1nn_cos": stay_cos,
        "stay_nn_ratio": stay_ratio,
        "line_1nn_cos": line_cos,
        "rare_30gram_attack": rare_q_hits,
    }


def _qi_k5_linkage(member_profiles, synth_profiles):
    qi_freq = collections.Counter(p["qi"] for p in member_profiles)
    rare_qis = {q for q, c in qi_freq.items() if 1 <= c < 5}
    synth_qis = {p["qi"] for p in synth_profiles}
    hits = len(rare_qis & synth_qis)
    return {
        "member_qi_lt5_total": len(rare_qis),
        "synth_qi_collisions": hits,
        "qi_collision_rate": round(hits / max(1, len(rare_qis)), 4),
    }


def _assign_tertile_labels(member_profiles, non_member_profiles):
    dx_freq = collections.Counter(p["dx4"] for p in member_profiles)
    uniq_freqs = sorted({dx_freq[p["dx4"]] for p in member_profiles})
    if len(uniq_freqs) < 3:
        return ["all"] * (len(member_profiles) + len(non_member_profiles))
    t1 = float(np.percentile([dx_freq[p["dx4"]] for p in member_profiles], 33.3))
    t2 = float(np.percentile([dx_freq[p["dx4"]] for p in member_profiles], 66.7))
    labels = []
    for p in member_profiles + non_member_profiles:
        c = dx_freq.get(p["dx4"], 0)
        if c <= t1:
            labels.append("tail")
        elif c <= t2:
            labels.append("medium")
        else:
            labels.append("common")
    return labels


def evaluate_mia_arm(name, synth_adms, member_adms, non_member_adms, n_boot=500, frozen_scorer=None, split_rehearsal=False):
    """Run S6-1 empirical attacks using either a locked Stage-1 scorer or fit-once scoring."""
    if not synth_adms or not member_adms or not non_member_adms:
        return {"arm": name, "status": "insufficient_data"}

    mem_prof = [_extract_profile(a) for a in member_adms]
    non_prof = [_extract_profile(a) for a in non_member_adms]
    syn_prof = [_extract_profile(a) for a in synth_adms]

    _, rare_stats = rare_span_counts(synth_adms, member_adms, grams=(20, 30), k=11)
    y_true = np.array([1] * len(mem_prof) + [0] * len(non_prof), dtype=int)

    scorer = frozen_scorer if frozen_scorer is not None else fit_frozen_arm_scorer(syn_prof, mem_prof, non_prof)
    scores_map = score_profiles_with_frozen_scorer(scorer, mem_prof + non_prof)
    combined = scores_map["combined"]

    tertiles = _assign_tertile_labels(mem_prof, non_prof)
    subgroup_auc = {}
    for tg in ("common", "medium", "tail"):
        idxs = [i for i, t in enumerate(tertiles) if t == tg]
        if idxs and len(np.unique(y_true[idxs])) >= 2:
            subgroup_auc[tg] = round(float(roc_auc_score(y_true[idxs], combined[idxs])), 4)

    nn_mia = {}
    frozen_thr_eval = {}
    for atk in ("combined", "stay_1nn_cos", "stay_nn_ratio", "line_1nn_cos", "rare_30gram_attack"):
        nn_mia[atk] = _boot_auc_ci(y_true, scores_map[atk], n_boot=n_boot)
        th_info = (scorer.get("thresholds") or {}).get(atk)
        if th_info is not None:
            frozen_thr_eval[atk] = eval_frozen_bidirectional_thresholds(
                y_true, scores_map[atk], th_info["thr_pos_1pct"], th_info["thr_neg_1pct"]
            )
    nn_mia["subgroup_combined_auc"] = subgroup_auc

    split_half_rep = None
    if split_rehearsal and len(mem_prof) >= 20 and len(non_prof) >= 20:
        rng_sp = np.random.RandomState(20261209)
        mi = rng_sp.permutation(len(mem_prof))
        ni = rng_sp.permutation(len(non_prof))
        hm, hn = len(mi) // 2, len(ni) // 2
        cal_m = [mem_prof[i] for i in mi[:hm]]
        ev_m = [mem_prof[i] for i in mi[hm:]]
        cal_n = [non_prof[i] for i in ni[:hn]]
        ev_n = [non_prof[i] for i in ni[hn:]]
        sp_scorer = fit_frozen_arm_scorer(syn_prof, cal_m, cal_n)
        sp_scores = score_profiles_with_frozen_scorer(sp_scorer, ev_m + ev_n)
        y_sp = np.array([1] * len(ev_m) + [0] * len(ev_n), dtype=int)
        split_half_rep = {}
        for atk in ("combined", "stay_1nn_cos", "stay_nn_ratio", "line_1nn_cos", "rare_30gram_attack"):
            th = sp_scorer["thresholds"][atk]
            split_half_rep[atk] = eval_frozen_bidirectional_thresholds(
                y_sp, sp_scores[atk], th["thr_pos_1pct"], th["thr_neg_1pct"]
            )

    rep = {
        "arm": name,
        "n_synth": len(synth_adms),
        "n_members": len(mem_prof),
        "n_non_members": len(non_prof),
        "scorer_frozen_mode": frozen_scorer is not None,
        "rare_30gram_mem": {
            "g30_candidates_with_rare": int(rare_stats.get("g30_candidates_with_rare", 0)),
            "g30_stay_hit_rate": round(int(rare_stats.get("g30_candidates_with_rare", 0)) / max(1, len(synth_adms)), 4),
            "g30_rare_writer_nodigit": int(rare_stats.get("g30_rare_writer_nodigit", 0)),
            "g30_rare_writer_digit": int(rare_stats.get("g30_rare_writer_digit", 0)),
        },
        "nn_mia": nn_mia,
        "frozen_threshold_eval": frozen_thr_eval,
        "qi_k5_linkage": _qi_k5_linkage(mem_prof, syn_prof),
    }
    if split_half_rep is not None:
        rep["stage1_frozen_split_rehearsal"] = split_half_rep
    return rep


def _iter_fold_records(fold_paths):
    for p in fold_paths:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def build_full_exported_artifact(synth_adms, fold_paths, out_dir):
    """Build real release candidate filtered against ALL 8,980 admissions in folds 2-7 and lock SHA256."""
    rare_map, rs_before = rare_span_counts(synth_adms, _iter_fold_records(fold_paths), grams=(30,), k=11)
    r30 = rare_map.get(30, {})
    kept = []
    dropped_rare = 0
    dropped_pattern = 0
    for a in synth_adms:
        aid = a.get("admission_id")
        if r30.get(aid, 0) > 0:
            dropped_rare += 1
            continue
        ph = pattern_hits(a)
        if any(ph.values()):
            dropped_pattern += 1
            continue
        kept.append(a)

    os.makedirs(out_dir, exist_ok=True)
    out_adm = os.path.join(out_dir, "admissions.jsonl")
    with open(out_adm, "w", encoding="utf-8") as f:
        for a in kept:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    h = hashlib.sha256()
    with open(out_adm, "rb") as f:
        while True:
            b = f.read(1 << 16)
            if not b:
                break
            h.update(b)
    sha = h.hexdigest()

    _, rs_after = rare_span_counts(kept, _iter_fold_records(fold_paths), grams=(30,), k=11)
    manifest = {
        "out_admissions": out_adm,
        "sha256": sha,
        "real_ref_admissions_scanned": int(rs_before.get("real_admissions_scanned", 0)),
        "input_n": len(synth_adms),
        "kept_n": len(kept),
        "dropped_rare30": dropped_rare,
        "dropped_pattern": dropped_pattern,
        "before_g30_candidates_with_rare": int(rs_before.get("g30_candidates_with_rare", 0)),
        "before_g30_rare_writer_nodigit": int(rs_before.get("g30_rare_writer_nodigit", 0)),
        "after_g30_candidates_with_rare": int(rs_after.get("g30_candidates_with_rare", 0)),
        "after_g30_rare_writer_nodigit": int(rs_after.get("g30_rare_writer_nodigit", 0)),
    }
    with open(os.path.join(out_dir, "export_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    return kept, manifest


def _sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 16)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def audit_export_composition(synth_full, exported_full, j_paths):
    """Compare pre-filter internal corpus (1,982) vs post-filter release artifact (901)."""
    jmap = {}
    for jp in j_paths or ():
        if jp and os.path.exists(jp):
            for j in _read_jsonl(jp):
                jmap[str(j["journey_id"])] = j

    def _stats(adms):
        los_v, doc_v, char_v = [], [], []
        surg = late_op = trans = 0
        for a in adms:
            aid = str(a.get("admission_id") or "")
            docs = a.get("docs") or []
            doc_v.append(len(docs))
            char_v.append(sum(len((d.get("text") or "").strip()) for d in docs))
            days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
            j = jmap.get(aid)
            if j is not None:
                los = int(j.get("los", max(days) if days else 0))
                procs = [x for x in (j.get("procedures") or ()) if x.get("kind") == "手术"]
                if procs:
                    surg += 1
                    if int(procs[0].get("decide_day", 0)) >= 2:
                        late_op += 1
                if any(int(e.get("day", 0)) >= 1 and e.get("state") in ("确诊", "排除") for e in (j.get("dx_events") or ())):
                    trans += 1
            else:
                los = max(days) if days else 0
            los_v.append(los)
        n = max(1, len(adms))
        return {
            "n_stays": len(adms),
            "mean_los": round(float(np.mean(los_v)), 3) if los_v else 0.0,
            "median_los": round(float(np.median(los_v)), 2) if los_v else 0.0,
            "los_ge3_rate": round(sum(1 for x in los_v if x >= 3) / n, 4),
            "mean_docs": round(float(np.mean(doc_v)), 3) if doc_v else 0.0,
            "mean_chars": round(float(np.mean(char_v)), 1) if char_v else 0.0,
            "surgical_rate": round(surg / n, 4) if jmap else None,
            "late_op_ge2_rate": round(late_op / n, 4) if jmap else None,
            "dx_transition_stay_rate": round(trans / n, 4) if jmap else None,
        }

    pre = _stats(synth_full)
    post = _stats(exported_full)
    deltas = {}
    for k in ("mean_los", "los_ge3_rate", "mean_docs", "mean_chars", "surgical_rate", "late_op_ge2_rate", "dx_transition_stay_rate"):
        if pre.get(k) is not None and post.get(k) is not None:
            deltas[k] = round(float(post[k]) - float(pre[k]), 4)
    return {
        "retention_rate": round(len(exported_full) / max(1, len(synth_full)), 4),
        "pre_filter_internal": pre,
        "post_filter_release": post,
        "delta_release_minus_internal": deltas,
    }


def _load_real_by_ids(fold_paths, id_file=None, max_n=600, seed=20261201):
    want = None
    if id_file and os.path.exists(id_file):
        want = {l.strip() for l in open(id_file, encoding="utf-8") if l.strip()}
    rows = []
    for p in fold_paths:
        if p and os.path.exists(p):
            for a in _read_jsonl(p):
                if want is None or str(a.get("admission_id")) in want:
                    rows.append(a)
    if max_n and len(rows) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows), size=max_n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("stage1_cal", "stage2_lock"))
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--data-dir", default="/home/weify/pyt/data")
    ap.add_argument("--non-member-fold", default=None)
    ap.add_argument("--direct32b-jsonl", default=None)
    ap.add_argument("--exported-jsonl", default=None, help="Existing locked Synth-Exported admissions.jsonl to audit")
    ap.add_argument("--build-export-out", default=None, help="Scan all folds 2-7 (8980) to build & lock Synth-Exported")
    ap.add_argument("--freeze-cal-dir", default=None, help="Serialize frozen Stage-1 MIA scorers & bidirectional 1% FPR thresholds")
    ap.add_argument("--frozen-cal-dir", default=None, help="Load frozen Stage-1 MIA scorers & thresholds for stage2_lock")
    ap.add_argument("--max-eval", type=int, default=600)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    non_mem_path = args.non_member_fold or os.path.join(
        args.data_dir, "deid_run9_fold8.jsonl" if args.mode == "stage1_cal" else "deid_run9_fold9.jsonl"
    )
    fold0_path = os.path.join(args.data_dir, "deid_run9_fold0.jsonl")
    folds_27 = [os.path.join(args.data_dir, f"deid_run9_fold{k}.jsonl") for k in range(2, 8)]

    fa, fj = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400)
    d32 = _resolve_direct32b_paths(args.s5_root, args.direct32b_jsonl)
    rb = [os.path.join(args.s5_root, "generated", "s5_2", "route_b", "admissions.jsonl")]
    lora_e200 = "/home/weify/pyt/product/lora_trial/s3_f0_r64/runs/e200/admissions.jsonl"
    lora_a256 = "/home/weify/pyt/product/lora_trial/s3_f0_r64/runs/a256/admissions.jsonl"

    assert_no_fold9([non_mem_path, fold0_path, *folds_27, *fa, *d32, *rb, args.out, args.exported_jsonl or ""], args.mode)

    frozen_arms_in = {}
    frozen_cal_sha = None
    if args.mode == "stage2_lock" and args.frozen_cal_dir:
        man_p = os.path.join(args.frozen_cal_dir, "freeze_manifest.json")
        pkl_p = os.path.join(args.frozen_cal_dir, "mia_frozen_cal.pkl")
        if not (os.path.exists(man_p) and os.path.exists(pkl_p)):
            raise FileNotFoundError(f"Missing frozen MIA bundle or manifest in {args.frozen_cal_dir}")
        with open(man_p, encoding="utf-8") as mf:
            fman = json.load(mf)
        frozen_cal_sha = _sha256_path(pkl_p)
        if frozen_cal_sha != fman["bundle_sha256"]:
            raise RuntimeError(f"FROZEN_MIA_HASH_MISMATCH: expected {fman['bundle_sha256']}, got {frozen_cal_sha}")
        with open(pkl_p, "rb") as pf:
            fobj = pickle.load(pf)
        frozen_arms_in = fobj["arms"]

    non_mem_pool = _read_jsonl(non_mem_path)
    non_mem_prof = [_extract_profile(a) for a in non_mem_pool]
    non_mem_by_id = {str(a.get("admission_id")): a for a in non_mem_pool}

    def _matched_real(mem_adms, seed=20261201):
        mp = [_extract_profile(a) for a in mem_adms]
        picked_p, diag = _match_non_members(mp, non_mem_prof, seed=seed)
        return [non_mem_by_id[p["aid"]] for p in picked_p if p["aid"] in non_mem_by_id], diag

    arm_reports = {}
    frozen_arms_out = {}

    def _eval_and_freeze(arm_key, s_adms, m_adms, nm_adms, diag_obj, role=None, extra_kv=None):
        f_scorer = frozen_arms_in.get(arm_key)
        if f_scorer is None and arm_key == "Synth-Exported-LockedArtifact":
            f_scorer = frozen_arms_in.get("Synth-Exported-FullRef8980")
        if f_scorer is None and args.mode == "stage1_cal":
            sp = [_extract_profile(a) for a in s_adms]
            mp = [_extract_profile(a) for a in m_adms]
            nmp = [_extract_profile(a) for a in nm_adms]
            f_scorer = fit_frozen_arm_scorer(sp, mp, nmp)
            frozen_arms_out[arm_key] = f_scorer
        rep = evaluate_mia_arm(
            arm_key, s_adms, m_adms, nm_adms,
            n_boot=args.n_boot,
            frozen_scorer=f_scorer,
            split_rehearsal=(args.mode == "stage1_cal"),
        )
        rep["match_diagnostics"] = diag_obj
        if role:
            rep["role"] = role
        if extra_kv:
            rep.update(extra_kv)
        arm_reports[arm_key] = rep

    if os.path.exists(lora_e200):
        mem_e200 = _load_real_by_ids([fold0_path], "/home/weify/pyt/product/s3_r2/work/e200_ids.txt", max_n=200)
        nm_e200, diag_e200 = _matched_real(mem_e200, seed=20261202)
        _eval_and_freeze("PosCtrl-LoRA-r64-e200", _read_jsonl(lora_e200), mem_e200, nm_e200, diag_e200)

    if os.path.exists(lora_a256):
        mem_a256 = _load_real_by_ids(
            [fold0_path], "/home/weify/pyt/product/lora_trial/s3_f0_v02/pairs/train_pool_ids.txt", max_n=args.max_eval
        )
        nm_a256, diag_a256 = _matched_real(mem_a256, seed=20261203)
        _eval_and_freeze("PosCtrl-LoRA-r64-a256", _read_jsonl(lora_a256), mem_a256, nm_a256, diag_a256)

    mem_27 = _load_real_by_ids(folds_27, max_n=args.max_eval, seed=20261204)
    nm_27, diag_27 = _matched_real(mem_27, seed=20261205)

    export_manifest = None
    export_comp = None
    if fa:
        synth_full = [a for p in fa for a in _read_jsonl(p)]
        _eval_and_freeze("Synth-Full", synth_full, mem_27, nm_27, diag_27)

        if args.exported_jsonl and os.path.exists(args.exported_jsonl):
            exported_full = _read_jsonl(args.exported_jsonl)
            exp_name = "Synth-Exported-LockedArtifact"
        elif args.build_export_out:
            exported_full, export_manifest = build_full_exported_artifact(synth_full, folds_27, args.build_export_out)
            exp_name = "Synth-Exported-FullRef8980"
        else:
            exported_full = None
        if exported_full is not None:
            export_comp = audit_export_composition(synth_full, exported_full, fj)
            extra = {"export_composition_shift": export_comp}
            if export_manifest:
                extra["export_manifest"] = export_manifest
            _eval_and_freeze(exp_name, exported_full, mem_27, nm_27, diag_27, extra_kv=extra)

    if os.path.exists(rb[0]):
        _eval_and_freeze("Route-B-ZeroHosp", _read_jsonl(rb[0]), mem_27, nm_27, diag_27, role="pseudo_member_negative_control")

    if d32:
        synth_d32 = [a for p in d32 for a in _read_jsonl(p)]
        _eval_and_freeze("DIRECT-32B", synth_d32, mem_27, nm_27, diag_27)

    freeze_info = None
    if args.mode == "stage1_cal" and args.freeze_cal_dir and frozen_arms_out:
        os.makedirs(args.freeze_cal_dir, exist_ok=True)
        pkl_p = os.path.join(args.freeze_cal_dir, "mia_frozen_cal.pkl")
        with open(pkl_p, "wb") as pf:
            pickle.dump({"version": "v1.6", "non_member_fold": os.path.basename(non_mem_path), "arms": frozen_arms_out}, pf)
        fsha = _sha256_path(pkl_p)
        man_obj = {
            "version": "v1.6",
            "bundle_sha256": fsha,
            "non_member_fold": os.path.basename(non_mem_path),
            "arm_keys": sorted(frozen_arms_out.keys()),
            "thresholds_summary": {
                k: {
                    atk: {mk: round(float(mv), 6) if isinstance(mv, float) else mv for mk, mv in tv.items()}
                    for atk, tv in v["thresholds"].items()
                }
                for k, v in frozen_arms_out.items()
            },
        }
        man_p = os.path.join(args.freeze_cal_dir, "freeze_manifest.json")
        with open(man_p, "w", encoding="utf-8") as mf:
            json.dump(man_obj, mf, ensure_ascii=False, indent=2)
        freeze_info = {"dir": args.freeze_cal_dir, "bundle_sha256": fsha, "arms": sorted(frozen_arms_out.keys())}

    attack_metrics = ("combined", "stay_1nn_cos", "stay_nn_ratio", "line_1nn_cos", "rare_30gram_attack")
    per_attack_valid = {}
    best_pos_auc = 0.0
    for mk in attack_metrics:
        per_arm = {}
        mx = 0.0
        for pk in ("PosCtrl-LoRA-r64-e200", "PosCtrl-LoRA-r64-a256"):
            v = (((arm_reports.get(pk) or {}).get("nn_mia") or {}).get(mk) or {}).get("roc_auc")
            if v is not None:
                per_arm[pk] = v
                mx = max(mx, float(v))
        best_pos_auc = max(best_pos_auc, mx)
        per_attack_valid[mk] = {
            **per_arm,
            "best_auc": round(mx, 4),
            "informative_ge_070": mx >= 0.70,
        }

    summary = {
        "mode": args.mode,
        "version": "v1.6",
        "non_member_fold": os.path.basename(non_mem_path),
        "n_boot": args.n_boot,
        "frozen_cal_sha256": frozen_cal_sha or (freeze_info["bundle_sha256"] if freeze_info else None),
        "pos_ctrl_best_auc": round(best_pos_auc, 4),
        "pos_ctrl_valid": best_pos_auc >= 0.70,
        "pos_ctrl_per_attack_validity": per_attack_valid,
        "export_composition_shift": export_comp,
        "arms": arm_reports,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        "S6_MIA_DONE",
        json.dumps(
            {
                "mode": args.mode,
                "pos_ctrl_valid": summary["pos_ctrl_valid"],
                "pos_ctrl_best_auc": summary["pos_ctrl_best_auc"],
                "informative_attacks": [k for k, v in per_attack_valid.items() if v["informative_ge_070"]],
                "frozen_cal_sha256": summary["frozen_cal_sha256"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
