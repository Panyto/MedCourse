# -*- coding: utf-8 -*-
"""S5-2 17-pathway unified text-feature and within-pathway early-order predictability probe."""
import argparse
import collections
import json
import os
import re

from s5_common import PATHWAY_SPECS, setup_paths
from plan_s5_corpora import EXAM_PATTERNS
from t123_bench import _in_day_window, _is_discharge, _is_hr, has_early_op_plan, has_late_op_event

setup_paths()

from medcourse.rules import setup  # noqa: E402

setup(("BENCH_G3", "BENCH_G2", "BENCH_B7", "BENCH_G45"))
from q45_kit.realtext import care7_from_admission  # noqa: E402
from ruler_g1 import extract_facts, split_sections  # noqa: E402

import numpy as np  # noqa: E402
try:
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    from sklearn.metrics import roc_auc_score  # noqa: E402
    from sklearn.model_selection import StratifiedKFold  # noqa: E402
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


_PAST_CLAUSE_RE = re.compile(r"既往|\d+\s*年前|\d+\s*月前|外院|曾|否认")
_SYMPTOM_PATS = (
    re.compile(r"发热|寒战|体温升高"),
    re.compile(r"腹痛|上腹痛|右下腹痛|腹胀"),
    re.compile(r"胸痛|胸闷|气促|呼吸困难"),
    re.compile(r"白细胞升高|CRP升高|淀粉酶升高|肌钙蛋白升高"),
)
_SURG_DEPT_RE = re.compile(r"外科|骨科|妇科|产科|泌尿|肝胆|胃肠|胸外|神经外")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _age_sex_from_adm(adm, rec=None):
    txt = "\n".join(d.get("text") or "" for d in (adm.get("docs") or [])[:2])
    age_s = str((rec or {}).get("Age") or "")
    if not age_s:
        m = re.search(r"(\d{1,3})\s*(岁|月)", txt[:500])
        age_s = m.group(0) if m else "50岁"
    m_num = re.search(r"(\d+)", age_s)
    age_v = float(m_num.group(1)) if m_num else 50.0
    if "月" in age_s and "岁" not in age_s:
        age_v = age_v / 12.0
    sex_s = (rec or {}).get("Gender") or ""
    if not sex_s:
        sm = re.search(r"性别[:：]\s*(男|女)|(男|女)性", txt[:500])
        sex_s = (sm.group(1) or sm.group(2)) if sm else "男"
    return age_v, (1.0 if sex_s == "男" else 0.0)


def match_real_pathway(adm, rec=None):
    """Map real admission strictly to one of the 17 kb_v22 pathways honoring age/acuity constraints."""
    pdx = (rec or {}).get("Primary Diagnosis") or ""
    if not pdx:
        hr = next((d for d in adm.get("docs") or () if _is_hr(d)), None)
        if hr:
            secs = {s.name: s.text for s in split_sections(hr["text"])}
            lines = [l.strip() for l in secs.get("入院诊断", "").splitlines() if l.strip()]
            pdx = re.sub(r"^\d+[\.、\s]*", "", lines[0]) if lines else ""
    age_v, _ = _age_sex_from_adm(adm, rec)
    if not pdx:
        return None
    if "结肠" in pdx and any(w in pdx for w in ("化疗", "术后", "恶性肿瘤", "癌")):
        return "C18_chemo"
    if "糖尿病" in pdx:
        return "E11"
    if "心绞痛" in pdx:
        return "I20.0"
    if "心肌梗死" in pdx or "STEMI" in pdx.upper():
        return "I21"
    if "冠状动脉粥样硬化" in pdx or "冠心病" in pdx:
        return "I25"
    if "心力衰竭" in pdx or "心衰" in pdx:
        return "I50"
    if "脑梗死" in pdx or "脑梗塞" in pdx:
        return "I63"
    if "痔" in pdx:
        return "I84"
    if "毛细支气管炎" in pdx and age_v < 14:
        return "J21"
    if "支气管肺炎" in pdx and age_v < 14:
        return "J18.0"
    if "肺炎" in pdx and age_v >= 14:
        return "J15"
    if "肠梗阻" in pdx:
        return "K56"
    if "胆囊" in pdx:
        return "K80.1_chronic" if "慢性" in pdx else "K80.0_acute"
    if "胰腺炎" in pdx:
        return "K85"
    if any(w in pdx for w in ("终末期肾病", "慢性肾脏病", "尿毒症", "慢性肾衰竭")):
        return "N18"
    if "肾结石" in pdx or "输尿管结石" in pdx:
        return "N20"
    return None


def _nonpast_exam_flags(d01_text):
    clauses = [
        cl for cl in re.split(r"[，,。；;！？!?\n]", d01_text or "")
        if cl and not _PAST_CLAUSE_RE.search(cl)
    ]
    joined = "\n".join(clauses)
    return [1.0 if pat.search(joined) else 0.0 for _, pat in EXAM_PATTERNS]


def extract_unified_cohort(adms, jmap=None, is_real=False, gt2_only=False):
    """Build strictly identical text-derived feature rows across synthetic and real cohorts."""
    pw_idx = {row[0]: i for i, row in enumerate(PATHWAY_SPECS)}
    rows = []
    for a in adms:
        docs = a.get("docs") or []
        days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
        if (max(days) if days else 0) < 3:
            continue
        rec = None
        if is_real:
            rec, _ = care7_from_admission(a, split_sections, extract_facts)
            kb_key = match_real_pathway(a, rec)
        else:
            j = (jmap or {}).get(a.get("admission_id"))
            kb_key = (j or {}).get("kb_pathway_key")
        if kb_key not in pw_idx:
            continue

        d01 = "\n".join(d["text"] for d in docs if _in_day_window(d, 0, 1) and not _is_discharge(d))
        if not d01 or has_early_op_plan(d01):
            continue
        if gt2_only:
            d2 = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, 2) and not _is_discharge(d))
            if has_early_op_plan(d2) or has_late_op_event(d2):
                continue
        start_d = 3 if gt2_only else 2
        late_txt = "\n".join(d["text"] for d in docs if _in_day_window(d, start_d, None) and not _is_discharge(d))
        y = 1 if has_late_op_event(late_txt) else 0

        age_v, sex_v = _age_sex_from_adm(a, rec)
        pw_oh = [0.0] * len(PATHWAY_SPECS)
        pw_oh[pw_idx[kb_key]] = 1.0
        symp = [1.0 if pat.search(d01) else 0.0 for pat in _SYMPTOM_PATS]
        exams = _nonpast_exam_flags(d01)
        rows.append({
            "kb_key": kb_key,
            "y": y,
            "x_base": [age_v / 100.0, sex_v, *symp, *pw_oh],
            "x_within": [age_v / 100.0, sex_v, *symp],
            "x_exams": exams,
        })
    return rows


def _cv_auc(X, y, seed=20261220):
    if not HAS_SKLEARN or len(y) == 0:
        return None
    pos, neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if pos < 2 or neg < 2 or X.shape[1] == 0:
        return None
    skf = StratifiedKFold(n_splits=min(5, pos, neg), shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    return round(float(roc_auc_score(y, probs)), 4)


def evaluate_unified_probe(name, rows, common_pw_weights=None):
    """Evaluate pooled and strict within-pathway incremental AUROC on unified text features."""
    if not rows:
        return {"cohort": name, "eligible_n": 0}
    y = np.array([r["y"] for r in rows], dtype=int)
    X_base = np.array([r["x_base"] for r in rows], dtype=float)
    X_ex = np.array([r["x_exams"] for r in rows], dtype=float)
    X_both = np.hstack([X_base, X_ex])

    auc_base = _cv_auc(X_base, y)
    auc_ex = _cv_auc(X_ex, y)
    auc_both = _cv_auc(X_both, y)
    pooled_delta = round(auc_both - auc_base, 4) if (auc_both is not None and auc_base is not None) else None

    by_pw = collections.defaultdict(list)
    for r in rows:
        by_pw[r["kb_key"]].append(r)

    per_pw = {}
    w_base_num = w_both_num = w_den = 0.0
    common_num = common_den = 0.0
    estimable = 0
    for row in PATHWAY_SPECS:
        k = row[0]
        sub = by_pw.get(k, [])
        ys = np.array([r["y"] for r in sub], dtype=int)
        pos, neg = int(np.sum(ys == 1)), int(np.sum(ys == 0))
        if pos < 2 or neg < 2:
            per_pw[k] = {"n": len(sub), "pos": pos, "estimable": False}
            continue
        estimable += 1
        Xw = np.array([r["x_within"] for r in sub], dtype=float)
        Xe = np.array([r["x_exams"] for r in sub], dtype=float)
        ab = _cv_auc(Xw, ys)
        ax = _cv_auc(np.hstack([Xw, Xe]), ys)
        delta_w = round(ax - ab, 4) if (ab is not None and ax is not None) else None
        per_pw[k] = {"n": len(sub), "pos": pos, "estimable": True, "auc_base": ab, "auc_with_exams": ax, "delta": delta_w}
        if ab is not None and ax is not None:
            pairs = float(pos * neg)
            w_base_num += pairs * ab
            w_both_num += pairs * ax
            w_den += pairs
            if common_pw_weights and k in common_pw_weights:
                cw = float(common_pw_weights[k])
                common_num += cw * ax
                common_den += cw

    w_auc_base = round(w_base_num / w_den, 4) if w_den > 0 else None
    w_auc_both = round(w_both_num / w_den, 4) if w_den > 0 else None
    return {
        "cohort": name,
        "eligible_n": len(rows),
        "late_surg_pos": int(np.sum(y)),
        "pos_rate": round(float(np.mean(y)), 4),
        "auc_disease_demo_only": auc_base,
        "auc_d01_exams_text_only": auc_ex,
        "auc_disease_demo_plus_exams": auc_both,
        "pooled_delta_auc": pooled_delta,
        "n_estimable_pathways": estimable,
        "within_pathway_weighted_base_auc": w_auc_base,
        "within_pathway_weighted_exams_auc": w_auc_both,
        "within_pathway_delta_auc": round(w_auc_both - w_auc_base, 4) if (w_auc_base is not None and w_auc_both is not None) else None,
        "common_weight_within_auc": round(common_num / common_den, 4) if common_den > 0 else None,
        "per_pathway": per_pw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--real-fold", default="/home/weify/pyt/data/deid_run9_fold1.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    real_adms = _read_jsonl(args.real_fold) if (args.real_fold and os.path.exists(args.real_fold)) else []
    s5_2_data = {}
    for arm in ("route_b", "route_a_only", "route_ab"):
        sdir = os.path.join(args.s5_root, "generated", "s5_2", arm)
        ap_p = os.path.join(sdir, "admissions.jsonl")
        jp_p = os.path.join(sdir, "journeys.jsonl")
        if os.path.exists(ap_p) and os.path.exists(jp_p):
            s5_2_data[arm] = (_read_jsonl(ap_p), {j["journey_id"]: j for j in _read_jsonl(jp_p)})

    out = {"real_fold": os.path.basename(args.real_fold), "cutoffs": {}}
    for cutoff_lbl, gt2_flag in (("ge2", False), ("gt2", True)):
        real_rows = extract_unified_cohort(real_adms, is_real=True, gt2_only=gt2_flag) if real_adms else []
        by_pw_r = collections.Counter(r["kb_key"] for r in real_rows)
        common_weights = {
            k: sum(1 for r in real_rows if r["kb_key"] == k and r["y"] == 1) * sum(1 for r in real_rows if r["kb_key"] == k and r["y"] == 0)
            for k in by_pw_r
        }
        sec = {}
        if real_rows:
            sec["Real-17d"] = evaluate_unified_probe(f"Real-17d ({cutoff_lbl})", real_rows, common_weights)
        for label, arm in (("Route-B", "route_b"), ("Route-A-Only", "route_a_only"), ("Route-AB", "route_ab")):
            if arm in s5_2_data:
                adms, jmap = s5_2_data[arm]
                rows = extract_unified_cohort(adms, jmap=jmap, is_real=False, gt2_only=gt2_flag)
                sec[label] = evaluate_unified_probe(f"{label} ({cutoff_lbl})", rows, common_weights)
        out["cutoffs"][cutoff_lbl] = sec

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("S5_2_17D_PROBE_DONE", json.dumps({"out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
