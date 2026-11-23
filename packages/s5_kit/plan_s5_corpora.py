# -*- coding: utf-8 -*-
"""Stage-1 CPU trajectory pre-planning and Reviewer Point-2 early-exam audit."""
import argparse
import collections
import csv
import json
import os
import random
import re

from s5_common import PATHWAY_SPECS, build_pathway_stats, setup_paths

setup_paths()

from medcourse import kb, planner  # noqa: E402
from medcourse.rules import setup  # noqa: E402


setup(("BENCH_G3", "BENCH_G2", "BENCH_B7", "BENCH_G45"))
from q45_kit.realtext import care7_from_admission  # noqa: E402
from ruler_g1 import extract_facts, split_sections  # noqa: E402

import numpy as np  # noqa: E402
try:
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    from sklearn.metrics import mutual_info_score, roc_auc_score  # noqa: E402
    from sklearn.model_selection import StratifiedKFold  # noqa: E402
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


EARLY_OP_RE = re.compile(r"今日拟行|拟于.{0,8}行.{0,12}术|术前小结|术后首次病程|在.{0,8}麻醉下行")
LATE_OP_RE = re.compile(
    r"术后首次病程|今日在.{0,8}麻醉下行|在.{0,10}麻醉下行.{0,18}(?:术|切除|修补|造影|置管|穿刺|清创|引流)"
)

EXAM_PATTERNS = [
    ("血常规", re.compile(r"血常规")),
    ("尿常规", re.compile(r"尿常规")),
    ("粪常规", re.compile(r"粪常规|大便常规|便常规")),
    ("C反应蛋白", re.compile(r"C反应蛋白|CRP")),
    ("凝血", re.compile(r"凝血")),
    ("肝功能", re.compile(r"肝功能|肝肾功")),
    ("肾功能", re.compile(r"肾功能|肌酐|尿素氮")),
    ("电解质", re.compile(r"电解质")),
    ("血糖", re.compile(r"血糖|糖化血红蛋白")),
    ("血脂", re.compile(r"血脂|甘油三酯|胆固醇")),
    ("心电图", re.compile(r"心电图|动态心电图")),
    ("CT", re.compile(r"CT")),
    ("X线", re.compile(r"X线|胸片|平片")),
    ("超声", re.compile(r"超声|彩超|B超")),
    ("心脏超声", re.compile(r"心脏超声|心动图")),
    ("磁共振", re.compile(r"磁共振|MRI|MRCP")),
    ("肌钙蛋白", re.compile(r"肌钙蛋白|TNI")),
    ("利钠肽", re.compile(r"利钠肽|BNP")),
    ("D二聚体", re.compile(r"D[-—]?二聚体")),
    ("淀粉酶", re.compile(r"淀粉酶")),
    ("胃肠镜", re.compile(r"胃镜|肠镜")),
    ("冠脉造影", re.compile(r"冠脉造影|冠状动脉造影")),
]

MATCH_17_KWS = (
    "结肠", "糖尿病", "心绞痛", "心肌梗死", "冠状动脉粥样硬化", "冠心病",
    "心力衰竭", "脑梗死", "痔", "肺炎", "毛细支气管炎", "肠梗阻",
    "胆囊", "胰腺炎", "肾脏病", "尿毒症", "肾衰竭", "肾结石",
)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def plan_s5_1_shards(out_dir, stats_ab, seeds=(20261210, 20261211, 20261212, 20261213), per_shard=500):
    summary = {}
    all_records = []
    for s_idx, seed in enumerate(seeds):
        js_rows, sp_rows, dr_rows = [], [], []
        idx = 0
        attempts = 0
        while len(js_rows) < per_shard:
            jid = f"MC{seed}-{idx:05d}"
            rng = random.Random(planner.seed_of(seed, jid))
            j = planner.plan_journey(stats_ab, rng, jid, None)
            attempts += 1
            idx += 1
            if j is None:
                continue
            specs = planner.schedule(j, stats_ab, rng)
            errs = planner.check_plan(j, specs)
            if errs:
                raise RuntimeError(f"S5-1 check_plan failed on {jid}: {errs}")
            js_rows.append(j)
            sp_rows.append({"slot_id": jid, "specs": specs})
            dr_rows.append({"slot_id": jid, "draft": planner.journey_to_draft(j)})
        sdir = os.path.join(out_dir, "s5_1_shards", f"shard_{s_idx}")
        _write_jsonl(os.path.join(sdir, "journeys.jsonl"), js_rows)
        _write_jsonl(os.path.join(sdir, "specs.jsonl"), sp_rows)
        _write_jsonl(os.path.join(sdir, "drafts.jsonl"), dr_rows)
        all_records.extend(js_rows)
        summary[f"shard_{s_idx}"] = {
            "seed": seed, "planned": len(js_rows), "attempts": attempts,
            "los_ge3": sum(1 for j in js_rows if j["los"] >= 3),
            "surg_dec_ge2": sum(
                1 for j in js_rows if j["los"] >= 3 and any(
                    p.get("kind") == "手术" and p["decide_day"] >= 2 for p in (j.get("procedures") or ())
                )
            ),
            "surg_dec_gt2": sum(
                1 for j in js_rows if j["los"] >= 3 and any(
                    p.get("kind") == "手术" and p["decide_day"] > 2 for p in (j.get("procedures") or ())
                )
            ),
        }
    return summary, all_records


def plan_s5_2_arms(out_dir, stats_a, kb_data, seed=20261220, n_per_pathway=20):
    summary = {}
    arm_records = {}
    for arm in ("route_b", "route_a_only", "route_ab"):
        js_rows, sp_rows, dr_rows = [], [], []
        by_pw = {}
        for row in PATHWAY_SPECS:
            kb_key = row[0]
            pstats = build_pathway_stats(arm, kb_key, stats_a=stats_a, kb_data=kb_data)
            cnt = 0
            idx = 0
            attempts = 0
            while cnt < n_per_pathway and attempts < 300:
                tag = kb_key.replace(".", "_")
                jid = f"S52-{arm}-{tag}-{idx:03d}"
                rng = random.Random(planner.seed_of(seed, jid))
                j = planner.plan_journey(pstats, rng, jid, None)
                attempts += 1
                idx += 1
                if j is None:
                    continue
                specs = planner.schedule(j, pstats, rng)
                errs = planner.check_plan(j, specs)
                if errs:
                    continue
                j["kb_pathway_key"] = kb_key
                j["s5_arm"] = arm
                js_rows.append(j)
                sp_rows.append({"slot_id": jid, "specs": specs})
                dr_rows.append({"slot_id": jid, "draft": planner.journey_to_draft(j)})
                cnt += 1
            if cnt < n_per_pathway:
                raise RuntimeError(f"Failed to plan {n_per_pathway} journeys for arm={arm} pathway={kb_key}: got {cnt}")
            by_pw[kb_key] = {"kept": cnt, "attempts": attempts}
        adir = os.path.join(out_dir, "s5_2_17d", arm)
        _write_jsonl(os.path.join(adir, "journeys.jsonl"), js_rows)
        _write_jsonl(os.path.join(adir, "specs.jsonl"), sp_rows)
        _write_jsonl(os.path.join(adir, "drafts.jsonl"), dr_rows)
        arm_records[arm] = js_rows
        diag_hits = sum(1 for j in js_rows if any(e.get("diagnostic") for e in j.get("exams") or ()))
        surg_cnt = sum(1 for j in js_rows if j.get("surgical"))
        surg_diag = sum(1 for j in js_rows if j.get("surgical") and any(e.get("diagnostic") for e in j.get("exams") or ()))
        summary[arm] = {
            "total": len(js_rows), "surgical": surg_cnt,
            "exam_diagnostic_rate": round(diag_hits / max(1, len(js_rows)), 4),
            "surgical_diag_rate": round(surg_diag / max(1, surg_cnt), 4),
            "by_pathway": by_pw,
        }
    return summary, arm_records


def _cv_auc(X, y, seed=20261220):
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos < 5 or neg < 5 or X.shape[1] == 0:
        return None
    n_splits = min(5, pos, neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=seed)
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    return round(float(roc_auc_score(y, probs)), 4)


def _synth_exam_predictability(name, journeys):
    elig = []
    for j in journeys:
        if j["los"] < 3:
            continue
        sp = [x for x in (j.get("procedures") or ()) if x.get("kind") == "手术"]
        if sp and sp[0]["decide_day"] <= 1:
            continue
        elig.append(j)
    keys = sorted({e["key"] for j in elig for e in (j.get("exams") or ()) if e["order_day"] <= 1})
    y = np.array([
        1 if any(p.get("kind") == "手术" and p["do_day"] >= 2 for p in (j.get("procedures") or ())) else 0
        for j in elig
    ], dtype=int)
    X_ex = np.zeros((len(elig), len(keys)), dtype=float)
    X_dm = np.zeros((len(elig), 2), dtype=float)
    for r_i, j in enumerate(elig):
        ordered = {e["key"] for e in (j.get("exams") or ()) if e["order_day"] <= 1}
        for c_i, k in enumerate(keys):
            X_ex[r_i, c_i] = 1.0 if k in ordered else 0.0
        X_dm[r_i, 0] = float(j["patient"]["age"]) / 100.0
        X_dm[r_i, 1] = 1.0 if j["patient"]["sex"] == "男" else 0.0

    mi_scores = sorted(
        ((k, round(float(mutual_info_score(X_ex[:, c_i], y)), 5)) for c_i, k in enumerate(keys)),
        key=lambda x: -x[1],
    )
    auc_ex = _cv_auc(X_ex, y)
    auc_dm = _cv_auc(X_dm, y)
    auc_both = _cv_auc(np.hstack([X_dm, X_ex]), y) if len(keys) else auc_dm
    return {
        "cohort": name,
        "total": len(journeys),
        "eligible_los3_no_early": len(elig),
        "late_surg_pos": int(np.sum(y)),
        "pos_rate": round(float(np.mean(y)), 4) if len(y) else 0.0,
        "n_exam_features": len(keys),
        "auc_demo_only": auc_dm,
        "auc_d01_exams_only": auc_ex,
        "auc_demo_plus_exams": auc_both,
        "top_exam_mi": mi_scores[:8],
    }


def _is_ds(doc):
    return doc.get("doc_type") == "discharge_summary" or doc.get("role") == "DS"


def _real_exam_predictability(name, fold_path, match_17=False):
    adms = [json.loads(l) for l in open(fold_path, encoding="utf-8") if l.strip()]
    missing_day_offset_docs = sum(1 for a in adms for d in a["docs"] if d.get("day_offset") is None)
    elig = []
    for a in adms:
        days = [int(d["day_offset"]) for d in a["docs"] if d.get("day_offset") is not None]
        if (max(days) if days else 0) < 3:
            continue
        rec, _ = care7_from_admission(a, split_sections, extract_facts)
        if match_17:
            pdx = (rec or {}).get("Primary Diagnosis") or ""
            if not any(w in pdx for w in MATCH_17_KWS):
                continue
        d01 = "\n".join(
            d["text"]
            for d in a["docs"]
            if d.get("day_offset") is not None and 0 <= int(d["day_offset"]) <= 1 and not _is_ds(d)
        )
        if EARLY_OP_RE.search(d01):
            continue
        d2p = "\n".join(
            d["text"]
            for d in a["docs"]
            if d.get("day_offset") is not None and int(d["day_offset"]) >= 2 and not _is_ds(d)
        )
        y_val = 1 if LATE_OP_RE.search(d2p) else 0
        age_s = str((rec or {}).get("Age") or "50")
        m_age = re.search(r"(\d+)", age_s)
        age_v = float(m_age.group(1)) if m_age else 50.0
        if "月" in age_s and "岁" not in age_s:
            age_v = age_v / 12.0
        sex_v = 1.0 if ((rec or {}).get("Gender") == "男") else 0.0
        feats = [1.0 if pat.search(d01) else 0.0 for _, pat in EXAM_PATTERNS]
        elig.append((y_val, age_v / 100.0, sex_v, feats))
    y = np.array([r[0] for r in elig], dtype=int)
    X_dm = np.array([[r[1], r[2]] for r in elig], dtype=float)
    X_ex = np.array([r[3] for r in elig], dtype=float)
    mi_scores = sorted(
        ((EXAM_PATTERNS[i][0], round(float(mutual_info_score(X_ex[:, i], y)), 5)) for i in range(len(EXAM_PATTERNS))),
        key=lambda x: -x[1],
    )
    return {
        "cohort": name,
        "total": len(adms),
        "missing_day_offset_docs": missing_day_offset_docs,
        "eligible_los3_no_early": len(elig),
        "late_surg_pos": int(np.sum(y)),
        "pos_rate": round(float(np.mean(y)), 4) if len(y) else 0.0,
        "n_exam_features": len(EXAM_PATTERNS),
        "auc_demo_only": _cv_auc(X_dm, y),
        "auc_d01_exams_only": _cv_auc(X_ex, y),
        "auc_demo_plus_exams": _cv_auc(np.hstack([X_dm, X_ex]), y),
        "top_exam_mi": mi_scores[:8],
    }


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="/home/weify/pyt/product/derived/m2_f27_v021/stats_v021.json")
    ap.add_argument("--fold1", default="/home/weify/pyt/data/deid_run9_fold1.jsonl")
    ap.add_argument("--out-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--audit-only", action="store_true", help="Read existing planned journeys without touching planned/")
    args = ap.parse_args()

    planned_dir = os.path.join(args.out_root, "planned")
    tables_dir = os.path.join(args.out_root, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    if args.audit_only:
        s5_1_js = []
        for s_idx in range(4):
            s5_1_js.extend(_read_jsonl(os.path.join(planned_dir, "s5_1_shards", f"shard_{s_idx}", "journeys.jsonl")))
        s5_2_arms = {
            arm: _read_jsonl(os.path.join(planned_dir, "s5_2_17d", arm, "journeys.jsonl"))
            for arm in ("route_b", "route_a_only", "route_ab")
        }
        s5_1_summ = None
    else:
        with open(args.stats, encoding="utf-8") as f:
            stats_a = json.load(f)
        kb_data = kb.load_default_kb()
        stats_ab = dict(stats_a, _kb=kb_data)
        os.makedirs(planned_dir, exist_ok=True)
        s5_1_summ, s5_1_js = plan_s5_1_shards(planned_dir, stats_ab)
        s5_2_summ, s5_2_arms = plan_s5_2_arms(planned_dir, stats_a, kb_data)
        with open(os.path.join(tables_dir, "plan_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"s5_1": s5_1_summ, "s5_2": s5_2_summ}, f, ensure_ascii=False, indent=2)

    pred_rows = [
        _synth_exam_predictability("S5-2 Route B (kb-only 340)", s5_2_arms["route_b"]),
        _synth_exam_predictability("S5-2 Route A-only (stats 340)", s5_2_arms["route_a_only"]),
        _synth_exam_predictability("S5-2 Route A+B (v0.3.4 340)", s5_2_arms["route_ab"]),
        _synth_exam_predictability("S5-1 Planned Cohort (2000)", s5_1_js),
    ]
    if args.fold1 and os.path.exists(args.fold1):
        pred_rows.append(_real_exam_predictability("Real fold 1 (17-pathway matched)", args.fold1, match_17=True))
        pred_rows.append(_real_exam_predictability("Real fold 1 (Full)", args.fold1, match_17=False))

    with open(os.path.join(tables_dir, "s5_early_exam_predictability.json"), "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(tables_dir, "s5_early_exam_predictability.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "total", "missing_day_offset_docs", "eligible_los3_no_early", "late_surg_pos",
                     "pos_rate", "auc_demo_only", "auc_d01_exams_only", "auc_demo_plus_exams"])
        for r in pred_rows:
            w.writerow([
                r.get(k, 0 if k == "missing_day_offset_docs" else None)
                for k in ("cohort", "total", "missing_day_offset_docs", "eligible_los3_no_early",
                          "late_surg_pos", "pos_rate", "auc_demo_only", "auc_d01_exams_only", "auc_demo_plus_exams")
            ])
    print("PLAN_S5_DONE", json.dumps({"audit_only": args.audit_only, "s5_1": s5_1_summ, "pred": pred_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
