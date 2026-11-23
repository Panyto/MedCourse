# -*- coding: utf-8 -*-
"""Downstream T1 (calendar day 0-1 warning), T2a/T2b (masked clause & silent-inclusive update), and T3 harness.

Hyperparameters and thresholds are tuned/calibrated solely on fold 1 or internal training splits;
Stage 1 never touches fold 9. Real patient text stays strictly inside the controlled server.
"""
import argparse
import collections
import hashlib
import json
import os
import pickle
import random
import re

from s5_common import ACUTE_SEVERITY_WORDS, T2_EXCLUDE_PREFIXES, setup_paths

setup_paths()

from medcourse.rules import setup  # noqa: E402

setup(("BENCH_G3", "BENCH_G2", "BENCH_B7", "BENCH_G45"))
from g2_kit import oracle2  # noqa: E402
from ruler_g1 import split_sections  # noqa: E402

import numpy as np  # noqa: E402
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    from sklearn.metrics import (  # noqa: E402
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
    from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


_PAST_CLAUSE_RE = re.compile(r"既往|\d+\s*年前|\d+\s*月前|外院|曾")
_CLAUSE_BREAK_RE = re.compile(r"[，,。；;！？!?\n]")
_LEGACY_V13_EARLY_OP_RE = re.compile(
    r"术前小结|术后首次病程|今日在.{0,8}麻醉下行|在.{0,10}麻醉下行.{0,18}(?:术|切除|修补|造影|置管|穿刺|清创|引流)"
)
EARLY_OP_RE = re.compile(
    r"今日拟行|明日拟?行|择期拟?行|拟于.{0,8}行|(?<!不)拟行.{0,16}(?:术|切除|修补|造影|置管|支架|介入|碎石|清创|引流)|"
    r"签署.{0,8}(?:手术|介入)(?:知情)?同意书|术前小结|术后首次病程|在.{0,8}麻醉下行"
)
LATE_OP_RE = re.compile(
    r"术后首次病程|今日在.{0,8}麻醉下行|在.{0,10}麻醉下行.{0,18}(?:术|切除|修补|造影|置管|穿刺|清创|引流|支架|碎石)|"
    r"(?:今日|局麻下)行(?:经皮冠状动脉介入|冠状动脉支架置入|内镜下.{0,10}(?:切除|取石|止血))"
)
LATE_OP_MAJOR_RE = re.compile(
    r"术后首次病程|(?:今日)?在.{0,10}麻醉下行.{0,18}(?:术|切除|修补|清创|支架置入|碎石)|"
    r"(?:今日|局麻下)行(?:经皮冠状动脉介入|冠状动脉支架置入)"
)
DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|g|ug|μg|ml|mL|U|IU|片|粒|支)(?![A-Za-z0-9])", re.I)


def has_early_op_plan(text):
    """True if any non-historical clause contains an explicit early surgical/interventional plan."""
    for cl in _CLAUSE_BREAK_RE.split(text or ""):
        if cl and not _PAST_CLAUSE_RE.search(cl) and EARLY_OP_RE.search(cl):
            return True
    return False


def has_late_op_event(text, major_only=False):
    """True if any non-historical clause records a surgical/interventional procedure performed."""
    pat = LATE_OP_MAJOR_RE if major_only else LATE_OP_RE
    for cl in _CLAUSE_BREAK_RE.split(text or ""):
        if cl and not _PAST_CLAUSE_RE.search(cl) and pat.search(cl):
            return True
    return False

T2_DIFF_RE = re.compile(
    r"[？?]|待排|待查|待明确|尚待|拟诊|疑诊|考虑.{0,12}(?:可能|待)|需与.{0,16}鉴别|鉴别诊断|待进一步"
)
T2_EXCL_RE = re.compile(
    r"(?:基本|可|已|予|予以|临床)?排除|不支持.{0,12}(?:诊断|可能)|未见.{0,10}证据|可能性小|可能性低"
)
T2_NOT_EXCL_RE = re.compile(r"待排除|需排除|排除禁忌|不能排除|尚不能排除|不排除")
T2_CONF_RE = re.compile(r"诊断明确|明确诊断|确诊为?|考虑为.{0,12}诊断|符合.{0,12}诊断|诊断[:：]?\s*1\.")

_BASE_CANONICAL_DRUG_GROUPS = [
    ("抗菌药", ("头孢", "青霉素", "阿莫西林", "哌拉西林", "莫西沙星", "左氧氟沙星", "环丙沙星", "阿奇霉素", "甲硝唑", "奥硝唑", "美罗培南", "万古霉素", "磷霉素", "依替米星")),
    ("抑酸护胃药", ("奥美拉唑", "泮托拉唑", "兰索拉唑", "雷贝拉唑", "艾司奥美拉唑", "法莫替丁", "雷尼替丁")),
    ("抗血小板药", ("阿司匹林", "氯吡格雷", "替格瑞洛", "波立维")),
    ("抗凝药", ("低分子肝素", "肝素", "依诺肝素", "利伐沙班", "华法林")),
    ("调脂药", ("阿托伐他汀", "瑞舒伐他汀", "辛伐他汀", "匹伐他汀", "非诺贝特")),
    ("降压_RAS_CCB_BB", ("氨氯地平", "硝苯地平", "非洛地平", "缬沙坦", "厄贝沙坦", "氯沙坦", "依那普利", "沙库巴曲", "美托洛尔", "比索洛尔")),
    ("利尿剂", ("呋塞米", "螺内酯", "氢氯噻嗪", "托拉塞米")),
    ("降糖药", ("二甲双胍", "阿卡波糖", "格列美脲", "达格列净", "胰岛素", "门冬胰岛素")),
    ("止痛解热抗炎", ("布洛芬", "对乙酰氨基酚", "洛索洛芬", "塞来昔布", "双氯芬酸", "氟比洛芬", "曲马多", "双氢可待因", "帕瑞昔布")),
    ("糖皮质激素", ("甲泼尼龙", "地塞米松", "泼尼松", "氢化可的松", "布地奈德")),
    ("祛痰平喘止咳", ("氨溴索", "乙酰半胱氨酸", "羧甲司坦", "异丙托溴铵", "沙丁胺醇", "多索茶碱", "氨茶碱")),
    ("抗肿瘤与止吐", ("奥沙利铂", "顺铂", "卡培他滨", "环磷酰胺", "紫杉醇", "昂丹司琼", "格拉司琼", "胃复安", "甲氧氯普胺")),
]

_DRUG_CLASS_TO_CANONICAL = {
    "抗菌药": "抗菌药",
    "抑酸药": "抑酸护胃药",
    "抗血小板药": "抗血小板药",
    "抗凝药": "抗凝药",
    "调脂药": "调脂药",
    "RAS抑制剂": "降压_RAS_CCB_BB",
    "钙通道阻滞剂": "降压_RAS_CCB_BB",
    "β受体阻滞剂": "降压_RAS_CCB_BB",
    "利尿剂": "利尿剂",
    "降糖药": "降糖药",
    "解热镇痛抗炎药": "止痛解热抗炎",
    "阿片类镇痛药": "止痛解热抗炎",
    "糖皮质激素": "糖皮质激素",
    "祛痰镇咳药": "祛痰平喘止咳",
    "平喘药": "祛痰平喘止咳",
    "抗肿瘤药": "抗肿瘤与止吐",
    "胃肠动力和止吐药": "抗肿瘤与止吐",
}


def _load_enriched_drug_groups():
    """Enrich canonical 12 drug groups with Chinese aliases from ruler_g3 drug_lexicon.csv when available."""
    buckets = {grp: list(kws) for grp, kws in _BASE_CANONICAL_DRUG_GROUPS}
    seen = {grp: set(kws) for grp, kws in _BASE_CANONICAL_DRUG_GROUPS}
    try:
        import csv
        import ruler_g3
        from medcourse import drug_class

        lex_csv = os.path.join(os.path.dirname(ruler_g3.__file__), "data", "drug_lexicon.csv")
        if os.path.exists(lex_csv):
            with open(lex_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    did = (row.get("drug_id") or "").strip()
                    dc = drug_class.DRUG_CLASS.get(did)
                    canon = _DRUG_CLASS_TO_CANONICAL.get(dc)
                    if not canon:
                        continue
                    aliases = str(row.get("aliases") or "")
                    for part in re.split(r"[|,;、]+", aliases):
                        alias = part.strip()
                        if len(alias) >= 2 and re.search(r"[\u4e00-\u9fa5]", alias) and alias not in seen[canon]:
                            buckets[canon].append(alias)
                            seen[canon].add(alias)
    except Exception:
        pass
    return [(grp, tuple(buckets[grp])) for grp, _ in _BASE_CANONICAL_DRUG_GROUPS]


CANONICAL_DRUG_GROUPS = _load_enriched_drug_groups()
DRUG_GROUP_NAMES = [x[0] for x in CANONICAL_DRUG_GROUPS]


def _read_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit is not None and len(out) >= limit:
                    break
    return out


def _is_discharge(doc):
    return doc.get("doc_type") == "discharge_summary" or doc.get("role") == "DS"


def _is_hr(doc):
    return doc.get("doc_type") == "hospital_record" or doc.get("role") == "HR"


def _in_day_window(doc, lo, hi=None):
    dy = doc.get("day_offset")
    if dy is None:
        return False
    d = int(dy)
    return d >= lo and (hi is None or d <= hi)


def _ece_score(y_true, probs, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(y_true[mask])) - float(np.mean(probs[mask])))
    return round(float(ece), 4)


def _cluster_bootstrap_binary(y, probs, group_ids, n_boot=2000, seed=20261201):
    rng = np.random.RandomState(seed)
    uniq = list(dict.fromkeys(group_ids))
    idx_by_g = collections.defaultdict(list)
    for idx, g in enumerate(group_ids):
        idx_by_g[g].append(idx)
    aucs, aps = [], []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sel = [i for g in sampled for i in idx_by_g[g]]
        yt, yp = y[sel], probs[sel]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        aps.append(average_precision_score(yt, yp))
    return {
        "auroc": round(float(roc_auc_score(y, probs)), 4) if len(np.unique(y)) >= 2 else None,
        "auroc_ci95": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)] if aucs else None,
        "auprc": round(float(average_precision_score(y, probs)), 4) if len(np.unique(y)) >= 2 else None,
        "auprc_ci95": [round(float(np.percentile(aps, 2.5)), 4), round(float(np.percentile(aps, 97.5)), 4)] if aps else None,
        "brier": round(float(brier_score_loss(y, probs)), 4),
        "ece": _ece_score(y, probs),
        "n": int(len(y)),
        "pos": int(np.sum(y)),
    }


def paired_bootstrap_diff(y, probs_a, probs_b, group_ids, n_boot=2000, seed=20261201):
    """Cluster-bootstrap paired difference (Arm A minus Arm B) on identical evaluation patients."""
    y = np.asarray(y, dtype=int)
    pa = np.asarray(probs_a, dtype=float)
    pb = np.asarray(probs_b, dtype=float)
    if len(y) == 0 or len(y) != len(pa) or len(y) != len(pb) or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "delta_auroc": None}
    auc_a = float(roc_auc_score(y, pa))
    auc_b = float(roc_auc_score(y, pb))
    ap_a = float(average_precision_score(y, pa))
    ap_b = float(average_precision_score(y, pb))

    rng = np.random.RandomState(seed)
    uniq = list(dict.fromkeys(group_ids))
    idx_by_g = collections.defaultdict(list)
    for idx, g in enumerate(group_ids):
        idx_by_g[g].append(idx)
    d_aucs, d_aps = [], []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sel = [i for g in sampled for i in idx_by_g[g]]
        yt = y[sel]
        if len(np.unique(yt)) < 2:
            continue
        d_aucs.append(roc_auc_score(yt, pa[sel]) - roc_auc_score(yt, pb[sel]))
        d_aps.append(average_precision_score(yt, pa[sel]) - average_precision_score(yt, pb[sel]))
    p_auc = None
    if d_aucs:
        arr = np.asarray(d_aucs, dtype=float)
        p_auc = round(float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))), 4)
    return {
        "n": int(len(y)),
        "pos": int(np.sum(y)),
        "auroc_a": round(auc_a, 4),
        "auroc_b": round(auc_b, 4),
        "delta_auroc": round(auc_a - auc_b, 4),
        "delta_auroc_ci95": [round(float(np.percentile(d_aucs, 2.5)), 4), round(float(np.percentile(d_aucs, 97.5)), 4)] if d_aucs else None,
        "auprc_a": round(ap_a, 4),
        "auprc_b": round(ap_b, 4),
        "delta_auprc": round(ap_a - ap_b, 4),
        "delta_auprc_ci95": [round(float(np.percentile(d_aps, 2.5)), 4), round(float(np.percentile(d_aps, 97.5)), 4)] if d_aps else None,
        "p_boot_two_sided": p_auc,
    }


def _ds_new_severity(docs):
    ds = next((d for d in reversed(docs) if _is_discharge(d)), None)
    if not ds:
        return False
    secs = {s.name: s.text for s in split_sections(ds["text"])}
    adx = secs.get("入院诊断", "")
    ddx = secs.get("出院诊断", "")
    for line in ddx.splitlines():
        clean_l = re.sub(r"^\d+[\.、\s]*", "", line).strip().rstrip("？?")
        if len(clean_l) >= 2 and clean_l not in adx:
            if any(w in clean_l for w in ACUTE_SEVERITY_WORDS):
                return True
    return False


def _id_list_sha256(ids):
    payload = "\n".join(sorted(str(x) for x in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_fold_docs(fold_path, out_private_ids_json=None):
    """Audit day_offset completeness, T1 proxy counts, and exact T1 CONSORT flow without emitting raw IDs."""
    adms = _read_jsonl(fold_path)
    fold_sha256 = _sha256_file(fold_path)
    total_docs = 0
    missing_docs = 0
    missing_by_type = collections.Counter()
    t1_elig_ge2 = 0
    t1_pos_all_d2p = 0
    t1_pos_non_ds_d2p = 0
    t1_pos_only_in_ds = 0
    t1_pos_gt2_non_ds = 0
    t1_pos_major_non_ds = 0
    earliest_day_hist = collections.Counter()

    buckets = {
        "all_admissions": [],
        "excluded_max_day_lt_3": [],
        "remaining_los_ge3": [],
        "excluded_empty_d01": [],
        "remaining_los_ge3_with_d01": [],
        "excluded_early_op_both_v13_and_v16": [],
        "excluded_early_op_v13_only": [],
        "excluded_early_op_v16_newly_added": [],
        "excluded_early_op_d01_total": [],
        "v13_eligible_t1_ge2": [],
        "eligible_t1_ge2": [],
    }

    for a in adms:
        aid = str(a["admission_id"])
        buckets["all_admissions"].append(aid)
        docs = a["docs"]
        for d in docs:
            total_docs += 1
            if d.get("day_offset") is None:
                missing_docs += 1
                missing_by_type[d.get("doc_type", "UNK")] += 1
        days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
        if (max(days) if days else 0) < 3:
            buckets["excluded_max_day_lt_3"].append(aid)
            continue
        buckets["remaining_los_ge3"].append(aid)
        d01_text = "\n".join(d["text"] for d in docs if _in_day_window(d, 0, 1) and not _is_discharge(d))
        if not d01_text:
            buckets["excluded_empty_d01"].append(aid)
            continue
        buckets["remaining_los_ge3_with_d01"].append(aid)

        old_hit = bool(_LEGACY_V13_EARLY_OP_RE.search(d01_text))
        new_hit = has_early_op_plan(d01_text)
        if not old_hit:
            buckets["v13_eligible_t1_ge2"].append(aid)
        if old_hit and new_hit:
            buckets["excluded_early_op_both_v13_and_v16"].append(aid)
            buckets["excluded_early_op_d01_total"].append(aid)
            continue
        if old_hit and not new_hit:
            buckets["excluded_early_op_v13_only"].append(aid)
            buckets["eligible_t1_ge2"].append(aid)
        elif new_hit and not old_hit:
            buckets["excluded_early_op_v16_newly_added"].append(aid)
            buckets["excluded_early_op_d01_total"].append(aid)
            continue
        else:
            buckets["eligible_t1_ge2"].append(aid)

        t1_elig_ge2 += 1
        d2p_all = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, None))
        d2p_nonds = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, None) and not _is_discharge(d))
        p_all = has_late_op_event(d2p_all, major_only=False)
        p_nonds = has_late_op_event(d2p_nonds, major_only=False)
        p_maj = has_late_op_event(d2p_nonds, major_only=True)
        t1_pos_all_d2p += int(p_all)
        t1_pos_non_ds_d2p += int(p_nonds)
        t1_pos_major_non_ds += int(p_maj)
        if p_all and not p_nonds:
            t1_pos_only_in_ds += 1
        if p_nonds:
            hit_days = [
                int(d["day_offset"])
                for d in docs
                if _in_day_window(d, 2, None) and not _is_discharge(d) and has_late_op_event(d["text"])
            ]
            if hit_days:
                earliest_day_hist[str(min(hit_days))] += 1
        d2_text = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, 2) and not _is_discharge(d))
        if not (has_early_op_plan(d2_text) or has_late_op_event(d2_text)):
            d3p_nonds = "\n".join(d["text"] for d in docs if _in_day_window(d, 3, None) and not _is_discharge(d))
            if has_late_op_event(d3p_nonds):
                t1_pos_gt2_non_ds += 1

    try:
        tune_ids, eval_ids = _rehearsal_patient_split(fold_path)
        buckets["eligible_t1_rehearsal_tune"] = [x for x in buckets["eligible_t1_ge2"] if x in tune_ids]
        buckets["eligible_t1_rehearsal_eval"] = [x for x in buckets["eligible_t1_ge2"] if x in eval_ids]
    except Exception:
        buckets["eligible_t1_rehearsal_tune"] = []
        buckets["eligible_t1_rehearsal_eval"] = []

    if out_private_ids_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_private_ids_json)), exist_ok=True)
        priv_obj = {
            "fold_path": os.path.basename(fold_path),
            "fold_file_sha256": fold_sha256,
            "buckets": {k: sorted(v) for k, v in buckets.items()},
        }
        with open(out_private_ids_json, "w", encoding="utf-8") as pf:
            json.dump(priv_obj, pf, ensure_ascii=False, indent=2)

    consort = {
        "fold_file_sha256": fold_sha256,
        "counts": {k: len(v) for k, v in buckets.items()},
        "id_set_sha256": {k: _id_list_sha256(v) for k, v in buckets.items()},
        "arithmetic_checks": {
            "los_partition_ok": (
                len(buckets["excluded_max_day_lt_3"])
                + len(buckets["excluded_empty_d01"])
                + len(buckets["excluded_early_op_d01_total"])
                + len(buckets["eligible_t1_ge2"])
                == len(buckets["all_admissions"])
            ),
            "v13_to_v16_delta_ok": (
                len(buckets["v13_eligible_t1_ge2"])
                - len(buckets["excluded_early_op_v16_newly_added"])
                + len(buckets["excluded_early_op_v13_only"])
                == len(buckets["eligible_t1_ge2"])
            ),
        },
        "note": "Regex exclusions on fold 1 are rule-based counts prior to clinician audit on the 69-stay stratified pool.",
    }

    return {
        "total_admissions": len(adms),
        "total_docs": total_docs,
        "missing_day_offset_docs": missing_docs,
        "missing_by_doc_type": dict(missing_by_type),
        "t1_elig_ge2": t1_elig_ge2,
        "t1_pos_all_d2p": t1_pos_all_d2p,
        "t1_pos_non_ds_d2p": t1_pos_non_ds_d2p,
        "t1_pos_major_non_ds": t1_pos_major_non_ds,
        "t1_pos_only_in_ds": t1_pos_only_in_ds,
        "t1_pos_gt2_non_ds": t1_pos_gt2_non_ds,
        "t1_pos_earliest_day_hist": dict(sorted(earliest_day_hist.items(), key=lambda kv: int(kv[0]))),
        "t1_consort_flow": consort,
    }


def export_t1_audit_pool(fold_path, out_dir, seed=20261201):
    """Export bidirectional stratified T1 audit pool (excluded early, pos, surgical-dept neg, other neg)."""
    adms = _read_jsonl(fold_path)
    buckets = {
        "excluded_early_op": [],
        "included_pos": [],
        "included_neg_surg_nearmiss": [],
        "included_neg_other": [],
    }
    near_re = re.compile(r"外科|骨科|泌尿|肝胆|胃肠|拟行|择期|手术|麻醉")
    for a in adms:
        aid = str(a["admission_id"])
        docs = a["docs"]
        days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
        if (max(days) if days else 0) < 3:
            continue
        d01_text = "\n".join(d["text"] for d in docs if _in_day_window(d, 0, 1) and not _is_discharge(d))
        d2p_nonds = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, None) and not _is_discharge(d))
        ds_txt = "\n".join(d["text"] for d in docs if _is_discharge(d))
        stay_summary = [
            {"k": idx, "day_offset": d.get("day_offset"), "doc_type": d.get("doc_type"), "text": d.get("text", "")}
            for idx, d in enumerate(docs)
        ]
        if has_early_op_plan(d01_text):
            st = "excluded_early_op"
        elif has_late_op_event(d2p_nonds):
            st = "included_pos"
        elif near_re.search(d01_text + "\n" + d2p_nonds + "\n" + ds_txt):
            st = "included_neg_surg_nearmiss"
        else:
            st = "included_neg_other"
        buckets[st].append({"admission_id": aid, "stratum": st, "docs": stay_summary})

    rng = random.Random(seed)
    quotas = (("excluded_early_op", 20), ("included_pos", 25), ("included_neg_surg_nearmiss", 15), ("included_neg_other", 15))
    picked = []
    strata_meta = {}
    for st, want in quotas:
        pool_st = list(buckets[st])
        rng.shuffle(pool_st)
        sel = pool_st[:min(want, len(pool_st))]
        N_h, n_h = len(pool_st), len(sel)
        strata_meta[st] = {"N_h": N_h, "n_h": n_h, "sampling_prob": round(n_h / max(1, N_h), 6), "ht_weight": round(N_h / max(1, n_h), 4)}
        for item in sel:
            picked.append((item, N_h, n_h))
    rng.shuffle(picked)

    os.makedirs(out_dir, exist_ok=True)
    sheet_p = os.path.join(out_dir, "t1_clinician_audit_sheet.jsonl")
    key_p = os.path.join(out_dir, "T1_SEALED_AUDIT_KEY.json")
    key_rows = []
    with open(sheet_p, "w", encoding="utf-8") as sf:
        for idx, (item, N_h, n_h) in enumerate(picked):
            bid = f"T1AUD-{idx:03d}"
            sf.write(json.dumps({
                "blind_id": bid,
                "docs": item["docs"],
                "c_early_already_planned_0or1": None,
                "c_decide_day": None,
                "c_perform_day": None,
                "c_first_recorded_day": None,
                "c_has_late_op_ge2_0or1": None,
                "c_note": "",
            }, ensure_ascii=False) + "\n")
            key_rows.append({
                "blind_id": bid,
                "admission_id": item["admission_id"],
                "auto_stratum": item["stratum"],
                "N_h": N_h,
                "n_h": n_h,
                "ht_weight": round(N_h / max(1, n_h), 4),
            })
    with open(key_p, "w", encoding="utf-8") as kf:
        json.dump({"seed": seed, "total": len(key_rows), "strata": strata_meta, "mapping": key_rows}, kf, ensure_ascii=False, indent=2)
    return {"total_exported": len(key_rows), "strata": strata_meta}


def build_t1_dataset(
    adm_paths,
    journey_paths=None,
    max_n=None,
    seed=20261201,
    gt2_only=False,
    max_raw_stays=None,
    major_op_only=False,
):
    """Build (text_d01, y_t1a, y_t1b, admission_id) with strict day_offset in {0, 1} input and no DS bleed."""
    adms = []
    for p in adm_paths:
        adms.extend(_read_jsonl(p))
    if max_raw_stays is not None and len(adms) > max_raw_stays:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(adms), size=max_raw_stays, replace=False)
        adms = [adms[i] for i in sorted(idx)]
    jmap = {}
    if journey_paths:
        for jp in journey_paths:
            if jp and os.path.exists(jp):
                for j in _read_jsonl(jp):
                    jmap[j["journey_id"]] = j

    early_cut = 2 if gt2_only else 1
    rows = []
    for a in adms:
        aid = a["admission_id"]
        docs = a["docs"]
        d01_text = "\n".join(d["text"] for d in docs if _in_day_window(d, 0, 1) and not _is_discharge(d))
        if not d01_text:
            continue
        j = jmap.get(aid)
        if j is not None:
            if j["los"] < 3:
                continue
            sp = [x for x in (j.get("procedures") or ()) if x.get("kind") == "手术"]
            if sp and (sp[0]["decide_day"] <= early_cut or sp[0]["do_day"] <= early_cut):
                continue
            y_a = 1 if (sp and sp[0]["decide_day"] > early_cut) else 0
            sev_ds = any(
                (not c.get("at_admission", True)) and any(w in c["name"] for w in ACUTE_SEVERITY_WORDS)
                for c in (j["patient"].get("comorbidities") or ())
            )
            y_b = 1 if (y_a or sev_ds) else 0
            rows.append((d01_text, y_a, y_b, aid))
        else:
            days = [int(d["day_offset"]) for d in docs if d.get("day_offset") is not None]
            if (max(days) if days else 0) < 3:
                continue
            if has_early_op_plan(d01_text):
                continue
            if gt2_only:
                d2_text = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, 2) and not _is_discharge(d))
                if has_early_op_plan(d2_text) or has_late_op_event(d2_text, major_only=major_op_only):
                    continue
            late_text = "\n".join(
                d["text"] for d in docs if _in_day_window(d, early_cut + 1, None) and not _is_discharge(d)
            )
            y_a = 1 if has_late_op_event(late_text, major_only=major_op_only) else 0
            y_b = 1 if (y_a or _ds_new_severity(docs)) else 0
            rows.append((d01_text, y_a, y_b, aid))

    if max_n is not None and len(rows) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows), size=max_n, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def _mask_dx_clauses(note_text, dx_name, window=90, replace_token="[TARGET_DX]", max_chars=300):
    """Extract local spans around dx_name inside a note with zero silent-tag or ASCII-delimiter leakage."""
    txt = note_text or ""
    if not dx_name or dx_name not in txt:
        return re.sub(r"\s+", " ", txt).strip()[:max_chars]
    spans = []
    for m in re.finditer(re.escape(dx_name), txt):
        lo = max(0, m.start() - window)
        hi = min(len(txt), m.end() + window)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    snippets = [re.sub(r"\s+", " ", txt[lo:hi].replace(dx_name, replace_token)).strip() for lo, hi in spans]
    return " ".join(snippets)[:max_chars]


def _extract_real_t2_candidates(docs):
    """Extract (dx_name, starts_uncertain) from HR admission diagnosis and Day-0 FP differentials."""
    hr = next((d for d in docs if _is_hr(d)), None)
    if not hr:
        return []
    hr_secs = {s.name: s.text for s in split_sections(hr["text"])}
    admit_lines = [
        re.sub(r"^\d+[\.、\s]*", "", l).strip()
        for l in hr_secs.get("入院诊断", "").splitlines()
        if len(l.strip()) >= 2
    ]
    cands = []
    seen = set()
    for ln in admit_lines[:3]:
        is_q = ("？" in ln or "?" in ln)
        base_dx = re.sub(r"[？?()（）].*$", "", ln).strip()
        if 2 <= len(base_dx) <= 18 and base_dx not in seen and not any(
            w in base_dx for w in ("化疗", "放疗", "维持", "待产", "流产")
        ):
            cands.append((base_dx, is_q))
            seen.add(base_dx)
    fp = next((d for d in docs if not _is_hr(d) and not _is_discharge(d) and _in_day_window(d, 0, 0)), None)
    if fp:
        for m in re.finditer(r"(?:与|排除)([\u4e00-\u9fa5A-Za-z0-9]{2,12}?)(?:相?鉴别|可能)", fp["text"]):
            nm = m.group(1)
            if (
                len(nm) >= 2
                and nm not in seen
                and not any(w in nm for w in ("相关", "其他", "临床", "上述", "手术", "目前", "禁忌"))
            ):
                cands.append((nm, True))
                seen.add(nm)
    return cands


def _clean_anchor_label(txt, dx_name):
    """Return 0=excluded, 1=differential, 2=confirmed if clean linguistic anchor hits; else None."""
    lbl = None
    for m in re.finditer(re.escape(dx_name), txt):
        lo, hi = max(0, m.start() - 45), min(len(txt), m.end() + 45)
        win = txt[lo:hi]
        fs = oracle2.firm_state(txt, m.start(), m.end())
        if fs is not None:
            return 2 if fs[1] == "确诊" else 0
        if T2_DIFF_RE.search(win) or T2_NOT_EXCL_RE.search(win):
            lbl = 1
        elif T2_EXCL_RE.search(win):
            lbl = 0
        elif T2_CONF_RE.search(win):
            lbl = 2
    return lbl


def _load_subset_stays(adm_paths, journey_paths=None, max_stays=None, seed=20261201, max_raw_stays=None):
    adms = []
    for p in adm_paths:
        adms.extend(_read_jsonl(p))
    cap = max_raw_stays if max_raw_stays is not None else max_stays
    if cap is not None and len(adms) > cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(adms), size=cap, replace=False)
        adms = [adms[i] for i in sorted(idx)]
    jmap = {}
    if journey_paths:
        for jp in journey_paths:
            if jp and os.path.exists(jp):
                for j in _read_jsonl(jp):
                    jmap[j["journey_id"]] = j
    return adms, jmap


_CTX_EXAMS = (
    ("血常规", re.compile(r"血常规")),
    ("C反应蛋白", re.compile(r"C反应蛋白|CRP")),
    ("肝肾功", re.compile(r"肝功能|肾功能|肌酐|转氨酶")),
    ("电解质", re.compile(r"电解质")),
    ("CT", re.compile(r"CT")),
    ("超声", re.compile(r"超声|彩超|B超")),
    ("核磁", re.compile(r"磁共振|MRI|MRCP")),
    ("心电图", re.compile(r"心电图")),
    ("肌钙蛋白", re.compile(r"肌钙蛋白|BNP|D[-—]?二聚体")),
    ("病理", re.compile(r"病理")),
)


def _prior_context_prefix(docs, up_to_idx):
    """Summarize prior admission diagnoses and prior exam mentions strictly before docs[up_to_idx]."""
    hr = next((d for d in docs[:up_to_idx] if _is_hr(d)), None)
    adx = ""
    if hr:
        secs = {s.name: s.text for s in split_sections(hr["text"])}
        adx = re.sub(r"\s+", " ", secs.get("入院诊断", "")).strip()[:60]
    prior_txt = "\n".join(d.get("text") or "" for d in docs[:up_to_idx])
    seen_ex = [nm for nm, pat in _CTX_EXAMS if pat.search(prior_txt)]
    return f"入院拟诊：{adx or '无'}；前序检查：{('、'.join(seen_ex[:6])) or '无'}。"


def _is_course_doc(doc):
    role = doc.get("role")
    if role is not None:
        return role in ("SUP", "DAILY")
    txt = doc.get("text") or ""
    if "术后首次病程" in txt or "术前小结" in txt:
        return False
    return (not _is_hr(doc)) and (not _is_discharge(doc))


def _is_pre_doc(doc):
    role = doc.get("role")
    if role is not None:
        return role == "PRE"
    return "术前小结" in (doc.get("text") or "")


def _is_post_doc(doc):
    role = doc.get("role")
    if role is not None:
        return role == "POST"
    return "术后首次病程" in (doc.get("text") or "")


_DOC_HM_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\s+([01]\d|2[0-3]):([0-5]\d)\b")


def _doc_hm(doc):
    tm = doc.get("time")
    if isinstance(tm, str) and len(tm) == 5 and tm[2] == ":":
        return tm
    m = _DOC_HM_RE.search((doc.get("text") or "")[:240])
    return f"{m.group(1)}:{m.group(2)}" if m else "23:59"


def _event_hm(j, dx_name, ev_day, ev_st):
    for ev in (j.get("dx_events") or ()):
        if ev.get("name") == dx_name and int(ev.get("day", -1)) == ev_day and ev.get("state") == ev_st:
            hm = ev.get("hm")
            if isinstance(hm, str) and len(hm) == 5 and hm[2] == ":":
                return hm
    return None


def _find_compiler_update_doc_idx(j, all_docs, dx_name, ev_day, ev_st):
    """Identify exact note index k_star matching medcourse/compiler.py (t, k) and optional intra-day hm visibility.

    Returns (k_star_idx_or_None, obs_p_indices) along chronological day>=1 non-HR/non-DS notes.
    """
    p_indices = [
        idx for idx, d in enumerate(all_docs)
        if not _is_hr(d) and not _is_discharge(d) and _in_day_window(d, 1, None)
    ]
    if ev_day is None or ev_st not in ("确诊", "排除") or int(ev_day) < 1:
        return None, p_indices

    ev_day = int(ev_day)
    ev_hm = _event_hm(j, dx_name, ev_day, ev_st)
    prim_name = ((j.get("patient") or {}).get("primary_dx") or {}).get("name") or ""
    k_star = None
    for idx in p_indices:
        d = all_docs[idx]
        dy = int(d["day_offset"])
        if _is_pre_doc(d):
            continue
        if _is_post_doc(d):
            cond = (
                (dy, _doc_hm(d)) >= (ev_day, ev_hm)
                if ev_hm is not None
                else dy >= ev_day
            )
            if cond and ev_st == "确诊" and dx_name == prim_name:
                k_star = idx
                break
            continue
        if ev_hm is not None:
            prev_ts = max(
                [(int(all_docs[p]["day_offset"]), _doc_hm(all_docs[p])) for p in p_indices if p < idx and _is_course_doc(all_docs[p])],
                default=(0, "23:59"),
            )
            if prev_ts < (ev_day, ev_hm) <= (dy, _doc_hm(d)):
                k_star = idx
                break
        else:
            prev_c = max(
                [int(all_docs[p]["day_offset"]) for p in p_indices if p < idx and _is_course_doc(all_docs[p])],
                default=0,
            )
            if prev_c < ev_day <= dy:
                k_star = idx
                break

    if k_star is not None:
        obs_indices = [idx for idx in p_indices if idx <= k_star]
    elif ev_hm is not None:
        obs_indices = [idx for idx in p_indices if (int(all_docs[idx]["day_offset"]), _doc_hm(all_docs[idx])) < (ev_day, ev_hm)]
    else:
        obs_indices = [idx for idx in p_indices if int(all_docs[idx]["day_offset"]) < ev_day]
    return k_star, obs_indices


def _dx_state_at_doc_idx(j, all_docs, dx_name, doc_idx, d_evs):
    """Resolve compiler-aligned diagnostic state for (dx_name, doc_idx)."""
    st = None
    dy = int(all_docs[doc_idx]["day_offset"])
    d_hm = _doc_hm(all_docs[doc_idx])
    for ev_day, ev_st in sorted(d_evs):
        ev_day = int(ev_day)
        if ev_day <= 0:
            st = ev_st
        elif ev_st in ("确诊", "排除"):
            k_star, _ = _find_compiler_update_doc_idx(j, all_docs, dx_name, ev_day, ev_st)
            if k_star is not None and doc_idx >= k_star:
                st = ev_st
            elif k_star is None:
                ev_hm = _event_hm(j, dx_name, ev_day, ev_st)
                if ev_hm is not None and (dy, d_hm) >= (ev_day, ev_hm):
                    st = ev_st
                elif ev_hm is None and dy > ev_day:
                    st = ev_st
        else:
            ev_hm = _event_hm(j, dx_name, ev_day, ev_st)
            if ev_hm is not None:
                if (ev_day, ev_hm) <= (dy, d_hm):
                    st = ev_st
            elif ev_day <= dy:
                st = ev_st
    return st


def build_t2a_clause_dataset(
    adm_paths,
    journey_paths=None,
    max_stays=None,
    seed=20261201,
    label_source="auto",
    allow_discharge=False,
    max_raw_stays=None,
):
    """T2a: masked clause 3-class (0=excluded, 1=differential, 2=confirmed) with note-level (t, k) compiler state."""
    adms, jmap = _load_subset_stays(adm_paths, journey_paths, max_stays, seed, max_raw_stays=max_raw_stays)
    samples = []
    for a in adms:
        aid = a["admission_id"]
        docs = a["docs"]
        j = jmap.get(aid)
        if j is not None and label_source in ("auto", "planner"):
            c3 = j["patient"]["primary_dx"].get("code3") or ""
            if any(c3.startswith(px) for px in T2_EXCLUDE_PREFIXES):
                continue
            evs = [e for e in (j.get("dx_events") or ()) if len(e.get("name") or "") >= 2]
            if not evs:
                continue
            by_dx = collections.defaultdict(list)
            for e in evs:
                by_dx[e["name"]].append((int(e["day"]), e["state"]))
            for doc_idx, d in enumerate(docs):
                if _is_hr(d) or (not allow_discharge and _is_discharge(d)) or d.get("day_offset") is None:
                    continue
                txt = d["text"]
                for dx_name, d_evs in by_dx.items():
                    if dx_name not in txt:
                        continue
                    st = _dx_state_at_doc_idx(j, docs, dx_name, doc_idx, d_evs)
                    if st is None:
                        continue
                    lbl = 2 if st == "确诊" else (0 if st == "排除" else 1)
                    samples.append((_mask_dx_clauses(txt, dx_name), lbl, aid, dx_name))
        else:
            cands = _extract_real_t2_candidates(docs)
            if not cands:
                continue
            for d in docs:
                if _is_hr(d) or (not allow_discharge and _is_discharge(d)) or d.get("day_offset") is None:
                    continue
                txt = d["text"]
                for dx_name, _ in cands:
                    if dx_name not in txt:
                        continue
                    lbl = _clean_anchor_label(txt, dx_name)
                    if lbl is not None:
                        samples.append((_mask_dx_clauses(txt, dx_name), lbl, aid, dx_name))
    return samples


def build_t2b_update_dataset(
    adm_paths,
    journey_paths=None,
    max_stays=None,
    seed=20261201,
    label_source="auto",
    with_context=False,
    max_raw_stays=None,
):
    """T2b: note-level update detection on day>=1 progress notes with compiler (t, k) timing."""
    adms, jmap = _load_subset_stays(adm_paths, journey_paths, max_stays, seed, max_raw_stays=max_raw_stays)
    rows = []
    for a in adms:
        aid = str(a["admission_id"])
        all_docs = a["docs"]
        p_indices = [
            idx for idx, d in enumerate(all_docs)
            if not _is_hr(d) and not _is_discharge(d) and _in_day_window(d, 1, None)
        ]
        if not p_indices:
            continue
        j = jmap.get(aid)
        if j is not None and label_source in ("auto", "planner"):
            c3 = j["patient"]["primary_dx"].get("code3") or ""
            if any(c3.startswith(px) for px in T2_EXCLUDE_PREFIXES):
                continue
            by_dx = collections.defaultdict(list)
            for e in (j.get("dx_events") or ()):
                if len(e.get("name") or "") >= 2:
                    by_dx[e["name"]].append((int(e["day"]), e["state"]))
            for dx_name, evs in by_dx.items():
                sevs = sorted(evs)
                if not (sevs[0][0] == 0 and sevs[0][1] == "疑似"):
                    continue
                trans = [e for e in sevs if e[0] >= 1 and e[1] in ("确诊", "排除")]
                tday, tst = trans[0] if trans else (None, None)
                k_star, obs_indices = _find_compiler_update_doc_idx(j, all_docs, dx_name, tday, tst)
                for doc_idx in obs_indices:
                    d = all_docs[doc_idx]
                    dy = int(d["day_offset"])
                    txt = d["text"]
                    feat = _mask_dx_clauses(txt, dx_name, replace_token="该病", max_chars=300)
                    if with_context:
                        feat = _prior_context_prefix(all_docs, doc_idx) + " " + feat
                    y = 1 if (k_star is not None and doc_idx == k_star) else 0
                    rows.append((feat, y, dx_name not in txt, aid, dx_name, dy, doc_idx))
        else:
            cands = [(dx, is_q) for dx, is_q in _extract_real_t2_candidates(all_docs) if is_q]
            for dx_name, _ in cands:
                settled = False
                for doc_idx in p_indices:
                    if settled:
                        break
                    d = all_docs[doc_idx]
                    txt = d["text"]
                    dy = int(d["day_offset"])
                    is_silent = dx_name not in txt
                    lbl = None if is_silent else _clean_anchor_label(txt, dx_name)
                    y = 1 if lbl in (0, 2) else 0
                    feat = _mask_dx_clauses(txt, dx_name, replace_token="该病", max_chars=300)
                    if with_context:
                        feat = _prior_context_prefix(all_docs, doc_idx) + " " + feat
                    rows.append((feat, y, is_silent, aid, dx_name, dy, doc_idx))
                    if y == 1:
                        settled = True
    return rows


def build_t2b_fixed_frame_2x2(full_adm_paths, full_j_paths, vis_adm_paths, vis_j_paths, with_context=False):
    """Build strictly shared-frame 2x2 T2b datasets (fixed stay/dx/note observation window across all 4 cells)."""
    full_by_id = {str(a["admission_id"]): a for a in _read_many_jsonl(full_adm_paths)}
    vis_by_id = {str(a["admission_id"]): a for a in _read_many_jsonl(vis_adm_paths)}
    jmap = {str(j["journey_id"]): j for j in (_read_many_jsonl(full_j_paths) + _read_many_jsonl(vis_j_paths))}
    paired_ids = sorted(set(full_by_id) & set(vis_by_id) & set(jmap))

    cell_names = (
        "Synth-Full | fixed_planner_tk",
        "Synth-Full | fixed_text_rule",
        "Synth-VisibilityOnly | fixed_planner_tk",
        "Synth-VisibilityOnly | fixed_text_rule",
    )
    cells = {c: [] for c in cell_names}
    aligned_stays = 0
    misaligned_stays = 0
    kstar_tot = 0
    kstar_hits = {"Full": 0, "VisibilityOnly": 0}

    for aid in paired_ids:
        af = full_by_id[aid]
        av = vis_by_id[aid]
        j = jmap[aid]
        df_docs = af["docs"]
        dv_docs = av["docs"]
        pf_idx = [i for i, d in enumerate(df_docs) if not _is_hr(d) and not _is_discharge(d) and _in_day_window(d, 1, None)]
        pv_idx = [i for i, d in enumerate(dv_docs) if not _is_hr(d) and not _is_discharge(d) and _in_day_window(d, 1, None)]
        sched_f = [(int(df_docs[i]["day_offset"]), df_docs[i].get("role")) for i in pf_idx]
        sched_v = [(int(dv_docs[i]["day_offset"]), dv_docs[i].get("role")) for i in pv_idx]
        if sched_f != sched_v or not pf_idx:
            misaligned_stays += 1
            continue
        aligned_stays += 1

        c3 = j["patient"]["primary_dx"].get("code3") or ""
        if any(c3.startswith(px) for px in T2_EXCLUDE_PREFIXES):
            continue
        by_dx = collections.defaultdict(list)
        for e in (j.get("dx_events") or ()):
            if len(e.get("name") or "") >= 2:
                by_dx[e["name"]].append((int(e["day"]), e["state"]))

        p_pos_map = {doc_idx: pos for pos, doc_idx in enumerate(pf_idx)}
        for dx_name, evs in sorted(by_dx.items()):
            sevs = sorted(evs)
            if not (sevs[0][0] == 0 and sevs[0][1] == "疑似"):
                continue
            trans = [e for e in sevs if e[0] >= 1 and e[1] in ("确诊", "排除")]
            tday, tst = trans[0] if trans else (None, None)
            k_star, obs_indices = _find_compiler_update_doc_idx(j, df_docs, dx_name, tday, tst)
            for idx_f in obs_indices:
                pos = p_pos_map[idx_f]
                idx_v = pv_idx[pos]
                df = df_docs[idx_f]
                dv = dv_docs[idx_v]
                dy = int(df["day_offset"])
                txt_f = df["text"]
                txt_v = dv["text"]

                feat_f = _mask_dx_clauses(txt_f, dx_name, replace_token="该病", max_chars=300)
                feat_v = _mask_dx_clauses(txt_v, dx_name, replace_token="该病", max_chars=300)
                if with_context:
                    feat_f = _prior_context_prefix(df_docs, idx_f) + " " + feat_f
                    feat_v = _prior_context_prefix(dv_docs, idx_v) + " " + feat_v

                sil_f = dx_name not in txt_f
                sil_v = dx_name not in txt_v
                y_plan = 1 if (k_star is not None and idx_f == k_star) else 0
                lbl_f = None if sil_f else _clean_anchor_label(txt_f, dx_name)
                lbl_v = None if sil_v else _clean_anchor_label(txt_v, dx_name)
                y_tr_f = 1 if lbl_f in (0, 2) else 0
                y_tr_v = 1 if lbl_v in (0, 2) else 0

                if y_plan == 1:
                    kstar_tot += 1
                    kstar_hits["Full"] += y_tr_f
                    kstar_hits["VisibilityOnly"] += y_tr_v

                cells["Synth-Full | fixed_planner_tk"].append((feat_f, y_plan, sil_f, aid, dx_name, dy, idx_f))
                cells["Synth-Full | fixed_text_rule"].append((feat_f, y_tr_f, sil_f, aid, dx_name, dy, idx_f))
                cells["Synth-VisibilityOnly | fixed_planner_tk"].append((feat_v, y_plan, sil_v, aid, dx_name, dy, idx_v))
                cells["Synth-VisibilityOnly | fixed_text_rule"].append((feat_v, y_tr_v, sil_v, aid, dx_name, dy, idx_v))

    n_windows = len(cells["Synth-Full | fixed_planner_tk"])
    assert all(len(v) == n_windows for v in cells.values()), "FIXED_FRAME_LENGTH_MISMATCH"
    frame_audit = {
        "n_paired_stays": len(paired_ids),
        "n_schedule_aligned_stays": aligned_stays,
        "misaligned_stays_n": misaligned_stays,
        "misaligned_rate": round(misaligned_stays / max(1, len(paired_ids)), 4),
        "n_common_windows": n_windows,
        "kstar_total_transitions": kstar_tot,
        "kstar_text_rule_hit_rate": {
            arm: round(hits / max(1, kstar_tot), 4) for arm, hits in kstar_hits.items()
        } if kstar_tot else {},
        "cells": {
            c: {
                "n_windows": len(rows),
                "n_pos": int(sum(r[1] for r in rows)),
                "n_silent_pos": int(sum(1 for r in rows if r[1] == 1 and r[2])),
            }
            for c, rows in cells.items()
        },
    }
    return cells, frame_audit


def build_t2a_fixed_frame_2x2(full_adm_paths, full_j_paths, vis_adm_paths, vis_j_paths):
    """Build strictly shared-frame 2x2 T2a datasets on aligned clauses valid under both planner and text_rule."""
    full_by_id = {str(a["admission_id"]): a for a in _read_many_jsonl(full_adm_paths)}
    vis_by_id = {str(a["admission_id"]): a for a in _read_many_jsonl(vis_adm_paths)}
    jmap = {str(j["journey_id"]): j for j in (_read_many_jsonl(full_j_paths) + _read_many_jsonl(vis_j_paths))}
    paired_ids = sorted(set(full_by_id) & set(vis_by_id) & set(jmap))

    cell_names = (
        "Synth-Full | fixed_planner_tk",
        "Synth-Full | fixed_text_rule",
        "Synth-VisibilityOnly | fixed_planner_tk",
        "Synth-VisibilityOnly | fixed_text_rule",
    )
    cells = {c: [] for c in cell_names}
    for aid in paired_ids:
        af = full_by_id[aid]
        av = vis_by_id[aid]
        j = jmap[aid]
        c3 = j["patient"]["primary_dx"].get("code3") or ""
        if any(c3.startswith(px) for px in T2_EXCLUDE_PREFIXES):
            continue
        by_dx = collections.defaultdict(list)
        for e in (j.get("dx_events") or ()):
            if len(e.get("name") or "") >= 2:
                by_dx[e["name"]].append((int(e["day"]), e["state"]))
        df_docs, dv_docs = af["docs"], av["docs"]
        if len(df_docs) != len(dv_docs):
            continue
        for doc_idx, (df, dv) in enumerate(zip(df_docs, dv_docs)):
            if df.get("day_offset") != dv.get("day_offset") or df.get("role") != dv.get("role"):
                continue
            if _is_hr(df) or _is_discharge(df) or df.get("day_offset") is None:
                continue
            txt_f, txt_v = df["text"], dv["text"]
            for dx_name, d_evs in sorted(by_dx.items()):
                if dx_name not in txt_f or dx_name not in txt_v:
                    continue
                st = _dx_state_at_doc_idx(j, df_docs, dx_name, doc_idx, d_evs)
                if st is None:
                    continue
                lf = _clean_anchor_label(txt_f, dx_name)
                lv = _clean_anchor_label(txt_v, dx_name)
                if lf is None or lv is None:
                    continue
                y_plan = 2 if st == "确诊" else (0 if st == "排除" else 1)
                mf = _mask_dx_clauses(txt_f, dx_name)
                mv = _mask_dx_clauses(txt_v, dx_name)
                cells["Synth-Full | fixed_planner_tk"].append((mf, y_plan, aid, dx_name))
                cells["Synth-Full | fixed_text_rule"].append((mf, lf, aid, dx_name))
                cells["Synth-VisibilityOnly | fixed_planner_tk"].append((mv, y_plan, aid, dx_name))
                cells["Synth-VisibilityOnly | fixed_text_rule"].append((mv, lv, aid, dx_name))
    return cells


def build_blind_window_eval_rows(fold_path, sealed_key_json, with_context=False):
    """Build full-coverage 210-window inference rows from T2_SEALED_SAMPLING_KEY without silver-settled filtering."""
    with open(sealed_key_json, encoding="utf-8") as f:
        key_obj = json.load(f)
    key_rows = key_obj["mapping"]
    seen_bids = {km["blind_id"] for km in key_rows}
    seen_wkeys = {km["window_key"] for km in key_rows}
    if len(seen_bids) != len(key_rows) or len(seen_wkeys) != len(key_rows):
        raise ValueError("DUPLICATE_BLIND_KEY_DETECTED")

    adms = {str(a["admission_id"]): a for a in _read_jsonl(fold_path)}
    rows = []
    for km in key_rows:
        aid = str(km["admission_id"])
        doc_idx = int(km["doc_idx"])
        dx_name = km["dx_name"]
        wkey = km["window_key"]
        if f"{aid}||{doc_idx}||{dx_name}" != wkey:
            raise ValueError(f"WINDOW_KEY_FORMAT_MISMATCH: {wkey}")
        all_docs = adms[aid]["docs"]
        d = all_docs[doc_idx]
        txt = d["text"]
        dy = int(d["day_offset"])
        is_silent = dx_name not in txt
        feat = _mask_dx_clauses(txt, dx_name, replace_token="该病", max_chars=300)
        if with_context:
            feat = _prior_context_prefix(all_docs, doc_idx) + " " + feat
        rows.append((feat, 0, is_silent, aid, dx_name, dy, doc_idx))
    return rows, key_rows


def export_t2_blind_pool(fold_path, out_target, seed=20261201):
    """Export 6-stratum exhaustive de-identified T2 blind sheet without exposing auto_stratum or predictions."""
    adms = _read_jsonl(fold_path)
    buckets = collections.defaultdict(list)
    for a in adms:
        aid = str(a["admission_id"])
        all_docs = a["docs"]
        cands = _extract_real_t2_candidates(all_docs)
        if not cands:
            continue
        p_indices = [
            idx for idx, d in enumerate(all_docs)
            if not _is_hr(d) and not _is_discharge(d) and d.get("day_offset") is not None
        ]
        by_day = collections.defaultdict(list)
        for idx in p_indices:
            by_day[int(all_docs[idx]["day_offset"])].append(idx)

        for dx_name, is_q in cands:
            settled = False
            for doc_idx in p_indices:
                d = all_docs[doc_idx]
                txt = d["text"]
                dy = int(d["day_offset"])
                day_list = by_day[dy]
                intra_pos = f"第{dy}天 第{day_list.index(doc_idx) + 1}/{len(day_list)}篇"
                cum_docs = [
                    {
                        "k": ci,
                        "day_offset": cd.get("day_offset"),
                        "doc_type": cd.get("doc_type"),
                        "text": cd.get("text", ""),
                    }
                    for ci, cd in enumerate(all_docs[:doc_idx + 1])
                    if not _is_discharge(cd)
                ]
                win_obj = {
                    "window_key": f"{aid}||{doc_idx}||{dx_name}",
                    "admission_id": aid,
                    "doc_idx": doc_idx,
                    "day_offset": dy,
                    "intra_day_pos": intra_pos,
                    "target_dx": dx_name,
                    "cumulative_context": cum_docs,
                }
                if dy == 0:
                    continue
                if settled:
                    if is_q:
                        buckets["post_silver_settled"].append({**win_obj, "auto_stratum": "post_silver_settled"})
                    continue
                if dx_name not in txt:
                    if is_q:
                        buckets["silent_in_riskset"].append({**win_obj, "auto_stratum": "silent_in_riskset"})
                    continue
                lbl = _clean_anchor_label(txt, dx_name)
                if lbl is None:
                    buckets["mentioned_unanchored"].append({**win_obj, "auto_stratum": "mentioned_unanchored"})
                else:
                    st = {0: "silver_excluded", 1: "silver_differential", 2: "silver_confirmed"}[lbl]
                    buckets[st].append({**win_obj, "auto_stratum": st})
                    if is_q and lbl in (0, 2):
                        settled = True

    rng = random.Random(seed)
    targets = (
        ("silver_confirmed", 40),
        ("silver_excluded", 30),
        ("silver_differential", 35),
        ("mentioned_unanchored", 45),
        ("silent_in_riskset", 40),
        ("post_silver_settled", 20),
    )
    pool = []
    strata_meta = {}
    for k, want in targets:
        items = list(buckets[k])
        rng.shuffle(items)
        sel = items[:min(want, len(items))]
        N_h, n_h = len(items), len(sel)
        strata_meta[k] = {
            "N_h": N_h,
            "n_h": n_h,
            "sampling_prob": round(n_h / max(1, N_h), 6),
            "ht_weight": round(N_h / max(1, n_h), 4),
        }
        for it in sel:
            pool.append((it, N_h, n_h))
    rng.shuffle(pool)

    if out_target.endswith(".jsonl") or out_target.endswith(".json"):
        sheet_p = out_target
        key_p = os.path.splitext(out_target)[0] + "_SEALED_KEY.json"
        out_dir = os.path.dirname(out_target)
    else:
        out_dir = out_target
        sheet_p = os.path.join(out_dir, "t2_clinician_blind_sheet.jsonl")
        key_p = os.path.join(out_dir, "T2_SEALED_SAMPLING_KEY.json")
    os.makedirs(out_dir, exist_ok=True)

    key_rows = []
    with open(sheet_p, "w", encoding="utf-8") as f:
        for idx, (item, N_h, n_h) in enumerate(pool):
            bid = f"T2BLIND-{idx:04d}"
            f.write(json.dumps({
                "blind_id": bid,
                "day_offset": item["day_offset"],
                "intra_day_pos": item["intra_day_pos"],
                "target_dx": item["target_dx"],
                "cumulative_context": item["cumulative_context"],
                "q_should_update_evidence": None,
                "q_expressed_state": None,
            }, ensure_ascii=False) + "\n")
            key_rows.append({
                "blind_id": bid,
                "window_key": item["window_key"],
                "admission_id": item["admission_id"],
                "doc_idx": item["doc_idx"],
                "dx_name": item["target_dx"],
                "auto_stratum": item["auto_stratum"],
                "N_h": N_h,
                "n_h": n_h,
                "ht_weight": round(N_h / max(1, n_h), 4),
            })
    with open(key_p, "w", encoding="utf-8") as kf:
        json.dump({"seed": seed, "total_exported": len(key_rows), "strata": strata_meta, "mapping": key_rows}, kf, ensure_ascii=False, indent=2)
    return {"total_exported": len(key_rows), "strata": strata_meta, "sheet_path": sheet_p, "key_path": key_p}


def _np_binary_curve(y_true, y_score, sample_weight=None):
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(ys, kind="mergesort")[::-1]
    ys_s = ys[order]
    yt_s = yt[order]
    w_s = np.asarray(sample_weight, dtype=np.float64)[order] if sample_weight is not None else np.ones_like(yt_s)
    distinct = np.where(np.diff(ys_s))[0]
    thr_idxs = np.r_[distinct, yt_s.size - 1]
    tps = np.cumsum(yt_s * w_s, dtype=np.float64)[thr_idxs]
    fps = np.cumsum((1.0 - yt_s) * w_s, dtype=np.float64)[thr_idxs]
    return fps, tps


def _np_weighted_roc_auc(y_true, y_score, sample_weight=None):
    yt = np.asarray(y_true)
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return None
    fps, tps = _np_binary_curve(yt, y_score, sample_weight=sample_weight)
    if tps[-1] <= 0 or fps[-1] <= 0:
        return None
    tps_r = np.r_[0.0, tps] / tps[-1]
    fps_r = np.r_[0.0, fps] / fps[-1]
    return float(np.sum(np.diff(fps_r) * (tps_r[:-1] + tps_r[1:]) * 0.5))


def _np_weighted_ap(y_true, y_score, sample_weight=None):
    yt = np.asarray(y_true)
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return None
    fps, tps = _np_binary_curve(yt, y_score, sample_weight=sample_weight)
    if tps[-1] <= 0:
        return 0.0
    denom = tps + fps
    prec = np.divide(tps, denom, out=np.zeros_like(tps), where=denom > 0)
    rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def score_t2_human_gold_multiarm(arm_preds_map, gold_sheet_jsonl, sealed_key_json):
    """Strict full-coverage bijection + identical common-valid-set Horvitz-Thompson scorer across arms."""
    gold_by_bid = {r["blind_id"]: r for r in _read_jsonl(gold_sheet_jsonl)}
    with open(sealed_key_json, encoding="utf-8") as f:
        key_rows = json.load(f)["mapping"]

    expected_wkeys = [km["window_key"] for km in key_rows]
    expected_wkey_set = set(expected_wkeys)
    expected_bid_set = {km["blind_id"] for km in key_rows}
    if len(expected_wkey_set) != len(key_rows) or len(expected_bid_set) != len(key_rows):
        raise ValueError("SEALED_KEY_DUPLICATE_KEYS")

    for arm_nm, (prob_d, bin_d) in arm_preds_map.items():
        if set(prob_d.keys()) != expected_wkey_set or set(bin_d.keys()) != expected_wkey_set:
            missing_n = len(expected_wkey_set - set(prob_d.keys()))
            extra_n = len(set(prob_d.keys()) - expected_wkey_set)
            raise ValueError(f"BLIND_KEY_MISMATCH({arm_nm}): missing={missing_n}, extra={extra_n}")

    field_specs = (
        (
            "q_expressed_state",
            {"confirmed", "excluded"},
            {"differential", "silent", "unmentioned", "none", "no_update"},
            {"timing_unclear", "insufficient_info", "indeterminate", "unclear"},
        ),
        (
            "q_should_update_evidence",
            {"confirm", "exclude", "confirmed", "excluded"},
            {"keep_differential", "differential", "no", "no_update", "none"},
            {"timing_unclear", "insufficient_info", "indeterminate", "unclear"},
        ),
    )

    out = {
        "n_sealed_windows": len(key_rows),
        "arms_compared": list(arm_preds_map.keys()),
        "exact_bijection_verified": True,
        "targets": {},
    }
    for target_field, pos_vals, neg_vals, indet_vals in field_specs:
        strata_stats = collections.OrderedDict()
        valid_rows = []
        indet_rows = []
        missing_count = 0

        for km in key_rows:
            st = km["auto_stratum"]
            if st not in strata_stats:
                strata_stats[st] = {
                    "n_sampled": 0,
                    "n_predicted": 0,
                    "n_annotated_nonempty": 0,
                    "n_valid_pos": 0,
                    "n_valid_neg": 0,
                    "n_indeterminate": 0,
                    "n_missing": 0,
                    "n_scored": 0,
                }
            ss = strata_stats[st]
            ss["n_sampled"] += 1
            if km["window_key"] in expected_wkey_set and all(km["window_key"] in pd for pd, _ in arm_preds_map.values()):
                ss["n_predicted"] += 1

            gr = gold_by_bid.get(km["blind_id"])
            raw_v = gr.get(target_field) if gr else None
            norm_v = str(raw_v).strip().lower() if raw_v not in (None, "") else ""
            if not norm_v:
                ss["n_missing"] += 1
                missing_count += 1
                continue
            ss["n_annotated_nonempty"] += 1
            if norm_v in pos_vals:
                ss["n_valid_pos"] += 1
                ss["n_scored"] += 1
                valid_rows.append((km, 1))
            elif norm_v in neg_vals:
                ss["n_valid_neg"] += 1
                ss["n_scored"] += 1
                valid_rows.append((km, 0))
            elif norm_v in indet_vals:
                ss["n_indeterminate"] += 1
                indet_rows.append((km, None))
            else:
                raise ValueError(f"UNKNOWN_CLINICIAN_CODE({target_field}): {raw_v}")

        arm_metrics = {}
        for arm_nm, (prob_d, bin_d) in arm_preds_map.items():
            if not valid_rows:
                arm_metrics[arm_nm] = {"n_scored": 0}
                continue
            y_true, y_prob, w_vec = [], [], []
            tp_w = fp_w = tn_w = fn_w = 0.0
            for km, yt in valid_rows:
                wkey = km["window_key"]
                yp = float(prob_d[wkey])
                yh = int(bin_d[wkey])
                w = float(km["ht_weight"])
                y_true.append(yt)
                y_prob.append(yp)
                w_vec.append(w)
                if yt == 1 and yh == 1:
                    tp_w += w
                elif yt == 0 and yh == 1:
                    fp_w += w
                elif yt == 0 and yh == 0:
                    tn_w += w
                else:
                    fn_w += w
            prec_w = tp_w / max(1e-12, tp_w + fp_w)
            rec_w = tp_w / max(1e-12, tp_w + fn_w)
            f1_w = 2.0 * prec_w * rec_w / max(1e-12, prec_w + rec_w)
            auc_w = _np_weighted_roc_auc(y_true, y_prob, sample_weight=w_vec)
            ap_w = _np_weighted_ap(y_true, y_prob, sample_weight=w_vec)

            bounds = {}
            if indet_rows:
                for bname, indet_y in (("all_indet_as_neg", 0), ("all_indet_as_pos", 1)):
                    by = list(y_true) + [indet_y] * len(indet_rows)
                    bp = list(y_prob) + [float(prob_d[km["window_key"]]) for km, _ in indet_rows]
                    bw = list(w_vec) + [float(km["ht_weight"]) for km, _ in indet_rows]
                    bau = _np_weighted_roc_auc(by, bp, sample_weight=bw)
                    bap = _np_weighted_ap(by, bp, sample_weight=bw)
                    bounds[bname] = {
                        "n": len(by),
                        "auroc": round(float(bau), 4) if bau is not None else None,
                        "auprc": round(float(bap), 4) if bap is not None else None,
                    }

            arm_metrics[arm_nm] = {
                "n_scored": len(y_true),
                "n_pos_raw": int(sum(y_true)),
                "weighted_confusion": {"TP": round(tp_w, 2), "FP": round(fp_w, 2), "TN": round(tn_w, 2), "FN": round(fn_w, 2)},
                "design_weighted_precision": round(prec_w, 4),
                "design_weighted_recall": round(rec_w, 4),
                "design_weighted_f1": round(f1_w, 4),
                "design_weighted_auroc": round(float(auc_w), 4) if auc_w is not None else None,
                "design_weighted_auprc": round(float(ap_w), 4) if ap_w is not None else None,
                "indeterminate_bounds": bounds,
            }

        out["targets"][target_field] = {
            "common_valid_n": len(valid_rows),
            "n_indeterminate_excluded_from_point": len(indet_rows),
            "n_missing": missing_count,
            "per_stratum_counts": strata_stats,
            "arms": arm_metrics,
        }
    return out


def score_t2_human_gold(pred_probs_by_key, pred_bin_by_key, gold_sheet_jsonl, sealed_key_json):
    """Single-arm wrapper around strict full-coverage score_t2_human_gold_multiarm."""
    multi = score_t2_human_gold_multiarm({"single_arm": (pred_probs_by_key, pred_bin_by_key)}, gold_sheet_jsonl, sealed_key_json)
    reports = {}
    for target_field, tinfo in multi["targets"].items():
        arm_rep = tinfo["arms"]["single_arm"]
        reports[target_field] = {
            "n_annotated": arm_rep.get("n_scored", 0),
            **arm_rep,
            "n_indeterminate": tinfo["n_indeterminate_excluded_from_point"],
            "n_missing": tinfo["n_missing"],
        }
    return reports


def score_t1_human_audit(gold_sheet_jsonl, sealed_key_json):
    """Score bidirectional 69-stay T1 clinician audit sheet without coercing indeterminate/missing."""
    gold_by_bid = {r["blind_id"]: r for r in _read_jsonl(gold_sheet_jsonl)}
    with open(sealed_key_json, encoding="utf-8") as f:
        key_rows = json.load(f)["mapping"]
    per_stratum = collections.OrderedDict()
    for km in key_rows:
        st = km["auto_stratum"]
        if st not in per_stratum:
            per_stratum[st] = {
                "N_h": km.get("N_h"),
                "n_h": km.get("n_h"),
                "ht_weight": km.get("ht_weight"),
                "early_plan_yes": 0,
                "early_plan_no": 0,
                "late_op_yes": 0,
                "late_op_no": 0,
                "missing_or_indet": 0,
            }
        gr = gold_by_bid.get(km["blind_id"])
        if not gr or gr.get("c_early_already_planned_0or1") in (None, "", "unclear") or gr.get("c_has_late_op_ge2_0or1") in (None, "", "unclear"):
            per_stratum[st]["missing_or_indet"] += 1
            continue
        ep = int(gr["c_early_already_planned_0or1"])
        lo = int(gr["c_has_late_op_ge2_0or1"])
        per_stratum[st]["early_plan_yes" if ep == 1 else "early_plan_no"] += 1
        per_stratum[st]["late_op_yes" if lo == 1 else "late_op_no"] += 1
    return {"total_stays": len(key_rows), "per_stratum": per_stratum}


def _read_many_jsonl(paths):
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for p in paths or ():
        if p and os.path.exists(p):
            rows.extend(_read_jsonl(p))
    return rows


def _eval_single_event_on_stay(adm, dx, ed, st):
    """Evaluate one planned diagnostic transition on one admission stay.

    Returns (is_evaluable, status_reason, missed_01, silent_01).
    """
    if adm is None:
        return False, "stay_not_survived", 0, 0
    notes = [d for d in adm["docs"] if not _is_hr(d) and not _is_discharge(d) and d.get("day_offset") is not None]
    post_notes = [d for d in notes if int(d["day_offset"]) >= ed]
    if not post_notes:
        return False, "no_post_event_progress_note", 0, 0
    updated = False
    any_mention = False
    for d in post_notes:
        if dx not in d["text"]:
            continue
        any_mention = True
        for clause in re.split(r"[。；;\n]+", d["text"]):
            if dx not in clause:
                continue
            for m in re.finditer(re.escape(dx), clause):
                fs = oracle2.firm_state(clause, m.start(), m.end())
                if fs is not None and fs[1] == st:
                    updated = True
                    break
            if (
                not updated
                and st == "排除"
                and T2_EXCL_RE.search(clause)
                and not T2_NOT_EXCL_RE.search(clause)
            ):
                updated = True
            if updated:
                break
        if updated:
            break
    missed = int(not updated)
    silent = int(not updated and not any_mention)
    return True, "evaluable", missed, silent


def audit_epistemic_asymmetry(vis_adm_paths, vis_j_paths, full_adm_paths, full_j_paths, planned_j_paths=None, n_boot=2000, seed=20261201):
    """Quantify post-evidence missed updates with strict 4-bucket event-level accounting and Manski bounds."""
    vis_by_id = {str(a["admission_id"]): a for a in _read_many_jsonl(vis_adm_paths)}
    full_by_id = {str(a["admission_id"]): a for a in _read_many_jsonl(full_adm_paths)}
    jmap = {str(j["journey_id"]): j for j in (_read_many_jsonl(vis_j_paths) + _read_many_jsonl(full_j_paths))}
    pj_map = {str(j["journey_id"]): j for j in _read_many_jsonl(planned_j_paths)} if planned_j_paths else {}
    universe_j = pj_map if pj_map else jmap

    paired_ids = sorted(set(vis_by_id) & set(full_by_id) & set(universe_j))
    empty_bkt = lambda: {
        "both_evaluable": 0,
        "full_only_evaluable": 0,
        "full_only_vis_reason": collections.Counter(),
        "full_only_full_missed": 0,
        "vis_only_evaluable": 0,
        "vis_only_full_reason": collections.Counter(),
        "vis_only_vis_missed": 0,
        "neither_evaluable": 0,
        "neither_stay_discarded_any": 0,
        "neither_both_survived_no_postnote": 0,
        "total_planned_events": 0,
    }
    bkts = {"确诊": empty_bkt(), "排除": empty_bkt()}
    common_by_stay = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # tuple layout per aid: [v_cm, v_ct, v_cs, f_cm, f_ct, f_cs, v_em, v_et, v_es, f_em, f_et, f_es]
    arm_all_eval = {
        "Visibility-Only": {"确诊": [0, 0, 0], "排除": [0, 0, 0]},
        "Explicit-State": {"确诊": [0, 0, 0], "排除": [0, 0, 0]},
    }
    surv_no_post_cons = {
        "Visibility-Only": {"确诊": [0, 0, 0], "排除": [0, 0, 0]},
        "Explicit-State": {"确诊": [0, 0, 0], "排除": [0, 0, 0]},
    }

    for jid in sorted(universe_j):
        j = universe_j[jid]
        adm_v = vis_by_id.get(jid)
        adm_f = full_by_id.get(jid)
        for e in (j.get("dx_events") or ()):
            st = e.get("state")
            ed = int(e.get("day", 0))
            dx = e.get("name") or ""
            if ed < 1 or st not in ("确诊", "排除") or len(dx) < 2:
                continue
            bk = bkts[st]
            bk["total_planned_events"] += 1

            ev_v, rsn_v, mv, sv = _eval_single_event_on_stay(adm_v, dx, ed, st)
            ev_f, rsn_f, mf, sf = _eval_single_event_on_stay(adm_f, dx, ed, st)

            if ev_v:
                arm_all_eval["Visibility-Only"][st][0] += mv
                arm_all_eval["Visibility-Only"][st][1] += 1
                arm_all_eval["Visibility-Only"][st][2] += sv
                surv_no_post_cons["Visibility-Only"][st][0] += mv
                surv_no_post_cons["Visibility-Only"][st][1] += 1
            elif rsn_v == "no_post_event_progress_note":
                surv_no_post_cons["Visibility-Only"][st][0] += 1
                surv_no_post_cons["Visibility-Only"][st][1] += 1
            else:
                surv_no_post_cons["Visibility-Only"][st][2] += 1

            if ev_f:
                arm_all_eval["Explicit-State"][st][0] += mf
                arm_all_eval["Explicit-State"][st][1] += 1
                arm_all_eval["Explicit-State"][st][2] += sf
                surv_no_post_cons["Explicit-State"][st][0] += mf
                surv_no_post_cons["Explicit-State"][st][1] += 1
            elif rsn_f == "no_post_event_progress_note":
                surv_no_post_cons["Explicit-State"][st][0] += 1
                surv_no_post_cons["Explicit-State"][st][1] += 1
            else:
                surv_no_post_cons["Explicit-State"][st][2] += 1

            if ev_v and ev_f:
                bk["both_evaluable"] += 1
                row = common_by_stay[jid]
                if st == "确诊":
                    row[0] += mv
                    row[1] += 1
                    row[2] += sv
                    row[3] += mf
                    row[4] += 1
                    row[5] += sf
                else:
                    row[6] += mv
                    row[7] += 1
                    row[8] += sv
                    row[9] += mf
                    row[10] += 1
                    row[11] += sf
            elif ev_f and not ev_v:
                bk["full_only_evaluable"] += 1
                bk["full_only_vis_reason"][rsn_v] += 1
                bk["full_only_full_missed"] += mf
            elif ev_v and not ev_f:
                bk["vis_only_evaluable"] += 1
                bk["vis_only_full_reason"][rsn_f] += 1
                bk["vis_only_vis_missed"] += mv
            else:
                bk["neither_evaluable"] += 1
                if rsn_v == "no_post_event_progress_note" and rsn_f == "no_post_event_progress_note":
                    bk["neither_both_survived_no_postnote"] += 1
                else:
                    bk["neither_stay_discarded_any"] += 1

    v_cm = v_ct = v_cs = f_cm = f_ct = f_cs = 0
    v_em = v_et = v_es = f_em = f_et = f_es = 0
    common_stay_ids = sorted(common_by_stay.keys())
    for jid in common_stay_ids:
        r = common_by_stay[jid]
        v_cm += r[0]; v_ct += r[1]; v_cs += r[2]
        f_cm += r[3]; f_ct += r[4]; f_cs += r[5]
        v_em += r[6]; v_et += r[7]; v_es += r[8]
        f_em += r[9]; f_et += r[10]; f_es += r[11]

    out = {
        "Visibility-Only": {
            "kept_stays": len(vis_by_id),
            "common_paired_stays": len(common_stay_ids),
            "confirmed_missed": v_cm,
            "confirmed_total": v_ct,
            "confirmed_miss_rate": round(v_cm / max(1, v_ct), 4),
            "confirmed_silent": v_cs,
            "excluded_missed": v_em,
            "excluded_total": v_et,
            "excluded_miss_rate": round(v_em / max(1, v_et), 4),
            "excluded_silent": v_es,
            "all_evaluable_on_arm": {
                "confirmed_missed": arm_all_eval["Visibility-Only"]["确诊"][0],
                "confirmed_total": arm_all_eval["Visibility-Only"]["确诊"][1],
                "confirmed_miss_rate": round(arm_all_eval["Visibility-Only"]["确诊"][0] / max(1, arm_all_eval["Visibility-Only"]["确诊"][1]), 4),
                "excluded_missed": arm_all_eval["Visibility-Only"]["排除"][0],
                "excluded_total": arm_all_eval["Visibility-Only"]["排除"][1],
                "excluded_miss_rate": round(arm_all_eval["Visibility-Only"]["排除"][0] / max(1, arm_all_eval["Visibility-Only"]["排除"][1]), 4),
            },
        },
        "Explicit-State": {
            "kept_stays": len(full_by_id),
            "common_paired_stays": len(common_stay_ids),
            "confirmed_missed": f_cm,
            "confirmed_total": f_ct,
            "confirmed_miss_rate": round(f_cm / max(1, f_ct), 4),
            "confirmed_silent": f_cs,
            "excluded_missed": f_em,
            "excluded_total": f_et,
            "excluded_miss_rate": round(f_em / max(1, f_et), 4),
            "excluded_silent": f_es,
            "all_evaluable_on_arm": {
                "confirmed_missed": arm_all_eval["Explicit-State"]["确诊"][0],
                "confirmed_total": arm_all_eval["Explicit-State"]["确诊"][1],
                "confirmed_miss_rate": round(arm_all_eval["Explicit-State"]["确诊"][0] / max(1, arm_all_eval["Explicit-State"]["确诊"][1]), 4),
                "excluded_missed": arm_all_eval["Explicit-State"]["排除"][0],
                "excluded_total": arm_all_eval["Explicit-State"]["排除"][1],
                "excluded_miss_rate": round(arm_all_eval["Explicit-State"]["排除"][0] / max(1, arm_all_eval["Explicit-State"]["排除"][1]), 4),
            },
        },
    }

    flow_summary = {}
    for st_cn, st_en in (("确诊", "confirmed"), ("排除", "excluded")):
        b = bkts[st_cn]
        flow_summary[st_en] = {
            "total_planned_events": b["total_planned_events"],
            "both_evaluable": b["both_evaluable"],
            "full_only_evaluable": b["full_only_evaluable"],
            "full_only_vis_reason": dict(b["full_only_vis_reason"]),
            "full_only_full_missed": b["full_only_full_missed"],
            "vis_only_evaluable": b["vis_only_evaluable"],
            "vis_only_full_reason": dict(b["vis_only_full_reason"]),
            "vis_only_vis_missed": b["vis_only_vis_missed"],
            "neither_evaluable": b["neither_evaluable"],
            "neither_stay_discarded_any": b["neither_stay_discarded_any"],
            "neither_both_survived_no_postnote": b["neither_both_survived_no_postnote"],
            "partition_sum_ok": (
                b["both_evaluable"] + b["full_only_evaluable"] + b["vis_only_evaluable"] + b["neither_evaluable"]
                == b["total_planned_events"]
            ),
        }
    out["event_flow_accounting"] = flow_summary

    def _manski_all(st_cn):
        tot = max(1, bkts[st_cn]["total_planned_events"])
        vm, ve, _ = arm_all_eval["Visibility-Only"][st_cn]
        fm, fe, _ = arm_all_eval["Explicit-State"][st_cn]
        v_miss_n = tot - ve
        f_miss_n = tot - fe
        v_lo, v_hi = vm / tot, (vm + v_miss_n) / tot
        f_lo, f_hi = fm / tot, (fm + f_miss_n) / tot
        # Also conservative sensitivity where survived stays lacking post-event progress note count as missed
        v_cm2, _, v_unk2 = surv_no_post_cons["Visibility-Only"][st_cn]
        f_cm2, _, f_unk2 = surv_no_post_cons["Explicit-State"][st_cn]
        return {
            "total_planned_events": tot,
            "manski_gain_bounds": [round(v_lo - f_hi, 4), round(v_hi - f_lo, 4)],
            "vis_miss_bounds": [round(v_lo, 4), round(v_hi, 4)],
            "full_miss_bounds": [round(f_lo, 4), round(f_hi, 4)],
            "conservative_no_postnote_as_miss_manski_bounds": [
                round(v_cm2 / tot - (f_cm2 + f_unk2) / tot, 4),
                round((v_cm2 + v_unk2) / tot - f_cm2 / tot, 4),
            ],
        }

    sens_conf = _manski_all("确诊")
    sens_excl = _manski_all("排除")
    out["all_planned_sensitivity"] = {"confirmed": sens_conf, "excluded": sens_excl}

    if common_stay_ids:
        rng = np.random.RandomState(seed)
        rows = [common_by_stay[jid] for jid in common_stay_ids]
        c_gain = v_cm / max(1, v_ct) - f_cm / max(1, f_ct)
        e_gain = v_em / max(1, v_et) - f_em / max(1, f_et)

        cg_b, eg_b, int_b = [], [], []
        n_p = len(rows)
        for _ in range(n_boot):
            bi = rng.choice(n_p, size=n_p, replace=True)
            b_ct = sum(rows[i][1] for i in bi)
            b_et = sum(rows[i][7] for i in bi)
            if b_ct == 0 or b_et == 0:
                continue
            bc = sum(rows[i][0] for i in bi) / b_ct - sum(rows[i][3] for i in bi) / b_ct
            be = sum(rows[i][6] for i in bi) / b_et - sum(rows[i][9] for i in bi) / b_et
            cg_b.append(bc)
            eg_b.append(be)
            int_b.append(bc - be)

        pi_obj = {
            "paired_stays_n": len(paired_ids),
            "common_event_stays_n": len(common_stay_ids),
            "planned_stays_n": len(universe_j),
            "unpaired_stays_n": max(0, len(universe_j) - len(paired_ids)),
            "confirmed_common_n": v_ct,
            "excluded_common_n": v_et,
            "confirmed_reduction": round(c_gain, 4),
            "confirmed_reduction_ci95": [round(float(np.percentile(cg_b, 2.5)), 4), round(float(np.percentile(cg_b, 97.5)), 4)] if cg_b else None,
            "excluded_reduction": round(e_gain, 4),
            "excluded_reduction_ci95": [round(float(np.percentile(eg_b, 2.5)), 4), round(float(np.percentile(eg_b, 97.5)), 4)] if eg_b else None,
            "interaction_conf_minus_excl": round(c_gain - e_gain, 4),
            "interaction_ci95": [round(float(np.percentile(int_b, 2.5)), 4), round(float(np.percentile(int_b, 97.5)), 4)] if int_b else None,
            "manski_confirmed_gain_bounds": sens_conf["manski_gain_bounds"],
            "manski_excluded_gain_bounds": sens_excl["manski_gain_bounds"],
            "scope_note": "Primary paired estimates use strictly identical common-evaluable event denominators; FirstTry reflects initial drafts within surviving intersections; all planned/unpaired events are bounded in all_planned_sensitivity.",
        }
        out["paired_common_events"] = pi_obj
        out["paired_interaction"] = pi_obj
    return out


def _drug_vector_from_text(text):
    vec = [1 if any(kw in (text or "") for kw in kws) else 0 for _, kws in CANONICAL_DRUG_GROUPS]
    has_dose = 1 if DOSE_RE.search(text or "") else 0
    return vec, has_dose


def build_t3_synth_notes(adm_paths, max_stays=None, seed=20261201):
    """Build note-level multi-label drug-group + has_dose dataset from synthetic admissions."""
    adms, _ = _load_subset_stays(adm_paths, max_stays=max_stays, seed=seed)
    rows = []
    for a in adms:
        aid = a["admission_id"]
        for d in a["docs"]:
            vec, has_dose = _drug_vector_from_text(d["text"])
            rows.append((d["text"], vec, has_dose, aid))
    return rows


def build_t3_llmcare_notes(care_jsonl_path, max_stays=None, seed=20261201):
    """Build note-level T3 rows from single-doc LLM-CARe records or multi-doc AdmissionJSONL."""
    raw_rows = _read_jsonl(care_jsonl_path)
    if max_stays is not None and len(raw_rows) > max_stays:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(raw_rows), size=max_stays, replace=False)
        raw_rows = [raw_rows[i] for i in sorted(idx)]
    rows = []
    for idx, r in enumerate(raw_rows):
        aid = str(r.get("admission_id") or r.get("slot_id") or r.get("rid") or f"CARE-{idx}")
        if r.get("docs"):
            for d in r["docs"]:
                txt = d.get("text") or ""
                vec, has_dose = _drug_vector_from_text(txt)
                rows.append((txt, vec, has_dose, aid))
        else:
            rec = r.get("record")
            if isinstance(rec, dict):
                txt = "\n".join(str(v) for v in rec.values() if v is not None)
            else:
                txt = str(rec or "")
            vec, has_dose = _drug_vector_from_text(txt)
            rows.append((txt, vec, has_dose, aid))
    return rows


def build_t3_s2_notes(fold_paths, s2_dirs, max_stays=None, seed=20261201):
    """Build note-level multi-label drug-group + has_dose dataset from real S2-extracted folds."""
    by_adm_doc = collections.defaultdict(list)
    for s2d in s2_dirs:
        p = os.path.join(s2d, "events_norm.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ev = json.loads(line)
                    if ev.get("event_type") == "medication" and ev.get("assertion") in ("done", "planned"):
                        by_adm_doc[(ev["admission_id"], int(ev.get("doc_idx", 0)))].append(ev)

    adms, _ = _load_subset_stays(fold_paths, max_stays=max_stays, seed=seed)
    rows = []
    for a in adms:
        aid = a["admission_id"]
        for d_idx, d in enumerate(a["docs"]):
            evs = by_adm_doc.get((aid, d_idx), ())
            names = " ".join(str(e.get("name") or "") for e in evs)
            vec = [1 if any(kw in names for kw in kws) else 0 for _, kws in CANONICAL_DRUG_GROUPS]
            has_dose = 1 if any(e.get("value") is not None and str(e.get("value")) != "None" for e in evs) else 0
            rows.append((d["text"], vec, has_dose, aid))
    return rows


def build_t3_gold_eval(gold_dir):
    """Build note-level active medication gold evaluation rows from human-annotated gold_s2_v3."""
    items = _read_jsonl(os.path.join(gold_dir, "items.jsonl"))
    rows = []
    for it in items:
        if it.get("part") == "B":
            evs = [
                e for e in (it.get("events") or ())
                if e.get("event_type") == "medication" and e.get("assertion") in ("done", "planned")
            ]
        else:
            ev = it.get("event") or {}
            evs = [ev] if (ev.get("event_type") == "medication" and ev.get("assertion") in ("done", "planned")) else []
        if not evs and it.get("part") == "A":
            continue
        names = " ".join(str(e.get("name") or "") for e in evs)
        vec = [1 if any(kw in names for kw in kws) else 0 for _, kws in CANONICAL_DRUG_GROUPS]
        has_dose = 1 if any(e.get("value") is not None and str(e.get("value")) != "None" for e in evs) else 0
        rows.append((it["text"], vec, has_dose, it["admission_id"]))
    return rows


def _fit_tfidf_pair(train_txt, tune_txt=None, eval_txt=None):
    min_df = 2 if len(train_txt) >= 30 else 1
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=min_df, max_features=30000)
    X_tr = vec.fit_transform(train_txt)
    X_tu = vec.transform(tune_txt) if tune_txt is not None else None
    X_te = vec.transform(eval_txt) if eval_txt is not None else None
    return vec, X_tr, X_tu, X_te


def _boot_class_indices(y, rng):
    idx = rng.choice(len(y), size=len(y), replace=True)
    if len(np.unique(y[idx])) < 2:
        idx = np.arange(len(y))
    return idx


def fit_t1_bundle(train_rows, tune_rows, seeds=(20261201, 20261202, 20261203)):
    """Fit frozen 3-seed T1a/T1b artifact strictly on training and tune_rows."""
    X_tr_txt = [r[0] for r in train_rows]
    X_tu_txt = [r[0] for r in tune_rows]
    vec, X_tr, X_tu, _ = _fit_tfidf_pair(X_tr_txt, X_tu_txt)
    targets = {}
    train_pos = {}
    for target_idx, label in ((1, "T1a"), (2, "T1b")):
        y_tr = np.array([r[target_idx] for r in train_rows], dtype=int)
        y_tu = np.array([r[target_idx] for r in tune_rows], dtype=int)
        train_pos[label] = int(np.sum(y_tr))
        if len(np.unique(y_tr)) < 2:
            targets[label] = {"single_class": True}
            continue
        best_c, best_auc = 1.0, -1.0
        if X_tu is not None and len(np.unique(y_tu)) >= 2:
            for c_val in (0.1, 0.3, 1.0, 3.0):
                clf = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced", random_state=seeds[0])
                clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_tu, clf.predict_proba(X_tu)[:, 1])
                if auc > best_auc:
                    best_auc, best_c = auc, c_val
        clfs = []
        for s in seeds:
            rng = np.random.RandomState(s)
            boot_idx = _boot_class_indices(y_tr, rng)
            clf = LogisticRegression(C=best_c, max_iter=1000, class_weight="balanced", random_state=s)
            clf.fit(X_tr[boot_idx], y_tr[boot_idx])
            clfs.append(clf)
        targets[label] = {
            "single_class": False,
            "clfs": clfs,
            "best_C": best_c,
            "tune_auroc": round(float(best_auc), 4) if best_auc >= 0 else None,
        }
    return {"kind": "t1_single", "vec": vec, "train_n": len(train_rows), "train_pos": train_pos, "targets": targets}


def predict_t1_bundle(artifact, eval_rows):
    """Read-only prediction for frozen T1 single-arm artifact (zero .fit calls)."""
    X_te = artifact["vec"].transform([r[0] for r in eval_rows])
    preds = {}
    for label in ("T1a", "T1b"):
        info = artifact["targets"][label]
        if info.get("single_class"):
            preds[label] = None
        else:
            sp = [clf.predict_proba(X_te)[:, 1] for clf in info["clfs"]]
            preds[label] = (np.mean(sp, axis=0), sp)
    return preds


def score_t1_bundle(artifact, preds, eval_rows, seeds=(20261201, 20261202, 20261203), n_boot=2000):
    g_te = [r[3] for r in eval_rows]
    out = {"train_n": artifact["train_n"]}
    for target_idx, label in ((1, "T1a"), (2, "T1b")):
        out[f"{label}_train_pos"] = artifact["train_pos"][label]
        y_te = np.array([r[target_idx] for r in eval_rows], dtype=int)
        pred_pair = preds.get(label)
        if pred_pair is None or len(np.unique(y_te)) < 2:
            out[label] = {"auroc": None, "note": "single_class"}
            continue
        mean_p, seed_probs = pred_pair
        res = _cluster_bootstrap_binary(y_te, mean_p, g_te, n_boot=n_boot, seed=seeds[0])
        info = artifact["targets"][label]
        res["best_C"] = info["best_C"]
        res["tune_auroc"] = info["tune_auroc"]
        res["seed_aurocs"] = [round(float(roc_auc_score(y_te, p)), 4) for p in seed_probs]
        out[label] = res
    return out


def run_t1_eval(train_rows, tune_rows, eval_rows, seeds=(20261201, 20261202, 20261203), n_boot=2000, return_preds=False):
    art = fit_t1_bundle(train_rows, tune_rows, seeds=seeds)
    preds = predict_t1_bundle(art, eval_rows)
    res = score_t1_bundle(art, preds, eval_rows, seeds=seeds, n_boot=n_boot)
    return (res, preds) if return_preds else res


def fit_t1_augment_bundle(real_rows, synth_rows, tune_rows, seeds=(20261201, 20261202, 20261203)):
    """Fit frozen domain-weighted T1 fusion bundle strictly on training and tune_rows."""
    if not synth_rows:
        return fit_t1_bundle(real_rows, tune_rows, seeds=seeds)
    if not real_rows:
        return fit_t1_bundle(synth_rows, tune_rows, seeds=seeds)

    r_txt = [r[0] for r in real_rows]
    s_txt = [r[0] for r in synth_rows]
    tu_txt = [r[0] for r in tune_rows]
    vec_r, X_r, X_tu_r, _ = _fit_tfidf_pair(r_txt, tu_txt)
    vec_s, X_s, X_tu_s, _ = _fit_tfidf_pair(s_txt, tu_txt)
    vec_j, X_j, X_tu_j, _ = _fit_tfidf_pair(r_txt + s_txt, tu_txt)
    n_r, n_s = len(real_rows), len(synth_rows)

    targets = {}
    r_pos, s_pos, j_pos = {}, {}, {}
    for target_idx, label in ((1, "T1a"), (2, "T1b")):
        y_r = np.array([r[target_idx] for r in real_rows], dtype=int)
        y_s = np.array([r[target_idx] for r in synth_rows], dtype=int)
        y_j = np.concatenate([y_r, y_s])
        y_tu = np.array([r[target_idx] for r in tune_rows], dtype=int)
        r_pos[label] = int(np.sum(y_r))
        s_pos[label] = int(np.sum(y_s))
        j_pos[label] = int(np.sum(y_j))
        if len(np.unique(y_r)) < 2:
            targets[label] = {"single_class": True}
            continue

        has_synth_cls = len(np.unique(y_s)) >= 2
        best_c_r, best_auc_r, p_r_tu = 1.0, -1.0, None
        best_c_s, best_auc_s, p_s_tu = 1.0, -1.0, None
        has_tune_cls = X_tu_r is not None and len(np.unique(y_tu)) >= 2
        if has_tune_cls:
            for c_val in (0.1, 0.3, 1.0, 3.0):
                clf_r = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced", random_state=seeds[0])
                clf_r.fit(X_r, y_r)
                pr = clf_r.predict_proba(X_tu_r)[:, 1]
                auc_r = roc_auc_score(y_tu, pr)
                if auc_r > best_auc_r:
                    best_auc_r, best_c_r, p_r_tu = auc_r, c_val, pr
                if has_synth_cls:
                    clf_s = LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced", random_state=seeds[0])
                    clf_s.fit(X_s, y_s)
                    ps = clf_s.predict_proba(X_tu_s)[:, 1]
                    auc_s = roc_auc_score(y_tu, ps)
                    if auc_s > best_auc_s:
                        best_auc_s, best_c_s, p_s_tu = auc_s, c_val, ps

        best_cfg = ("prob_blend", 0.0, best_c_r)
        best_auc = best_auc_r
        n_cands = 4
        if has_tune_cls and has_synth_cls and p_r_tu is not None and p_s_tu is not None:
            for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
                n_cands += 1
                auc = roc_auc_score(y_tu, (1.0 - alpha) * p_r_tu + alpha * p_s_tu)
                if auc > best_auc + 1e-12:
                    best_auc = auc
                    best_cfg = ("prob_blend", alpha, (best_c_r, best_c_s))
            for w_s in (0.05, 0.10, 0.20, 0.35):
                sw = np.concatenate([np.ones(n_r, dtype=float), np.full(n_s, w_s, dtype=float)])
                for c_j in (0.1, 0.3, 1.0, 3.0):
                    n_cands += 1
                    clf_j = LogisticRegression(C=c_j, max_iter=1000, class_weight="balanced", random_state=seeds[0])
                    clf_j.fit(X_j, y_j, sample_weight=sw)
                    auc = roc_auc_score(y_tu, clf_j.predict_proba(X_tu_j)[:, 1])
                    if auc > best_auc + 1e-12:
                        best_auc = auc
                        best_cfg = ("weighted_joint", w_s, c_j)

        strat, weight, c_param = best_cfg
        models = []
        for s in seeds:
            rng = np.random.RandomState(s)
            r_idx = _boot_class_indices(y_r, rng)
            if strat == "prob_blend":
                c_r = c_param[0] if isinstance(c_param, tuple) else c_param
                clf_r = LogisticRegression(C=c_r, max_iter=1000, class_weight="balanced", random_state=s)
                clf_r.fit(X_r[r_idx], y_r[r_idx])
                clf_s = None
                if weight > 0.0 and has_synth_cls:
                    s_idx = _boot_class_indices(y_s, rng)
                    clf_s = LogisticRegression(C=c_param[1], max_iter=1000, class_weight="balanced", random_state=s)
                    clf_s.fit(X_s[s_idx], y_s[s_idx])
                models.append((clf_r, clf_s))
            else:
                s_idx = _boot_class_indices(y_s, rng) if has_synth_cls else np.arange(n_s)
                j_idx = np.concatenate([r_idx, n_r + s_idx])
                sw_boot = np.concatenate([np.ones(len(r_idx), dtype=float), np.full(len(s_idx), weight, dtype=float)])
                clf_j = LogisticRegression(C=c_param, max_iter=1000, class_weight="balanced", random_state=s)
                clf_j.fit(X_j[j_idx], y_j[j_idx], sample_weight=sw_boot)
                models.append((clf_j,))
        targets[label] = {
            "single_class": False,
            "strategy": strat,
            "weight": float(weight),
            "c_param": list(c_param) if isinstance(c_param, tuple) else c_param,
            "tune_auroc": round(float(best_auc), 4) if best_auc >= 0 else None,
            "real_only_tune_auroc": round(float(best_auc_r), 4) if best_auc_r >= 0 else None,
            "tune_candidates_tested": n_cands,
            "contribution_note": "no_synthetic_gain_weight_zero" if float(weight) == 0.0 else "positive_tune_gain",
            "models": models,
        }
    return {
        "kind": "t1_augment",
        "vec_r": vec_r,
        "vec_s": vec_s,
        "vec_j": vec_j,
        "real_train_n": n_r,
        "synth_train_n": n_s,
        "train_n": n_r + n_s,
        "r_pos": r_pos,
        "s_pos": s_pos,
        "j_pos": j_pos,
        "targets": targets,
    }


def predict_t1_augment_bundle(artifact, eval_rows):
    if artifact.get("kind") == "t1_single":
        return predict_t1_bundle(artifact, eval_rows)
    te_txt = [r[0] for r in eval_rows]
    X_te_r = artifact["vec_r"].transform(te_txt)
    X_te_s = artifact["vec_s"].transform(te_txt)
    X_te_j = artifact["vec_j"].transform(te_txt)
    preds = {}
    for label in ("T1a", "T1b"):
        info = artifact["targets"][label]
        if info.get("single_class"):
            preds[label] = None
            continue
        sp = []
        w = info["weight"]
        if info["strategy"] == "prob_blend":
            for clf_r, clf_s in info["models"]:
                p = clf_r.predict_proba(X_te_r)[:, 1]
                if w > 0.0 and clf_s is not None:
                    p = (1.0 - w) * p + w * clf_s.predict_proba(X_te_s)[:, 1]
                sp.append(p)
        else:
            for (clf_j,) in info["models"]:
                sp.append(clf_j.predict_proba(X_te_j)[:, 1])
        preds[label] = (np.mean(sp, axis=0), sp)
    return preds


def score_t1_augment_bundle(artifact, preds, eval_rows, seeds=(20261201, 20261202, 20261203), n_boot=2000):
    if artifact.get("kind") == "t1_single":
        return score_t1_bundle(artifact, preds, eval_rows, seeds=seeds, n_boot=n_boot)
    g_te = [r[3] for r in eval_rows]
    out = {
        "real_train_n": artifact["real_train_n"],
        "synth_train_n": artifact["synth_train_n"],
        "train_n": artifact["train_n"],
    }
    for target_idx, label in ((1, "T1a"), (2, "T1b")):
        out[f"{label}_real_train_pos"] = artifact["r_pos"][label]
        out[f"{label}_synth_train_pos"] = artifact["s_pos"][label]
        out[f"{label}_train_pos"] = artifact["j_pos"][label]
        y_te = np.array([r[target_idx] for r in eval_rows], dtype=int)
        pred_pair = preds.get(label)
        if pred_pair is None or len(np.unique(y_te)) < 2:
            out[label] = {"auroc": None, "note": "single_class"}
            continue
        mean_p, seed_probs = pred_pair
        res = _cluster_bootstrap_binary(y_te, mean_p, g_te, n_boot=n_boot, seed=seeds[0])
        info = artifact["targets"][label]
        res["selected_strategy"] = info["strategy"]
        res["selected_weight"] = info["weight"]
        res["best_C"] = info["c_param"]
        res["tune_auroc"] = info["tune_auroc"]
        res["real_only_tune_auroc"] = info["real_only_tune_auroc"]
        res["tune_candidates_tested"] = info["tune_candidates_tested"]
        res["contribution_note"] = info["contribution_note"]
        res["real_train_n"] = artifact["real_train_n"]
        res["synth_train_n"] = artifact["synth_train_n"]
        res["seed_aurocs"] = [round(float(roc_auc_score(y_te, p)), 4) for p in seed_probs]
        out[label] = res
    return out


def run_t1_augment_eval(
    real_rows,
    synth_rows,
    tune_rows,
    eval_rows,
    seeds=(20261201, 20261202, 20261203),
    n_boot=2000,
    return_preds=False,
):
    """Domain-weighted fusion tuned strictly on tune_rows."""
    art = fit_t1_augment_bundle(real_rows, synth_rows, tune_rows, seeds=seeds)
    preds = predict_t1_augment_bundle(art, eval_rows)
    res = score_t1_augment_bundle(art, preds, eval_rows, seeds=seeds, n_boot=n_boot)
    return (res, preds) if return_preds else res


def fit_t2a_bundle(train_rows, seeds=(20261201, 20261202, 20261203)):
    if not train_rows:
        return {"kind": "t2a", "empty": True, "n_train": 0}
    y_tr = np.array([r[1] for r in train_rows], dtype=int)
    dist = {str(k): v for k, v in sorted(collections.Counter(y_tr.tolist()).items())}
    all_classes = np.unique(y_tr)
    if len(all_classes) < 2:
        return {"kind": "t2a", "single_class": True, "n_train": len(train_rows), "train_dist": dist, "classes": all_classes.tolist()}
    vec, X_tr, _, _ = _fit_tfidf_pair([r[0] for r in train_rows])
    clfs = []
    for s in seeds:
        rng = np.random.RandomState(s)
        bi = rng.choice(len(y_tr), size=len(y_tr), replace=True)
        if len(np.unique(y_tr[bi])) < len(all_classes):
            bi = np.arange(len(y_tr))
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=s)
        clf.fit(X_tr[bi], y_tr[bi])
        clfs.append(clf)
    return {"kind": "t2a", "single_class": False, "empty": False, "n_train": len(train_rows), "train_dist": dist, "vec": vec, "clfs": clfs, "classes": all_classes.tolist()}


def predict_t2a_bundle(artifact, val_rows):
    if artifact.get("empty") or artifact.get("single_class") or not val_rows:
        return None
    X_te = artifact["vec"].transform([r[0] for r in val_rows])
    prob_sum = np.zeros((len(val_rows), 3), dtype=float)
    for clf in artifact["clfs"]:
        p = clf.predict_proba(X_te)
        for col_idx, cls_val in enumerate(clf.classes_):
            c_i = int(cls_val)
            if 0 <= c_i < 3:
                prob_sum[:, c_i] += p[:, col_idx]
    probs = prob_sum / max(1, len(artifact["clfs"]))
    canonical = np.array([0, 1, 2], dtype=int)
    return canonical[np.argmax(probs, axis=1)], probs


def score_t2a_bundle(artifact, pred_pair, val_rows):
    if artifact.get("empty") or not val_rows:
        return {"n_train": artifact.get("n_train", 0), "n_val": len(val_rows), "macro_f1_3class": None}
    y_te = np.array([r[1] for r in val_rows], dtype=int)
    if artifact.get("single_class") or pred_pair is None:
        return {"n_train": artifact["n_train"], "n_val": len(val_rows), "macro_f1_3class": None, "note": "single_class_train"}
    pred, _ = pred_pair
    f1s = f1_score(y_te, pred, labels=[0, 1, 2], average=None, zero_division=0)
    present = sorted(set(int(k) for k in artifact["train_dist"]) | set(y_te.tolist()))
    return {
        "n_train": artifact["n_train"],
        "n_val": len(val_rows),
        "train_dist": artifact["train_dist"],
        "val_dist": {str(k): v for k, v in sorted(collections.Counter(y_te.tolist()).items())},
        "macro_f1_3class": round(float(f1_score(y_te, pred, labels=present, average="macro", zero_division=0)), 4),
        "f1_excluded_0": round(float(f1s[0]), 4),
        "f1_diff_1": round(float(f1s[1]), 4),
        "f1_conf_2": round(float(f1s[2]), 4),
        "val_premature_conf_rate": round(float(np.mean(pred[y_te == 1] == 2)), 4) if np.any(y_te == 1) else None,
        "val_missed_update_rate": round(float(np.mean(pred[y_te != 1] == 1)), 4) if np.any(y_te != 1) else None,
    }


def run_t2a_eval(train_rows, val_rows, seeds=(20261201, 20261202, 20261203), return_preds=False, seed=None):
    """Evaluate T2a masked clause 3-class prediction with 3-seed stay ensemble."""
    if seed is not None:
        seeds = (seed, seed + 1, seed + 2)
    art = fit_t2a_bundle(train_rows, seeds=seeds)
    pred_pair = predict_t2a_bundle(art, val_rows)
    res = score_t2a_bundle(art, pred_pair, val_rows)
    return (res, pred_pair) if return_preds else res


def fit_t2b_bundle(train_rows, tune_rows=None, seeds=(20261201, 20261202, 20261203)):
    if not train_rows:
        return {"kind": "t2b", "empty": True, "n_train": 0}
    y_tr = np.array([r[1] for r in train_rows], dtype=int)
    t_pos = int(np.sum(y_tr))
    t_sil_pos = int(sum(1 for r in train_rows if r[1] == 1 and r[2]))
    if len(np.unique(y_tr)) < 2:
        return {"kind": "t2b", "empty": False, "single_class": True, "n_train": len(train_rows), "train_pos": t_pos, "train_silent_pos": t_sil_pos}

    vec, X_tr, _, _ = _fit_tfidf_pair([r[0] for r in train_rows])
    clfs = []
    for s in seeds:
        rng = np.random.RandomState(s)
        bi = _boot_class_indices(y_tr, rng)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=s)
        clf.fit(X_tr[bi], y_tr[bi])
        clfs.append(clf)

    cal_thr = 0.5
    cal_src = "default_0.5"
    if tune_rows and len(tune_rows) >= 10:
        y_cal = np.array([r[1] for r in tune_rows], dtype=int)
        if len(np.unique(y_cal)) >= 2:
            X_cal = vec.transform([r[0] for r in tune_rows])
            p_cal = np.mean([c.predict_proba(X_cal)[:, 1] for c in clfs], axis=0)
            best_f = -1.0
            for t in (0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65):
                f = f1_score(y_cal, (p_cal >= t).astype(int), zero_division=0)
                if f > best_f:
                    best_f, cal_thr = f, t
            cal_src = "tune_fold"
    elif len(np.unique([r[3] for r in train_rows])) >= 5:
        groups = np.array([r[3] for r in train_rows])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seeds[0])
        tr_i, cal_i = next(gss.split(X_tr, y_tr, groups=groups))
        if len(np.unique(y_tr[tr_i])) >= 2 and len(np.unique(y_tr[cal_i])) >= 2:
            c_cal = LogisticRegression(C=1.0, max_iter=1000, random_state=seeds[0])
            c_cal.fit(X_tr[tr_i], y_tr[tr_i])
            p_cal = c_cal.predict_proba(X_tr[cal_i])[:, 1]
            best_f = -1.0
            for t in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
                f = f1_score(y_tr[cal_i], (p_cal >= t).astype(int), zero_division=0)
                if f > best_f:
                    best_f, cal_thr = f, t
            cal_src = "internal_train_split"

    return {
        "kind": "t2b",
        "empty": False,
        "single_class": False,
        "n_train": len(train_rows),
        "train_pos": t_pos,
        "train_silent_pos": t_sil_pos,
        "vec": vec,
        "clfs": clfs,
        "thr_cal": float(cal_thr),
        "thr_cal_source": cal_src,
    }


def predict_t2b_bundle(artifact, val_rows):
    if artifact.get("empty") or artifact.get("single_class") or not val_rows:
        return None
    X_te = artifact["vec"].transform([r[0] for r in val_rows])
    return np.mean([c.predict_proba(X_te)[:, 1] for c in artifact["clfs"]], axis=0)


def score_t2b_bundle(artifact, probs, val_rows, seeds=(20261201, 20261202, 20261203), n_boot=2000):
    if artifact.get("empty") or not val_rows:
        return {"n_train": artifact.get("n_train", 0), "n_val": len(val_rows), "auroc": None}
    y_te = np.array([r[1] for r in val_rows], dtype=int)
    sil_te = np.array([bool(r[2]) for r in val_rows], dtype=bool)
    non_sil = ~sil_te
    if artifact.get("single_class") or probs is None or len(np.unique(y_te)) < 2:
        return {
            "n_train": artifact["n_train"], "n_val": len(val_rows),
            "train_pos": artifact.get("train_pos", 0), "val_pos": int(np.sum(y_te)),
            "val_mentioned": int(np.sum(non_sil)),
            "auroc": None, "mentioned_only_auroc": None, "note": "single_class",
        }
    g_all = [r[3] for r in val_rows]
    g_ment = [r[3] for i, r in enumerate(val_rows) if non_sil[i]]
    full_boot = _cluster_bootstrap_binary(y_te, probs, g_all, n_boot=n_boot, seed=seeds[0])

    has_m2 = int(np.sum(non_sil)) > 0 and len(np.unique(y_te[non_sil])) >= 2
    ment_boot = _cluster_bootstrap_binary(y_te[non_sil], probs[non_sil], g_ment, n_boot=n_boot, seed=seeds[0]) if has_m2 else {}

    p05 = (probs >= 0.5).astype(int)
    t_cal = float(artifact.get("thr_cal", 0.5))
    pcal = (probs >= t_cal).astype(int)
    sil_lens = [len(r[0]) for i, r in enumerate(val_rows) if sil_te[i]]
    men_lens = [len(r[0]) for i, r in enumerate(val_rows) if non_sil[i]]

    return {
        "n_train": artifact["n_train"],
        "train_pos": artifact["train_pos"],
        "train_silent_pos": artifact["train_silent_pos"],
        "n_val": len(val_rows),
        "val_pos": int(np.sum(y_te)),
        "val_silent": int(np.sum(sil_te)),
        "val_mentioned": int(np.sum(non_sil)),
        "silent_mean_chars": round(float(np.mean(sil_lens)), 1) if sil_lens else None,
        "mentioned_mean_chars": round(float(np.mean(men_lens)), 1) if men_lens else None,
        "trivial_mention_only_full_auroc": round(float(roc_auc_score(y_te, non_sil.astype(float))), 4),
        "auroc": full_boot["auroc"],
        "auroc_ci95": full_boot["auroc_ci95"],
        "auprc": full_boot["auprc"],
        "auprc_ci95": full_boot["auprc_ci95"],
        "mentioned_only_auroc": ment_boot.get("auroc"),
        "mentioned_only_auroc_ci95": ment_boot.get("auroc_ci95"),
        "mentioned_only_auprc": ment_boot.get("auprc"),
        "mentioned_only_auprc_ci95": ment_boot.get("auprc_ci95"),
        "mentioned_only_auroc_null": 0.5,
        "mentioned_only_auprc_null": round(float(np.mean(y_te[non_sil])), 4) if int(np.sum(non_sil)) > 0 else None,
        "precision_05": round(float(precision_score(y_te, p05, zero_division=0)), 4),
        "recall_05": round(float(recall_score(y_te, p05, zero_division=0)), 4),
        "f1_05": round(float(f1_score(y_te, p05, zero_division=0)), 4),
        "f1_pos": round(float(f1_score(y_te, p05, zero_division=0)), 4),
        "silent_false_update_rate": round(float(np.mean(p05[sil_te] == 1)), 4) if np.any(sil_te) else None,
        "calibrated_threshold": round(t_cal, 4),
        "thr_cal_source": artifact.get("thr_cal_source"),
        "precision_cal": round(float(precision_score(y_te, pcal, zero_division=0)), 4),
        "recall_cal": round(float(recall_score(y_te, pcal, zero_division=0)), 4),
        "f1_cal": round(float(f1_score(y_te, pcal, zero_division=0)), 4),
        "silent_false_update_rate_cal": round(float(np.mean(pcal[sil_te] == 1)), 4) if np.any(sil_te) else None,
    }


def run_t2b_eval(
    train_rows,
    val_rows,
    tune_rows=None,
    seeds=(20261201, 20261202, 20261203),
    n_boot=2000,
    return_preds=False,
    seed=None,
):
    """Evaluate T2b binary update detection with clean inputs, dual null baselines, and frozen threshold."""
    if seed is not None:
        seeds = (seed, seed + 1, seed + 2)
    art = fit_t2b_bundle(train_rows, tune_rows=tune_rows, seeds=seeds)
    probs = predict_t2b_bundle(art, val_rows)
    res = score_t2b_bundle(art, probs, val_rows, seeds=seeds, n_boot=n_boot)
    return (res, probs) if return_preds else res


def _calibrate_thresholds(
    probs,
    y_true,
    grid=(0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
):
    """Select per-column F1-maximizing probability thresholds on internal calibration split."""
    thrs = []
    for col in range(y_true.shape[1]):
        yt = y_true[:, col]
        if len(np.unique(yt)) < 2:
            thrs.append(0.65)
            continue
        best_t, best_f = 0.65, -1.0
        for t in grid:
            f = f1_score(yt, (probs[:, col] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thrs.append(best_t)
    return thrs


def fit_t3_bundle(train_rows, seeds=(20261201, 20261202, 20261203)):
    Y_tr = np.array([r[1] for r in train_rows], dtype=int)
    d_tr = np.array([r[2] for r in train_rows], dtype=int)
    groups = np.array([r[3] for r in train_rows])
    vec, X_tr, _, _ = _fit_tfidf_pair([r[0] for r in train_rows])

    active_cols = [c for c in range(Y_tr.shape[1]) if len(np.unique(Y_tr[:, c])) >= 2]
    calibrated_thrs = [0.65] * Y_tr.shape[1]
    drug_models = []
    if active_cols:
        uniq_g = np.unique(groups)
        if len(uniq_g) >= 5:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seeds[0])
            tr_i, cal_i = next(gss.split(X_tr, Y_tr, groups=groups))
            cal_cols = [c for c in active_cols if len(np.unique(Y_tr[tr_i, c])) >= 2]
            if cal_cols:
                cal_clf = OneVsRestClassifier(
                    LogisticRegression(
                        C=1.0, max_iter=800, class_weight="balanced", solver="liblinear", random_state=seeds[0]
                    )
                )
                cal_clf.fit(X_tr[tr_i], Y_tr[tr_i][:, cal_cols])
                thrs_sub = _calibrate_thresholds(cal_clf.predict_proba(X_tr[cal_i]), Y_tr[cal_i][:, cal_cols])
                for j_idx, c in enumerate(cal_cols):
                    calibrated_thrs[c] = thrs_sub[j_idx]

        for s in seeds:
            rng = np.random.RandomState(s)
            bi = rng.choice(len(train_rows), size=len(train_rows), replace=True)
            s_cols = [c for c in active_cols if len(np.unique(Y_tr[bi, c])) >= 2]
            if len(s_cols) != len(active_cols):
                bi = np.arange(len(train_rows))
            clf = OneVsRestClassifier(
                LogisticRegression(
                    C=1.0, max_iter=800, class_weight="balanced", solver="liblinear", random_state=s
                )
            )
            clf.fit(X_tr[bi], Y_tr[bi][:, active_cols])
            drug_models.append(clf)

    d_thr = 0.5
    dose_models = []
    if len(np.unique(d_tr)) >= 2:
        uniq_g = np.unique(groups)
        if len(uniq_g) >= 5:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seeds[0])
            tr_i, cal_i = next(gss.split(X_tr, d_tr, groups=groups))
            if len(np.unique(d_tr[tr_i])) >= 2:
                d_cal = LogisticRegression(C=1.0, max_iter=800, solver="liblinear", random_state=seeds[0])
                d_cal.fit(X_tr[tr_i], d_tr[tr_i])
                p_dc = d_cal.predict_proba(X_tr[cal_i])[:, 1]
                best_acc = -1.0
                for t in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
                    acc = float(np.mean((p_dc >= t).astype(int) == d_tr[cal_i]))
                    if acc > best_acc:
                        best_acc, d_thr = acc, t
        for s in seeds:
            rng = np.random.RandomState(s)
            bi = _boot_class_indices(d_tr, rng)
            d_clf = LogisticRegression(C=1.0, max_iter=800, solver="liblinear", random_state=s)
            d_clf.fit(X_tr[bi], d_tr[bi])
            dose_models.append(d_clf)

    return {
        "kind": "t3",
        "n_train_notes": len(train_rows),
        "n_cols": Y_tr.shape[1],
        "vec": vec,
        "active_cols": active_cols,
        "calibrated_thrs": calibrated_thrs,
        "drug_models": drug_models,
        "d_thr": d_thr,
        "dose_models": dose_models,
    }


def predict_t3_bundle(artifact, gold_rows):
    X_te = artifact["vec"].transform([r[0] for r in gold_rows])
    pred_Y = np.zeros((len(gold_rows), artifact["n_cols"]), dtype=int)
    if artifact["active_cols"] and artifact["drug_models"]:
        p_te = np.mean([m.predict_proba(X_te) for m in artifact["drug_models"]], axis=0)
        for j_idx, c in enumerate(artifact["active_cols"]):
            pred_Y[:, c] = (p_te[:, j_idx] >= artifact["calibrated_thrs"][c]).astype(int)
    pred_dose = None
    if artifact["dose_models"]:
        pd_mean = np.mean([m.predict_proba(X_te)[:, 1] for m in artifact["dose_models"]], axis=0)
        pred_dose = (pd_mean >= artifact["d_thr"]).astype(int)
    return pred_Y, pred_dose


def score_t3_bundle(artifact, pred_pair, gold_rows):
    Y_te = np.array([r[1] for r in gold_rows], dtype=int)
    d_te = np.array([r[2] for r in gold_rows], dtype=int)
    pred_Y, pred_dose = pred_pair
    d_pos = int(np.sum(d_te))
    d_maj = round(float(max(d_pos, len(d_te) - d_pos) / max(1, len(d_te))), 4)
    dose_acc = round(float(np.mean(pred_dose == d_te)), 4) if pred_dose is not None else None
    dose_rec = round(float(recall_score(d_te, pred_dose, zero_division=0)), 4) if (pred_dose is not None and d_pos > 0) else None
    return {
        "n_train_notes": artifact["n_train_notes"],
        "n_gold_notes": len(gold_rows),
        "gold_group_positives": int(np.sum(Y_te)),
        "gold_active_groups": int(np.sum(np.sum(Y_te, axis=0) > 0)),
        "gold_dose_positives": d_pos,
        "gold_dose_majority_baseline_acc": d_maj,
        "drug_micro_f1": round(float(f1_score(Y_te, pred_Y, average="micro", zero_division=0)), 4),
        "drug_macro_f1": round(float(f1_score(Y_te, pred_Y, average="macro", zero_division=0)), 4),
        "has_dose_acc": dose_acc,
        "has_dose_recall": dose_rec,
    }


def run_t3_eval(train_rows, gold_rows, seeds=(20261201, 20261202, 20261203), seed=None):
    """Evaluate T3 with 3-seed balanced learners and internal grouped threshold calibration."""
    if seed is not None:
        seeds = (seed, seed + 1, seed + 2)
    art = fit_t3_bundle(train_rows, seeds=seeds)
    return score_t3_bundle(art, predict_t3_bundle(art, gold_rows), gold_rows)


def _completed_shard_paths(
    s5_root,
    subdir,
    prefix,
    n_shards=4,
    min_kept=100,
    adm_file="admissions.jsonl",
    with_journeys=True,
):
    """Auto-discover shards whose counts.json confirms finished zero-error completion."""
    base = os.path.join(s5_root, "generated", subdir)
    adms, js = [], []
    for i in range(n_shards):
        sdir = os.path.join(base, f"{prefix}{i}")
        cnt_p = os.path.join(sdir, "counts.json")
        adm_p = os.path.join(sdir, adm_file)
        j_p = os.path.join(sdir, "journeys.jsonl")
        if not (os.path.exists(cnt_p) and os.path.exists(adm_p)):
            continue
        try:
            with open(cnt_p, encoding="utf-8") as f:
                c = json.load(f)
        except Exception:
            continue
        if int(c.get("llm_errors", 0)) != 0 or int(c.get("kept", 0)) < min_kept:
            continue
        if with_journeys and not os.path.exists(j_p):
            continue
        adms.append(adm_p)
        if with_journeys:
            js.append(j_p)
    return adms, js


def _matched_s4_shard_pairs(s5_root, adm_file="admissions.jsonl", min_kept=400):
    """Collect trajectory-matched completed (vis_only, full, planned_journeys) shard paths."""
    base = os.path.join(s5_root, "generated", "s5_1")
    va, vj, fa, fj, pj = [], [], [], [], []
    for i in range(4):
        vdir = os.path.join(base, f"vis_only_shard{i}")
        fdir = os.path.join(base, f"full_shard{i}")
        ok = True
        for d in (vdir, fdir):
            cnt_p = os.path.join(d, "counts.json")
            if not (os.path.exists(cnt_p) and os.path.exists(os.path.join(d, adm_file)) and os.path.exists(os.path.join(d, "journeys.jsonl"))):
                ok = False
                break
            try:
                c = json.load(open(cnt_p, encoding="utf-8"))
            except Exception:
                ok = False
                break
            if int(c.get("llm_errors", 0)) != 0 or int(c.get("kept", 0)) < min_kept:
                ok = False
                break
        if ok:
            va.append(os.path.join(vdir, adm_file))
            vj.append(os.path.join(vdir, "journeys.jsonl"))
            fa.append(os.path.join(fdir, adm_file))
            fj.append(os.path.join(fdir, "journeys.jsonl"))
            pp = os.path.join(s5_root, "planned", "s5_1_shards", f"shard_{i}", "journeys.jsonl")
            if os.path.exists(pp):
                pj.append(pp)
    return va, vj, fa, fj, pj


def _resolve_direct32b_paths(s5_root, explicit_jsonl=None):
    if explicit_jsonl:
        return [explicit_jsonl] if os.path.exists(explicit_jsonl) else []
    shard_adms, _ = _completed_shard_paths(
        s5_root, "s5_ext", "direct_32b_shard", n_shards=4, min_kept=50, with_journeys=False
    )
    if shard_adms:
        return shard_adms
    merged = os.path.join(s5_root, "generated", "s5_ext", "direct_32b", "admissions.jsonl")
    return [merged] if os.path.exists(merged) else []


def _resolve_exported_synth_path(s5_root, explicit_jsonl=None):
    if explicit_jsonl:
        return [explicit_jsonl] if os.path.exists(explicit_jsonl) else []
    for sub in ("synth_exported_final2k_v17", "synth_exported_final2k_v15"):
        cand = os.path.join(s5_root, "generated", sub, "admissions.jsonl")
        if os.path.exists(cand):
            return [cand]
    return []


def build_t2a_llmcare_clauses(care_jsonl_path, max_stays=None, seed=20261201):
    """Extract T2a clauses from single-doc LLM-CARe discharge summaries for supplementary comparison."""
    raw_rows = _read_jsonl(care_jsonl_path)
    if max_stays is not None and len(raw_rows) > max_stays:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(raw_rows), size=max_stays, replace=False)
        raw_rows = [raw_rows[i] for i in sorted(idx)]
    samples = []
    for idx, r in enumerate(raw_rows):
        aid = str(r.get("admission_id") or r.get("slot_id") or r.get("rid") or f"CARE-{idx}")
        diag_keys = ("Primary Diagnosis", "Discharge Diagnosis", "Secondary Diagnosis", "出院诊断", "主要诊断")
        rec = r.get("record") or {}
        if isinstance(rec, dict):
            txt = "\n".join(str(rec[k]) for k in rec.keys() if k not in diag_keys and rec.get(k) is not None)
            dx_txt = "\n".join(str(rec.get(k) or "") for k in diag_keys)
        else:
            txt = str(rec or "")
            dx_txt = ""
        cands = []
        for ln in re.split(r"[\n;；,，、]+", dx_txt):
            nm = re.sub(r"^\d+[\.、\s]*|[？?()（）].*$", "", ln).strip()
            if 2 <= len(nm) <= 18:
                cands.append(nm)
        if not cands:
            for s in split_sections(txt):
                if "诊断" in s.name:
                    for ln in s.text.splitlines():
                        nm = re.sub(r"^\d+[\.、\s]*|[？?()（）].*$", "", ln).strip()
                        if 2 <= len(nm) <= 18:
                            cands.append(nm)
        for dx_name in dict.fromkeys(cands[:3]):
            if dx_name in txt:
                lbl = _clean_anchor_label(txt, dx_name)
                if lbl is not None:
                    samples.append((_mask_dx_clauses(txt, dx_name), lbl, aid, dx_name))
    return samples


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 16)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rehearsal_patient_split(fold_path, seed=20261201):
    """Deterministically partition unique fold admissions 50/50 stratified by T1 proxy status."""
    from sklearn.model_selection import StratifiedShuffleSplit
    adms = _read_jsonl(fold_path)
    aids, ys = [], []
    for a in adms:
        aid = str(a["admission_id"])
        docs = a["docs"]
        d2p = "\n".join(d["text"] for d in docs if _in_day_window(d, 2, None) and not _is_discharge(d))
        aids.append(aid)
        ys.append(1 if has_late_op_event(d2p) else 0)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    tr_i, ev_i = next(sss.split(aids, ys))
    return {aids[i] for i in tr_i}, {aids[i] for i in ev_i}


def _filter_rows_aid(rows, aid_set, aid_idx=3):
    return [r for r in rows if str(r[aid_idx]) in aid_set]


def _collect_source_defs(
    args, real_folds_27, s2_dirs_27, direct_paths, exported_paths, fa, fj, ffa, ffj, va, vj, vfa, vfj, oa, oj
):
    pilot_a = [
        "/home/weify/pyt/product/v031_s3/work/a256_w32b_v031/admissions.jsonl",
        "/home/weify/pyt/product/v031_s3/work/p80_w32b_v034/admissions.jsonl",
    ]
    pilot_j = [
        "/home/weify/pyt/product/v031_s3/work/a256_w32b_v031/journeys.jsonl",
        "/home/weify/pyt/product/v031_s3/work/p80_w32b_v034/journeys.jsonl",
    ]
    has_direct = bool(direct_paths)
    has_exported = bool(exported_paths)
    has_care = bool(args.llmcare_jsonl and os.path.exists(args.llmcare_jsonl))
    is_smoke_like = args.mode == "smoke" and not (args.fit_and_freeze and not args.rehearsal_split)

    t1_sources = {
        "Real-Small (250)": (real_folds_27, None, 250, None),
        "Real-Full (2000)": (real_folds_27, None, 2000, None),
    }
    if is_smoke_like:
        t1_sources["Synth v033+v034 Pilot (333)"] = (pilot_a, pilot_j, None, None)
    if fa:
        t1_sources["Synth-Full"] = (fa, fj, None, None)
        t1_sources["Synth-Full (Matched N=600)"] = (fa, fj, None, 600)
    if has_exported:
        t1_sources["Synth-Exported (Release N=901)"] = (exported_paths, fj, None, None)
    if va:
        t1_sources["Synth-VisibilityOnly"] = (va, vj, None, None)
    if oa:
        t1_sources["Synth-OracleLeak"] = (oa, oj, None, None)
    if has_direct:
        t1_sources["DIRECT-32B"] = (direct_paths, None, None, None)

    t2_sources = {
        "Real-Small (250)": (real_folds_27, None, 250, None),
        "Real-Full (2000)": (real_folds_27, None, 2000, None),
    }
    if fa:
        t2_sources["Synth-Full (PostGate)"] = (fa, fj, None, None)
        t2_sources["Synth-Full (Matched N=600)"] = (fa, fj, None, 600)
    if has_exported:
        t2_sources["Synth-Exported (Release N=901)"] = (exported_paths, fj, None, None)
    if ffa:
        t2_sources["Synth-Full (FirstTry)"] = (ffa, ffj, None, None)
    if va:
        t2_sources["Synth-VisibilityOnly (PostGate)"] = (va, vj, None, None)
    if vfa:
        t2_sources["Synth-VisibilityOnly (FirstTry)"] = (vfa, vfj, None, None)
    if oa:
        t2_sources["Synth-OracleLeak"] = (oa, oj, None, None)
    if has_direct:
        t2_sources["DIRECT-32B"] = (direct_paths, None, None, None)

    t3_arms = {
        "Real-Small S2 (250)": build_t3_s2_notes(real_folds_27, s2_dirs_27, max_stays=250),
        "Real-Full S2 (2000)": build_t3_s2_notes(real_folds_27, s2_dirs_27, max_stays=2000),
    }
    if fa:
        t3_arms["Synth-Full"] = build_t3_synth_notes(fa)
        t3_arms["Synth-Full (Matched N=600)"] = build_t3_synth_notes(fa, max_stays=600)
        t3_arms["Augment (Real250+SynthFull)"] = (
            build_t3_s2_notes(real_folds_27, s2_dirs_27, max_stays=250) + build_t3_synth_notes(fa)
        )
    if has_exported:
        t3_arms["Synth-Exported (Release N=901)"] = build_t3_synth_notes(exported_paths)
        t3_arms["Augment (Real250+SynthExported)"] = (
            build_t3_s2_notes(real_folds_27, s2_dirs_27, max_stays=250) + build_t3_synth_notes(exported_paths)
        )
    if va:
        t3_arms["Synth-VisibilityOnly"] = build_t3_synth_notes(va)
    if oa:
        t3_arms["Synth-OracleLeak"] = build_t3_synth_notes(oa)
    if has_direct:
        t3_arms["DIRECT-32B"] = build_t3_synth_notes(direct_paths)
    if has_care:
        t3_arms["LLM-CARe"] = build_t3_llmcare_notes(args.llmcare_jsonl)

    return t1_sources, t2_sources, t3_arms, (pilot_a, pilot_j)


def _compute_paired_t1_diffs(prob_map, eval_rows, target_label, pairs_to_compare, n_boot=2000):
    y_idx = 1 if target_label == "T1a" else 2
    y_te = np.array([r[y_idx] for r in eval_rows], dtype=int)
    g_te = [r[3] for r in eval_rows]
    diffs = {}
    for diff_name, arm_a, arm_b in pairs_to_compare:
        pa = (prob_map.get(arm_a) or {}).get(target_label)
        pb = (prob_map.get(arm_b) or {}).get(target_label)
        if pa is not None and pb is not None:
            diffs[diff_name] = paired_bootstrap_diff(y_te, pa[0], pb[0], g_te, n_boot=n_boot)
    return diffs


def _compute_paired_t2b_ment_diffs(prob_map, val_rows, pairs_to_compare, n_boot=2000):
    sil_te = np.array([bool(r[2]) for r in val_rows], dtype=bool)
    non_sil = ~sil_te
    if not np.any(non_sil):
        return {}
    y_m = np.array([r[1] for r in val_rows], dtype=int)[non_sil]
    g_m = [r[3] for i, r in enumerate(val_rows) if non_sil[i]]
    diffs = {}
    for diff_name, arm_a, arm_b in pairs_to_compare:
        pa = prob_map.get(arm_a)
        pb = prob_map.get(arm_b)
        if pa is not None and pb is not None:
            diffs[diff_name] = paired_bootstrap_diff(y_m, pa[non_sil], pb[non_sil], g_m, n_boot=n_boot)
    return diffs


def _fit_all_artifacts(
    args,
    real_folds_27,
    t1_sources,
    t2_sources,
    t3_arms,
    t1_tune,
    t1_tune_gt2,
    t1_tune_maj,
    t2b_tune,
    fa,
    fj,
    va,
    vj,
    direct_paths,
    exported_paths,
):
    artifacts = {
        "T1": {},
        "T1_gt2_sensitivity": {},
        "T1_major_op_sensitivity": {},
        "T1_low_resource_curve": {},
        "T2a": {},
        "T2b": {},
        "T2_2x2_supervision_ablation": {
            "T2a": {},
            "T2b": {},
            "T2b_dynamic_stop_supplementary": {},
            "fixed_frame_audit": {},
        },
        "T2b_context_sensitivity": {},
        "T2a_supplementary_discharge_clauses": {},
        "T3": {},
    }

    maj_arms = ("Real-Small (250)", "Synth-Full", "Synth-Full (Matched N=600)", "Synth-Exported (Release N=901)", "DIRECT-32B")
    for k, (apths, jpth, mx, mr) in t1_sources.items():
        if not apths:
            continue
        tr_ge2 = build_t1_dataset(apths, jpth, max_n=mx, max_raw_stays=mr, gt2_only=False)
        tr_gt2 = build_t1_dataset(apths, jpth, max_n=mx, max_raw_stays=mr, gt2_only=True)
        tr_maj = build_t1_dataset(apths, jpth, max_n=mx, max_raw_stays=mr, gt2_only=False, major_op_only=True)
        if tr_ge2:
            artifacts["T1"][k] = fit_t1_bundle(tr_ge2, t1_tune)
        if tr_gt2:
            artifacts["T1_gt2_sensitivity"][k] = fit_t1_bundle(tr_gt2, t1_tune_gt2)
        if tr_maj and k in maj_arms:
            artifacts["T1_major_op_sensitivity"][k] = fit_t1_bundle(tr_maj, t1_tune_maj)
    print(f"FIT_PROGRESS T1_arms={len(artifacts['T1'])} low_res_pending=True", flush=True)

    r250_ge2 = build_t1_dataset(real_folds_27, None, max_n=250, gt2_only=False)
    r250_gt2 = build_t1_dataset(real_folds_27, None, max_n=250, gt2_only=True)
    r250_maj = build_t1_dataset(real_folds_27, None, max_n=250, gt2_only=False, major_op_only=True)
    sf_ge2 = build_t1_dataset(fa, fj, gt2_only=False) if fa else []
    sf_gt2 = build_t1_dataset(fa, fj, gt2_only=True) if fa else []
    sf_maj = build_t1_dataset(fa, fj, gt2_only=False, major_op_only=True) if fa else []
    ex_ge2 = build_t1_dataset(exported_paths, fj, gt2_only=False) if exported_paths else []
    ex_gt2 = build_t1_dataset(exported_paths, fj, gt2_only=True) if exported_paths else []
    ex_maj = build_t1_dataset(exported_paths, fj, gt2_only=False, major_op_only=True) if exported_paths else []
    d32_ge2 = build_t1_dataset(direct_paths, None, gt2_only=False) if direct_paths else []
    d32_gt2 = build_t1_dataset(direct_paths, None, gt2_only=True) if direct_paths else []

    if sf_ge2:
        artifacts["T1"]["Augment (Real250+SynthFull)"] = fit_t1_augment_bundle(r250_ge2, sf_ge2, t1_tune)
    if sf_gt2:
        artifacts["T1_gt2_sensitivity"]["Augment (Real250+SynthFull)"] = fit_t1_augment_bundle(r250_gt2, sf_gt2, t1_tune_gt2)
    if sf_maj:
        artifacts["T1_major_op_sensitivity"]["Augment (Real250+SynthFull)"] = fit_t1_augment_bundle(r250_maj, sf_maj, t1_tune_maj)
    if ex_ge2:
        artifacts["T1"]["Augment (Real250+SynthExported)"] = fit_t1_augment_bundle(r250_ge2, ex_ge2, t1_tune)
    if ex_gt2:
        artifacts["T1_gt2_sensitivity"]["Augment (Real250+SynthExported)"] = fit_t1_augment_bundle(r250_gt2, ex_gt2, t1_tune_gt2)
    if ex_maj:
        artifacts["T1_major_op_sensitivity"]["Augment (Real250+SynthExported)"] = fit_t1_augment_bundle(r250_maj, ex_maj, t1_tune_maj)
    if d32_ge2:
        artifacts["T1"]["Augment (Real250+DIRECT32B)"] = fit_t1_augment_bundle(r250_ge2, d32_ge2, t1_tune)
    if d32_gt2:
        artifacts["T1_gt2_sensitivity"]["Augment (Real250+DIRECT32B)"] = fit_t1_augment_bundle(r250_gt2, d32_gt2, t1_tune_gt2)

    if sf_ge2 or d32_ge2 or ex_ge2:
        for n_real in (50, 100, 250):
            r_sub = r250_ge2 if n_real == 250 else build_t1_dataset(real_folds_27, None, max_n=n_real, gt2_only=False)
            artifacts["T1_low_resource_curve"][f"Real-{n_real}"] = fit_t1_bundle(r_sub, t1_tune)
            if sf_ge2:
                artifacts["T1_low_resource_curve"][f"Augment (Real{n_real}+SynthFull)"] = fit_t1_augment_bundle(r_sub, sf_ge2, t1_tune)
            if ex_ge2:
                artifacts["T1_low_resource_curve"][f"Augment (Real{n_real}+SynthExported)"] = fit_t1_augment_bundle(r_sub, ex_ge2, t1_tune)
            if d32_ge2:
                artifacts["T1_low_resource_curve"][f"Augment (Real{n_real}+DIRECT32B)"] = fit_t1_augment_bundle(r_sub, d32_ge2, t1_tune)

    for k, (apths, jpth, mx, mr) in t2_sources.items():
        if apths:
            t2a_r = build_t2a_clause_dataset(apths, jpth, max_stays=mx, max_raw_stays=mr)
            t2b_r = build_t2b_update_dataset(apths, jpth, max_stays=mx, max_raw_stays=mr)
            artifacts["T2a"][k] = fit_t2a_bundle(t2a_r)
            artifacts["T2b"][k] = fit_t2b_bundle(t2b_r, tune_rows=t2b_tune)
    if fa:
        aug_t2a = build_t2a_clause_dataset(real_folds_27, None, max_stays=250) + build_t2a_clause_dataset(fa, fj)
        aug_t2b = build_t2b_update_dataset(real_folds_27, None, max_stays=250) + build_t2b_update_dataset(fa, fj)
        artifacts["T2a"]["Augment (Real250+SynthFull)"] = fit_t2a_bundle(aug_t2a)
        artifacts["T2b"]["Augment (Real250+SynthFull)"] = fit_t2b_bundle(aug_t2b, tune_rows=t2b_tune)
    if exported_paths:
        aug_ex_t2a = build_t2a_clause_dataset(real_folds_27, None, max_stays=250) + build_t2a_clause_dataset(exported_paths, fj)
        aug_ex_t2b = build_t2b_update_dataset(real_folds_27, None, max_stays=250) + build_t2b_update_dataset(exported_paths, fj)
        artifacts["T2a"]["Augment (Real250+SynthExported)"] = fit_t2a_bundle(aug_ex_t2a)
        artifacts["T2b"]["Augment (Real250+SynthExported)"] = fit_t2b_bundle(aug_ex_t2b, tune_rows=t2b_tune)

    if fa and va:
        ff_t2a = build_t2a_fixed_frame_2x2(fa, fj, va, vj)
        for cell_k, cell_rows in ff_t2a.items():
            artifacts["T2_2x2_supervision_ablation"]["T2a"][cell_k] = fit_t2a_bundle(cell_rows)
        ff_t2b, ff_audit = build_t2b_fixed_frame_2x2(fa, fj, va, vj, with_context=False)
        artifacts["T2_2x2_supervision_ablation"]["fixed_frame_audit"] = ff_audit
        for cell_k, cell_rows in ff_t2b.items():
            artifacts["T2_2x2_supervision_ablation"]["T2b"][cell_k] = fit_t2b_bundle(cell_rows, tune_rows=t2b_tune)
        for arm_lbl, ap, jp in (("Synth-Full", fa, fj), ("Synth-VisibilityOnly", va, vj)):
            for lsrc in ("planner", "text_rule"):
                ls_arg = "auto" if lsrc == "planner" else "text_rule"
                key_dyn = f"{arm_lbl} | dynamic_{lsrc}"
                artifacts["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][key_dyn] = fit_t2b_bundle(
                    build_t2b_update_dataset(ap, jp, label_source=ls_arg), tune_rows=t2b_tune
                )

    t2b_ctx_sources = [("Real-Small (250)", real_folds_27, None, 250)]
    if fa:
        t2b_ctx_sources.append(("Synth-Full", fa, fj, None))
    if va:
        t2b_ctx_sources.append(("Synth-VisibilityOnly", va, vj, None))
    if direct_paths:
        t2b_ctx_sources.append(("DIRECT-32B", direct_paths, None, None))
    for k, ap, jp, mx in t2b_ctx_sources:
        artifacts["T2b_context_sensitivity"][k] = fit_t2b_bundle(
            build_t2b_update_dataset(ap, jp, max_stays=mx, with_context=True), tune_rows=t2b_tune
        )

    if args.llmcare_jsonl and os.path.exists(args.llmcare_jsonl):
        care_t2a = build_t2a_llmcare_clauses(args.llmcare_jsonl)
        if care_t2a:
            artifacts["T2a_supplementary_discharge_clauses"]["LLM-CARe"] = fit_t2a_bundle(care_t2a)

    for k, tr in t3_arms.items():
        if tr:
            artifacts["T3"][k] = fit_t3_bundle(tr)
    print(f"FIT_PROGRESS T2b_arms={len(artifacts['T2b'])} T2a_supp={len(artifacts['T2a_supplementary_discharge_clauses'])} T3_arms={len(artifacts['T3'])}", flush=True)

    return artifacts


def _evaluate_artifacts(artifacts, args, target_adms_or_fold, eval_aid_filter, t3_gold_rows, n_boot=2000, t2_key_path=None):
    target_paths = [target_adms_or_fold]
    t1_eval = build_t1_dataset(target_paths, gt2_only=False)
    t1_eval_gt2 = build_t1_dataset(target_paths, gt2_only=True)
    t1_eval_maj = build_t1_dataset(target_paths, gt2_only=False, major_op_only=True)
    t2a_val = build_t2a_clause_dataset(target_paths, max_stays=(1600 if eval_aid_filter is None else None))
    t2b_val = build_t2b_update_dataset(target_paths, max_stays=(1600 if eval_aid_filter is None else None))
    t2b_ctx_val = build_t2b_update_dataset(target_paths, max_stays=(1600 if eval_aid_filter is None else None), with_context=True)

    if eval_aid_filter is not None:
        t1_eval = _filter_rows_aid(t1_eval, eval_aid_filter, 3)
        t1_eval_gt2 = _filter_rows_aid(t1_eval_gt2, eval_aid_filter, 3)
        t1_eval_maj = _filter_rows_aid(t1_eval_maj, eval_aid_filter, 3)
        t2a_val = _filter_rows_aid(t2a_val, eval_aid_filter, 2)
        t2b_val = _filter_rows_aid(t2b_val, eval_aid_filter, 3)
        t2b_ctx_val = _filter_rows_aid(t2b_ctx_val, eval_aid_filter, 3)

    t1_p, t1_maj_p, t2b_p, t2b_ff_p, t2b_dyn_p, t2b_ctx_p = {}, {}, {}, {}, {}, {}
    out = {
        "T1": {},
        "T1_gt2_sensitivity": {},
        "T1_major_op_sensitivity": {},
        "T1_low_resource_curve": {},
        "T2a": {},
        "T2b": {},
        "T2_2x2_supervision_ablation": {
            "T2a": {},
            "T2b": {},
            "T2b_dynamic_stop_supplementary": {},
            "fixed_frame_audit": artifacts["T2_2x2_supervision_ablation"].get("fixed_frame_audit", {}),
        },
        "T2b_context_sensitivity": {},
        "T2a_supplementary_discharge_clauses": {},
        "paired_comparisons": {},
    }

    for k, art in artifacts["T1"].items():
        pr = predict_t1_augment_bundle(art, t1_eval)
        t1_p[k] = pr
        out["T1"][k] = score_t1_augment_bundle(art, pr, t1_eval, n_boot=n_boot)
    for k, art in artifacts["T1_gt2_sensitivity"].items():
        pr = predict_t1_augment_bundle(art, t1_eval_gt2)
        out["T1_gt2_sensitivity"][k] = score_t1_augment_bundle(art, pr, t1_eval_gt2, n_boot=n_boot)
    for k, art in artifacts["T1_major_op_sensitivity"].items():
        pr = predict_t1_augment_bundle(art, t1_eval_maj)
        t1_maj_p[k] = pr
        out["T1_major_op_sensitivity"][k] = score_t1_augment_bundle(art, pr, t1_eval_maj, n_boot=n_boot)

    lr_p = {}
    for k, art in artifacts["T1_low_resource_curve"].items():
        pr = predict_t1_augment_bundle(art, t1_eval)
        lr_p[k] = pr
        out["T1_low_resource_curve"][k] = score_t1_augment_bundle(art, pr, t1_eval, n_boot=n_boot)
    print(f"EVAL_PROGRESS T1_done n_boot={n_boot}", flush=True)

    for k, art in artifacts["T2a"].items():
        out["T2a"][k] = score_t2a_bundle(art, predict_t2a_bundle(art, t2a_val), t2a_val)
    for k, art in artifacts["T2b"].items():
        pr = predict_t2b_bundle(art, t2b_val)
        t2b_p[k] = pr
        out["T2b"][k] = score_t2b_bundle(art, pr, t2b_val, n_boot=n_boot)

    for k, art in artifacts["T2_2x2_supervision_ablation"]["T2a"].items():
        out["T2_2x2_supervision_ablation"]["T2a"][k] = score_t2a_bundle(art, predict_t2a_bundle(art, t2a_val), t2a_val)
    for k, art in artifacts["T2_2x2_supervision_ablation"]["T2b"].items():
        pr = predict_t2b_bundle(art, t2b_val)
        t2b_ff_p[k] = pr
        out["T2_2x2_supervision_ablation"]["T2b"][k] = score_t2b_bundle(art, pr, t2b_val, n_boot=n_boot)
    for k, art in artifacts["T2_2x2_supervision_ablation"].get("T2b_dynamic_stop_supplementary", {}).items():
        pr = predict_t2b_bundle(art, t2b_val)
        t2b_dyn_p[k] = pr
        out["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][k] = score_t2b_bundle(art, pr, t2b_val, n_boot=n_boot)

    for k, art in artifacts["T2b_context_sensitivity"].items():
        pr = predict_t2b_bundle(art, t2b_ctx_val)
        t2b_ctx_p[k] = pr
        out["T2b_context_sensitivity"][k] = score_t2b_bundle(art, pr, t2b_ctx_val, n_boot=n_boot)

    for k, art in artifacts["T2a_supplementary_discharge_clauses"].items():
        out["T2a_supplementary_discharge_clauses"][k] = score_t2a_bundle(art, predict_t2a_bundle(art, t2a_val), t2a_val)

    na_obj = {"status": "N/A_single_doc_no_daily_timeline"}
    if "LLM-CARe" in artifacts.get("T3", {}) or args.mode == "full":
        out["T1"]["LLM-CARe"] = na_obj
        out["T1_gt2_sensitivity"]["LLM-CARe"] = na_obj
        out["T2b"]["LLM-CARe"] = na_obj

    t3_res = {}
    if t3_gold_rows:
        for k, art in artifacts["T3"].items():
            t3_res[k] = score_t3_bundle(art, predict_t3_bundle(art, t3_gold_rows), t3_gold_rows)
    print(f"EVAL_PROGRESS T2_T3_done t2b_arms={len(out['T2b'])}", flush=True)

    out["paired_comparisons"]["T1a"] = _compute_paired_t1_diffs(
        t1_p,
        t1_eval,
        "T1a",
        (
            ("Augment_SynthFull_minus_Real250", "Augment (Real250+SynthFull)", "Real-Small (250)"),
            ("Augment_SynthExported_minus_Real250", "Augment (Real250+SynthExported)", "Real-Small (250)"),
            ("Augment_DIRECT32B_minus_Real250", "Augment (Real250+DIRECT32B)", "Real-Small (250)"),
            ("SynthFull_minus_VisOnly", "Synth-Full", "Synth-VisibilityOnly"),
            ("SynthFull600_minus_DIRECT32B", "Synth-Full (Matched N=600)", "DIRECT-32B"),
            ("SynthExported_minus_DIRECT32B", "Synth-Exported (Release N=901)", "DIRECT-32B"),
            ("SynthFull_minus_SynthExported", "Synth-Full", "Synth-Exported (Release N=901)"),
        ),
        n_boot=n_boot,
    )
    if lr_p:
        for n_r in (50, 100):
            for s_lbl in ("SynthFull", "SynthExported"):
                pa = (lr_p.get(f"Augment (Real{n_r}+{s_lbl})") or {}).get("T1a")
                pb = (lr_p.get(f"Real-{n_r}") or {}).get("T1a")
                if pa is not None and pb is not None:
                    y_te = np.array([r[1] for r in t1_eval], dtype=int)
                    g_te = [r[3] for r in t1_eval]
                    out["paired_comparisons"]["T1a"][f"Augment_Real{n_r}_{s_lbl}_minus_Real{n_r}"] = paired_bootstrap_diff(
                        y_te, pa[0], pb[0], g_te, n_boot=n_boot
                    )

    out["paired_comparisons"]["T1_major_op"] = _compute_paired_t1_diffs(
        t1_maj_p,
        t1_eval_maj,
        "T1a",
        (
            ("Augment_SynthFull_minus_Real250", "Augment (Real250+SynthFull)", "Real-Small (250)"),
            ("Augment_SynthExported_minus_Real250", "Augment (Real250+SynthExported)", "Real-Small (250)"),
            ("SynthFull600_minus_DIRECT32B", "Synth-Full (Matched N=600)", "DIRECT-32B"),
            ("SynthExported_minus_DIRECT32B", "Synth-Exported (Release N=901)", "DIRECT-32B"),
        ),
        n_boot=n_boot,
    )
    out["paired_comparisons"]["T2b_mentioned_only"] = _compute_paired_t2b_ment_diffs(
        t2b_p,
        t2b_val,
        (
            ("SynthFull_PostGate_minus_VisOnly_PostGate", "Synth-Full (PostGate)", "Synth-VisibilityOnly (PostGate)"),
            ("SynthFull_FirstTry_minus_VisOnly_FirstTry", "Synth-Full (FirstTry)", "Synth-VisibilityOnly (FirstTry)"),
            ("SynthFull600_minus_DIRECT32B", "Synth-Full (Matched N=600)", "DIRECT-32B"),
            ("SynthExported_minus_VisOnly_PostGate", "Synth-Exported (Release N=901)", "Synth-VisibilityOnly (PostGate)"),
            ("SynthExported_minus_DIRECT32B", "Synth-Exported (Release N=901)", "DIRECT-32B"),
            ("Augment_SynthExported_minus_Real250", "Augment (Real250+SynthExported)", "Real-Small (250)"),
            ("SynthFull_minus_SynthExported", "Synth-Full (PostGate)", "Synth-Exported (Release N=901)"),
        ),
        n_boot=n_boot,
    )
    out["paired_comparisons"]["T2b_fixed_frame_2x2"] = _compute_paired_t2b_ment_diffs(
        t2b_ff_p,
        t2b_val,
        (
            ("Full_fixed_textrule_minus_VisOnly_fixed_textrule", "Synth-Full | fixed_text_rule", "Synth-VisibilityOnly | fixed_text_rule"),
            ("Full_fixed_planner_minus_Full_fixed_textrule", "Synth-Full | fixed_planner_tk", "Synth-Full | fixed_text_rule"),
            ("Full_fixed_planner_minus_VisOnly_fixed_planner", "Synth-Full | fixed_planner_tk", "Synth-VisibilityOnly | fixed_planner_tk"),
        ),
        n_boot=n_boot,
    )
    if t2b_dyn_p:
        out["paired_comparisons"]["T2b_dynamic_stop_2x2"] = _compute_paired_t2b_ment_diffs(
            t2b_dyn_p,
            t2b_val,
            (
                ("Full_dyn_textrule_minus_VisOnly_dyn_textrule", "Synth-Full | dynamic_text_rule", "Synth-VisibilityOnly | dynamic_text_rule"),
                ("Full_dyn_planner_minus_Full_dyn_textrule", "Synth-Full | dynamic_planner", "Synth-Full | dynamic_text_rule"),
            ),
            n_boot=n_boot,
        )
    out["paired_comparisons"]["T2b_context_mentioned_only"] = _compute_paired_t2b_ment_diffs(
        t2b_ctx_p,
        t2b_ctx_val,
        (("SynthFull_Ctx_minus_VisOnly_Ctx", "Synth-Full", "Synth-VisibilityOnly"),),
        n_boot=n_boot,
    )

    t2_blind_preds = {}
    blind_audit = {"status": "no_sealed_key_provided"}
    if t2_key_path and os.path.exists(t2_key_path) and os.path.exists(target_adms_or_fold):
        blind_rows, blind_key_rows = build_blind_window_eval_rows(target_adms_or_fold, t2_key_path, with_context=False)
        blind_ctx_rows, _ = build_blind_window_eval_rows(target_adms_or_fold, t2_key_path, with_context=True)
        expected_wkeys = [km["window_key"] for km in blind_key_rows]
        per_stratum = dict(collections.Counter(km["auto_stratum"] for km in blind_key_rows))

        for arm_name, art in artifacts["T2b"].items():
            pr = predict_t2b_bundle(art, blind_rows)
            if pr is None or len(pr) != len(expected_wkeys):
                continue
            t_cal = float(art.get("thr_cal", 0.5))
            prob_d = {wkey: float(pr[i]) for i, wkey in enumerate(expected_wkeys)}
            bin_d = {wkey: int(pr[i] >= t_cal) for i, wkey in enumerate(expected_wkeys)}
            if set(prob_d) != set(expected_wkeys):
                raise ValueError(f"BLIND_PRED_BIJECTION_FAILED: {arm_name}")
            t2_blind_preds[arm_name] = (prob_d, bin_d)

        for ctx_name, art in artifacts.get("T2b_context_sensitivity", {}).items():
            pr = predict_t2b_bundle(art, blind_ctx_rows)
            if pr is None or len(pr) != len(expected_wkeys):
                continue
            t_cal = float(art.get("thr_cal", 0.5))
            lbl = f"{ctx_name} [with_context]"
            t2_blind_preds[lbl] = (
                {wkey: float(pr[i]) for i, wkey in enumerate(expected_wkeys)},
                {wkey: int(pr[i] >= t_cal) for i, wkey in enumerate(expected_wkeys)},
            )

        blind_audit = {
            "sealed_key_path": os.path.basename(t2_key_path),
            "n_sealed_windows": len(expected_wkeys),
            "exact_bijection_verified": len(t2_blind_preds) > 0,
            "arms_verified": list(t2_blind_preds.keys()),
            "per_stratum_predicted": per_stratum,
        }

    out["blind_pack_prediction_audit"] = blind_audit
    print("EVAL_PROGRESS paired_done", flush=True)
    return out, t3_res, t2_blind_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=("smoke", "full"))
    ap.add_argument("--fit-and-freeze", default=None, help="Fit and serialize Stage-1 artifacts into bundle_dir")
    ap.add_argument("--rehearsal-split", action="store_true", help="Hold out 50%% fold1 patients for rehearsal eval")
    ap.add_argument("--eval-frozen", default=None, help="Load frozen bundle_dir for strict read-only eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--val-fold", default="/home/weify/pyt/data/deid_run9_fold1.jsonl")
    ap.add_argument("--eval-fold", default=None)
    ap.add_argument("--fold9-gold", default=None)
    ap.add_argument("--export-t2-pool", default=None)
    ap.add_argument("--export-t1-pool", default=None)
    ap.add_argument("--export-consort-ids-private", default=None)
    ap.add_argument("--t2-human-gold", default=None)
    ap.add_argument("--t2-sealed-key", default=None)
    ap.add_argument("--t1-human-gold", default=None)
    ap.add_argument("--t1-sealed-key", default=None)
    ap.add_argument("--exported-synth-jsonl", default=None)
    ap.add_argument("--direct32b-jsonl", default=None)
    ap.add_argument("--llmcare-jsonl", default="/home/weify/pyt/gates/Q3/work/care_s2_768.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.fit_and_freeze or (args.mode == "smoke" and not args.eval_frozen):
        chk_str = f"{args.val_fold} {args.eval_fold or ''} {args.fold9_gold or ''}".lower()
        if "fold9" in chk_str:
            raise ValueError("STAGE1_FORBIDS_FOLD9")

    real_folds_27 = [f"/home/weify/pyt/data/deid_run9_fold{k}.jsonl" for k in range(2, 8)]
    s2_dirs_27 = [f"/home/weify/pyt/product/derived/fold{k}/s2_full_v03" for k in range(2, 8)]
    gold_dir_f0 = "/home/weify/pyt/product/derived/fold0/gold_s2_v3"
    direct_paths = _resolve_direct32b_paths(args.s5_root, args.direct32b_jsonl)
    exported_paths = _resolve_exported_synth_path(args.s5_root, args.exported_synth_jsonl)

    fa, fj = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400)
    ffa, ffj = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400, adm_file="first_try_admissions.jsonl")
    va, vj = _completed_shard_paths(args.s5_root, "s5_1", "vis_only_shard", min_kept=400)
    vfa, vfj = _completed_shard_paths(args.s5_root, "s5_1", "vis_only_shard", min_kept=400, adm_file="first_try_admissions.jsonl")
    oa, oj = _completed_shard_paths(args.s5_root, "s5_1", "oracle_leak_shard", min_kept=400)

    t3_gold_dev = build_t3_gold_eval(gold_dir_f0) if os.path.exists(gold_dir_f0) else []
    t3_gold_f9 = build_t3_gold_eval(args.fold9_gold) if (args.fold9_gold and os.path.exists(args.fold9_gold)) else []

    if args.export_t2_pool and os.path.exists(args.val_fold):
        t2_exp_info = export_t2_blind_pool(args.val_fold, args.export_t2_pool)
        print("T2_BLIND_EXPORT", json.dumps(t2_exp_info, ensure_ascii=False))
    if args.export_t1_pool and os.path.exists(args.val_fold):
        t1_exp_info = export_t1_audit_pool(args.val_fold, args.export_t1_pool)
        print("T1_AUDIT_EXPORT", json.dumps(t1_exp_info, ensure_ascii=False))

    if args.fit_and_freeze:
        os.makedirs(args.fit_and_freeze, exist_ok=True)
        if args.rehearsal_split:
            tune_ids, eval_ids = _rehearsal_patient_split(args.val_fold)
        else:
            tune_ids = {str(a["admission_id"]) for a in _read_jsonl(args.val_fold)}
            eval_ids = None

        t1_tune = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=False), tune_ids, 3)
        t1_tune_gt2 = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=True), tune_ids, 3)
        t1_tune_maj = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=False, major_op_only=True), tune_ids, 3)
        t2b_tune = _filter_rows_aid(build_t2b_update_dataset([args.val_fold]), tune_ids, 3)

        t1_src, t2_src, t3_arms, _ = _collect_source_defs(
            args, real_folds_27, s2_dirs_27, direct_paths, exported_paths, fa, fj, ffa, ffj, va, vj, vfa, vfj, oa, oj
        )
        artifacts = _fit_all_artifacts(
            args, real_folds_27, t1_src, t2_src, t3_arms, t1_tune, t1_tune_gt2, t1_tune_maj, t2b_tune,
            fa, fj, va, vj, direct_paths, exported_paths,
        )
        bundle = {
            "version": "v1.6",
            "rehearsal_split": bool(args.rehearsal_split),
            "tune_fold": args.val_fold,
            "tune_patient_count": len(tune_ids),
            "rehearsal_eval_ids": sorted(eval_ids) if eval_ids is not None else None,
            "expected_t1_keys": sorted(artifacts["T1"].keys()),
            "expected_t2_keys": sorted(artifacts["T2b"].keys()),
            "expected_t3_keys": sorted(artifacts["T3"].keys()),
            "artifacts": artifacts,
        }
        pkl_p = os.path.join(args.fit_and_freeze, "frozen_bundle.pkl")
        with open(pkl_p, "wb") as bf:
            pickle.dump(bundle, bf)
        source_hashes = {}
        for p in [os.path.abspath(__file__), args.val_fold, *fa, *va, *oa, *direct_paths, *exported_paths]:
            if p and os.path.exists(p):
                source_hashes[p] = _sha256_file(p)
        manifest = {
            "version": "v1.6",
            "bundle_sha256": _sha256_file(pkl_p),
            "rehearsal_split": bool(args.rehearsal_split),
            "completed_shards": {"full": len(fa), "vis_only": len(va), "oracle_leak": len(oa), "direct_32b": len(direct_paths), "exported_901": len(exported_paths)},
            "expected_t1_keys": bundle["expected_t1_keys"],
            "expected_t2_keys": bundle["expected_t2_keys"],
            "expected_t3_keys": bundle["expected_t3_keys"],
            "source_hashes": source_hashes,
        }
        man_p = os.path.join(args.fit_and_freeze, "freeze_manifest.json")
        with open(man_p, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        print("FIT_FREEZE_DONE", json.dumps({"bundle_dir": args.fit_and_freeze, "bundle_sha256": manifest["bundle_sha256"]}, ensure_ascii=False))
        return

    if args.eval_frozen:
        if not args.out:
            raise ValueError("--eval-frozen requires --out")
        man_p = os.path.join(args.eval_frozen, "freeze_manifest.json")
        pkl_p = os.path.join(args.eval_frozen, "frozen_bundle.pkl")
        if not (os.path.exists(man_p) and os.path.exists(pkl_p)):
            raise FileNotFoundError(f"Missing frozen bundle or manifest in {args.eval_frozen}")
        with open(man_p, encoding="utf-8") as mf:
            manifest = json.load(mf)
        actual_sha = _sha256_file(pkl_p)
        if actual_sha != manifest["bundle_sha256"]:
            raise RuntimeError(f"FROZEN_BUNDLE_HASH_MISMATCH: expected {manifest['bundle_sha256']}, got {actual_sha}")
        with open(pkl_p, "rb") as bf:
            bundle = pickle.load(bf)
        artifacts = bundle["artifacts"]
        for req_k in manifest.get("expected_t1_keys", ()):
            if req_k not in artifacts["T1"]:
                raise KeyError(f"Missing frozen T1 key: {req_k}")
        for req_k in manifest.get("expected_t2_keys", ()):
            if req_k not in artifacts["T2b"]:
                raise KeyError(f"Missing frozen T2b key: {req_k}")

        eval_target = args.eval_fold or bundle["tune_fold"]
        eval_aid_filter = set(bundle["rehearsal_eval_ids"]) if (bundle.get("rehearsal_eval_ids") and not args.eval_fold) else None
        t3_target = t3_gold_f9 if args.fold9_gold else t3_gold_dev

        t2_key_path = args.t2_sealed_key
        if not t2_key_path and (not args.eval_fold or "fold1" in args.eval_fold.lower()):
            cand_key = os.path.join(args.s5_root, "blind_pools", "t2_fold1_v15", "T2_SEALED_SAMPLING_KEY.json")
            if os.path.exists(cand_key):
                t2_key_path = cand_key

        task_out, t3_eval_res, t2_blind_preds = _evaluate_artifacts(
            artifacts, args, eval_target, eval_aid_filter, t3_target, n_boot=args.n_boot, t2_key_path=t2_key_path
        )

        s4_asym = {}
        m_va, m_vj, m_fa, m_fj, m_pj = _matched_s4_shard_pairs(args.s5_root, "admissions.jsonl")
        if m_va and m_fa:
            s4_asym["Completed_Shards_PostGate"] = audit_epistemic_asymmetry(m_va, m_vj, m_fa, m_fj, planned_j_paths=m_pj, n_boot=args.n_boot)
        m_vfa, m_vfj, m_ffa, m_ffj, m_pfj = _matched_s4_shard_pairs(args.s5_root, "first_try_admissions.jsonl")
        if m_vfa and m_ffa:
            s4_asym["Completed_Shards_FirstTry"] = audit_epistemic_asymmetry(m_vfa, m_vfj, m_ffa, m_ffj, planned_j_paths=m_pfj, n_boot=args.n_boot)

        human_gold_rep = {"status": "pending_clinician_annotation"}
        if args.t2_human_gold and t2_key_path and os.path.exists(args.t2_human_gold) and t2_blind_preds:
            human_gold_rep = score_t2_human_gold_multiarm(t2_blind_preds, args.t2_human_gold, t2_key_path)

        t1_human_rep = {"status": "pending_clinician_annotation"}
        t1_key_path = args.t1_sealed_key
        if not t1_key_path and (not args.eval_fold or "fold1" in args.eval_fold.lower()):
            for sub in ("t1_fold1_v17", "t1_fold1_v15"):
                cand = os.path.join(args.s5_root, "blind_pools", sub, "T1_SEALED_AUDIT_KEY.json")
                if os.path.exists(cand):
                    t1_key_path = cand
                    break
        if args.t1_human_gold and t1_key_path and os.path.exists(args.t1_human_gold) and os.path.exists(t1_key_path):
            t1_human_rep = score_t1_human_audit(args.t1_human_gold, t1_key_path)

        results = {
            "mode": "eval_frozen",
            "version": "v1.6",
            "bundle_dir": args.eval_frozen,
            "bundle_sha256": actual_sha,
            "eval_fold": args.eval_fold or "fold1_rehearsal_split_eval",
            "n_boot": args.n_boot,
            "completed_shards": manifest.get("completed_shards", {}),
            "data_audit": {
                "eval_fold": audit_fold_docs(eval_target, out_private_ids_json=args.export_consort_ids_private)
            } if os.path.exists(eval_target) else {},
            "s4_asymmetry": s4_asym,
            **task_out,
            "T3_gold_dev": t3_eval_res if not args.fold9_gold else {},
            "T3_fold9_ref": t3_eval_res if args.fold9_gold else {"status": "pending_stage2_fold9_unlock"},
            "T2_human_gold": human_gold_rep,
            "T1_human_audit": t1_human_rep,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("EVAL_FROZEN_DONE", json.dumps({"out": args.out, "bundle_sha256": actual_sha}, ensure_ascii=False))
        return

    if not args.out:
        raise ValueError("--mode smoke/full requires --out")

    if args.eval_fold:
        tune_ids = {str(a["admission_id"]) for a in _read_jsonl(args.val_fold)}
        eval_ids = None
        eval_target = args.eval_fold
    else:
        tune_ids, eval_ids = _rehearsal_patient_split(args.val_fold)
        eval_target = args.val_fold

    t1_tune = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=False), tune_ids, 3)
    t1_tune_gt2 = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=True), tune_ids, 3)
    t1_tune_maj = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=False, major_op_only=True), tune_ids, 3)
    t2b_tune = _filter_rows_aid(build_t2b_update_dataset([args.val_fold]), tune_ids, 3)

    t1_src, t2_src, t3_arms, _ = _collect_source_defs(
        args, real_folds_27, s2_dirs_27, direct_paths, exported_paths, fa, fj, ffa, ffj, va, vj, vfa, vfj, oa, oj
    )
    artifacts = _fit_all_artifacts(
        args, real_folds_27, t1_src, t2_src, t3_arms, t1_tune, t1_tune_gt2, t1_tune_maj, t2b_tune,
        fa, fj, va, vj, direct_paths, exported_paths,
    )
    t3_target = t3_gold_f9 if (args.mode == "full" and t3_gold_f9) else t3_gold_dev
    task_out, t3_eval_res, _ = _evaluate_artifacts(artifacts, args, eval_target, eval_ids, t3_target, n_boot=args.n_boot)

    s4_asym = {}
    m_va, m_vj, m_fa, m_fj, m_pj = _matched_s4_shard_pairs(args.s5_root, "admissions.jsonl")
    if m_va and m_fa:
        s4_asym["Completed_Shards_PostGate"] = audit_epistemic_asymmetry(m_va, m_vj, m_fa, m_fj, planned_j_paths=m_pj, n_boot=args.n_boot)
    m_vfa, m_vfj, m_ffa, m_ffj, m_pfj = _matched_s4_shard_pairs(args.s5_root, "first_try_admissions.jsonl")
    if m_vfa and m_ffa:
        s4_asym["Completed_Shards_FirstTry"] = audit_epistemic_asymmetry(m_vfa, m_vfj, m_ffa, m_ffj, planned_j_paths=m_pfj, n_boot=args.n_boot)

    results = {
        "mode": args.mode,
        "version": "v1.6",
        "val_fold": args.val_fold,
        "eval_fold": args.eval_fold or "fold1_split_eval",
        "n_boot": args.n_boot,
        "completed_shards": {"full": len(fa), "vis_only": len(va), "oracle_leak": len(oa), "direct_32b": len(direct_paths), "exported_901": len(exported_paths)},
        "data_audit": {"val_fold": audit_fold_docs(args.val_fold)} if os.path.exists(args.val_fold) else {},
        "s4_asymmetry": s4_asym,
        **task_out,
        "T3_gold_dev": t3_eval_res if not args.fold9_gold else {},
        "T3_fold9_ref": t3_eval_res if args.fold9_gold else {"status": "pending_stage2_fold9_unlock"},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("T123_DONE", json.dumps({"mode": args.mode, "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
