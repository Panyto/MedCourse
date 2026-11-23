# -*- coding: utf-8 -*-
"""Pretrained Chinese BERT encoder student probe (T1, T2a, T2b) with Stage-1 freeze and Stage-2 read-only eval.

Zero sklearn runtime dependency in metric/bootstrap paths so it runs directly inside GPU environments
that ship PyTorch + Transformers without scikit-learn, while sharing identical dataset builders and
patient splits from t123_bench.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import os
import pickle
import random

import numpy as np

from t123_bench import (
    _completed_shard_paths,
    _filter_rows_aid,
    _in_day_window,
    _is_discharge,
    _read_jsonl,
    _resolve_direct32b_paths,
    _resolve_exported_synth_path,
    _sha256_file,
    audit_fold_docs,
    build_blind_window_eval_rows,
    build_t1_dataset,
    build_t2a_clause_dataset,
    build_t2a_fixed_frame_2x2,
    build_t2b_fixed_frame_2x2,
    build_t2b_update_dataset,
    has_late_op_event,
    score_t2_human_gold_multiarm,
)

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    HAS_TORCH_HF = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    HAS_TORCH_HF = False

SEEDS = (20261201, 20261202, 20261203)
DEFAULT_ENCODER_PATH = "/home/weify/pyt/models/bge-small-zh-v1.5"
T2B_THR_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
BLEND_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
PTFT_BETAS = (0.25, 0.50, 0.75, 1.00)


def assert_stage1_fit_allowed(val_fold, eval_fold=None):
    """Enforce zero fold9 contact during Stage-1 fit/freeze or dev rehearsal."""
    chk_str = f"{val_fold or ''} {eval_fold or ''}".lower()
    if "fold9" in chk_str:
        raise ValueError("STAGE1_FORBIDS_FOLD9")


def _binary_clf_curve(y_true, y_score, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_s = y_score[order]
    y_true_s = y_true[order]
    if sample_weight is not None:
        w_s = np.asarray(sample_weight, dtype=np.float64)[order]
    else:
        w_s = np.ones_like(y_true_s, dtype=np.float64)
    distinct = np.where(np.diff(y_score_s))[0]
    thr_idxs = np.r_[distinct, y_true_s.size - 1]
    tps = np.cumsum(y_true_s * w_s, dtype=np.float64)[thr_idxs]
    fps = np.cumsum((1.0 - y_true_s) * w_s, dtype=np.float64)[thr_idxs]
    return fps, tps


def np_roc_auc(y_true, y_score, sample_weight=None):
    """Exact trapezoidal ROC-AUC with tie grouping matching scikit-learn."""
    y_arr = np.asarray(y_true)
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2:
        return None
    fps, tps = _binary_clf_curve(y_arr, y_score, sample_weight=sample_weight)
    if tps[-1] <= 0 or fps[-1] <= 0:
        return None
    tps_r = np.r_[0.0, tps] / tps[-1]
    fps_r = np.r_[0.0, fps] / fps[-1]
    return float(np.sum(np.diff(fps_r) * (tps_r[:-1] + tps_r[1:]) * 0.5))


def np_average_precision(y_true, y_score, sample_weight=None):
    """Exact step-interpolated Average Precision matching scikit-learn."""
    y_arr = np.asarray(y_true)
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2:
        return None
    fps, tps = _binary_clf_curve(y_arr, y_score, sample_weight=sample_weight)
    if tps[-1] <= 0:
        return 0.0
    denom = tps + fps
    precision = np.divide(tps, denom, out=np.zeros_like(tps), where=denom > 0)
    recall = tps / tps[-1]
    delta_r = np.diff(np.r_[0.0, recall])
    return float(np.sum(delta_r * precision))


def np_brier(y_true, y_prob):
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p_arr - y_arr) ** 2))


def _ece_score(y_true, probs, n_bins=10):
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(probs, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(y_arr))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_arr >= lo) & (p_arr <= hi if i == n_bins - 1 else p_arr < hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(y_arr[mask])) - float(np.mean(p_arr[mask])))
    return round(float(ece), 4)


def np_prf1(y_true, y_pred):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def np_f1_per_class(y_true, y_pred, labels=(0, 1, 2)):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    out = []
    for c in labels:
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        denom = 2 * tp + fp + fn
        out.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return np.asarray(out, dtype=np.float64)


def cluster_bootstrap_binary(y, probs, group_ids, n_boot=2000, seed=20261201):
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probs, dtype=np.float64)
    if len(y_arr) == 0 or len(np.unique(y_arr)) < 2:
        return {"auroc": None, "n": int(len(y_arr)), "pos": int(np.sum(y_arr)) if len(y_arr) else 0}
    rng = np.random.RandomState(seed)
    uniq = list(dict.fromkeys(group_ids))
    idx_by_g = collections.defaultdict(list)
    for idx, g in enumerate(group_ids):
        idx_by_g[g].append(idx)
    aucs, aps = [], []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sel = [i for g in sampled for i in idx_by_g[g]]
        yt, yp = y_arr[sel], p_arr[sel]
        if len(np.unique(yt)) < 2:
            continue
        auc_v = np_roc_auc(yt, yp)
        ap_v = np_average_precision(yt, yp)
        if auc_v is not None and ap_v is not None:
            aucs.append(auc_v)
            aps.append(ap_v)
    auc_pt = np_roc_auc(y_arr, p_arr)
    ap_pt = np_average_precision(y_arr, p_arr)
    return {
        "auroc": round(auc_pt, 4) if auc_pt is not None else None,
        "auroc_ci95": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)] if aucs else None,
        "auprc": round(ap_pt, 4) if ap_pt is not None else None,
        "auprc_ci95": [round(float(np.percentile(aps, 2.5)), 4), round(float(np.percentile(aps, 97.5)), 4)] if aps else None,
        "brier": round(np_brier(y_arr, p_arr), 4),
        "ece": _ece_score(y_arr, p_arr),
        "n": int(len(y_arr)),
        "pos": int(np.sum(y_arr)),
    }


def paired_bootstrap_diff(y, probs_a, probs_b, group_ids, n_boot=2000, seed=20261201):
    """Patient cluster-bootstrap paired difference (Arm A minus Arm B)."""
    y_arr = np.asarray(y, dtype=int)
    pa = np.asarray(probs_a, dtype=np.float64)
    pb = np.asarray(probs_b, dtype=np.float64)
    if len(y_arr) == 0 or len(y_arr) != len(pa) or len(y_arr) != len(pb) or len(np.unique(y_arr)) < 2:
        return {"n": int(len(y_arr)), "delta_auroc": None}
    auc_a = np_roc_auc(y_arr, pa)
    auc_b = np_roc_auc(y_arr, pb)
    ap_a = np_average_precision(y_arr, pa)
    ap_b = np_average_precision(y_arr, pb)

    rng = np.random.RandomState(seed)
    uniq = list(dict.fromkeys(group_ids))
    idx_by_g = collections.defaultdict(list)
    for idx, g in enumerate(group_ids):
        idx_by_g[g].append(idx)
    d_aucs, d_aps = [], []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sel = [i for g in sampled for i in idx_by_g[g]]
        yt = y_arr[sel]
        if len(np.unique(yt)) < 2:
            continue
        aa, ab = np_roc_auc(yt, pa[sel]), np_roc_auc(yt, pb[sel])
        pa_s, pb_s = np_average_precision(yt, pa[sel]), np_average_precision(yt, pb[sel])
        if aa is not None and ab is not None and pa_s is not None and pb_s is not None:
            d_aucs.append(aa - ab)
            d_aps.append(pa_s - pb_s)
    p_auc = None
    if d_aucs:
        arr = np.asarray(d_aucs, dtype=np.float64)
        p_auc = round(float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))), 4)
    return {
        "n": int(len(y_arr)),
        "pos": int(np.sum(y_arr)),
        "auroc_a": round(auc_a, 4) if auc_a is not None else None,
        "auroc_b": round(auc_b, 4) if auc_b is not None else None,
        "delta_auroc": round(auc_a - auc_b, 4) if (auc_a is not None and auc_b is not None) else None,
        "delta_auroc_ci95": [round(float(np.percentile(d_aucs, 2.5)), 4), round(float(np.percentile(d_aucs, 97.5)), 4)] if d_aucs else None,
        "auprc_a": round(ap_a, 4) if ap_a is not None else None,
        "auprc_b": round(ap_b, 4) if ap_b is not None else None,
        "delta_auprc": round(ap_a - ap_b, 4) if (ap_a is not None and ap_b is not None) else None,
        "delta_auprc_ci95": [round(float(np.percentile(d_aps, 2.5)), 4), round(float(np.percentile(d_aps, 97.5)), 4)] if d_aps else None,
        "p_boot_two_sided": p_auc,
    }


def _load_or_make_rehearsal_split(val_fold, split_json=None, seed=20261201):
    """Load exact tune/eval patient IDs when provided, else deterministic stratified 50/50 split."""
    if split_json and os.path.exists(split_json):
        with open(split_json, encoding="utf-8") as f:
            obj = json.load(f)
        return set(str(x) for x in obj["tune_ids"]), set(str(x) for x in obj["eval_ids"])

    adms = _read_jsonl(val_fold)
    pos_ids, neg_ids = [], []
    for a in adms:
        aid = str(a["admission_id"])
        d2p = "\n".join(d["text"] for d in a["docs"] if _in_day_window(d, 2, None) and not _is_discharge(d))
        if has_late_op_event(d2p):
            pos_ids.append(aid)
        else:
            neg_ids.append(aid)
    rng = np.random.RandomState(seed)
    pos_arr = np.array(pos_ids)
    neg_arr = np.array(neg_ids)
    rng.shuffle(pos_arr)
    rng.shuffle(neg_arr)
    hp, hn = len(pos_arr) // 2, len(neg_arr) // 2
    tune_ids = set(pos_arr[:hp].tolist() + neg_arr[:hn].tolist())
    eval_ids = set(pos_arr[hp:].tolist() + neg_arr[hn:].tolist())
    return tune_ids, eval_ids


def score_t1_neural(meta, preds, eval_rows, seeds=SEEDS, n_boot=2000):
    g_te = [r[3] for r in eval_rows]
    out = {"train_n": meta["train_n"]}
    if "real_train_n" in meta:
        out["real_train_n"] = meta["real_train_n"]
        out["synth_train_n"] = meta["synth_train_n"]
    for target_idx, label in ((1, "T1a"), (2, "T1b")):
        info = meta["targets"][label]
        if "real_train_n" in meta:
            out[f"{label}_real_train_pos"] = meta["r_pos"][label]
            out[f"{label}_synth_train_pos"] = meta["s_pos"][label]
            out[f"{label}_train_pos"] = meta["r_pos"][label] + meta["s_pos"][label]
        else:
            out[f"{label}_train_pos"] = meta["train_pos"][label]
        y_te = np.array([r[target_idx] for r in eval_rows], dtype=int)
        pred_pair = preds.get(label)
        if pred_pair is None or len(np.unique(y_te)) < 2:
            out[label] = {"auroc": None, "note": "single_class"}
            continue
        mean_p, seed_probs = pred_pair
        res = cluster_bootstrap_binary(y_te, mean_p, g_te, n_boot=n_boot, seed=seeds[0])
        for k in (
            "best_lr",
            "best_epochs",
            "tune_auroc",
            "selected_strategy",
            "selected_weight",
            "pt_ft_epochs",
            "real_only_tune_auroc",
            "contribution_note",
            "real_train_n",
            "synth_train_n",
        ):
            if k in info:
                res[k] = info[k]
        res["seed_aurocs"] = [round(np_roc_auc(y_te, p), 4) for p in seed_probs if np_roc_auc(y_te, p) is not None]
        out[label] = res
    return out


def score_t2a_neural(meta, pred_pair, val_rows, n_boot=2000, seed=20261201):
    if meta.get("empty") or not val_rows:
        return {"n_train": meta.get("n_train", 0), "n_val": len(val_rows), "macro_f1_3class": None}
    y_te = np.array([r[1] for r in val_rows], dtype=int)
    if meta.get("single_class") or pred_pair is None:
        return {"n_train": meta["n_train"], "n_val": len(val_rows), "macro_f1_3class": None, "note": "single_class_train"}
    pred, _ = pred_pair
    f1s = np_f1_per_class(y_te, pred, labels=(0, 1, 2))
    present = sorted(set(int(k) for k in meta["train_dist"]) | set(y_te.tolist()))
    macro_pt = float(np.mean(np_f1_per_class(y_te, pred, labels=present)))

    g_ids = [r[2] for r in val_rows]
    rng = np.random.RandomState(seed)
    uniq = list(dict.fromkeys(g_ids))
    idx_by_g = collections.defaultdict(list)
    for idx, g in enumerate(g_ids):
        idx_by_g[g].append(idx)
    macros = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sel = [i for g in sampled for i in idx_by_g[g]]
        yt_b = y_te[sel]
        pres_b = sorted(set(int(k) for k in meta["train_dist"]) | set(yt_b.tolist()))
        macros.append(float(np.mean(np_f1_per_class(yt_b, pred[sel], labels=pres_b))))

    res = {
        "n_train": meta["n_train"],
        "n_val": len(val_rows),
        "train_dist": meta["train_dist"],
        "val_dist": {str(k): v for k, v in sorted(collections.Counter(y_te.tolist()).items())},
        "macro_f1_3class": round(macro_pt, 4),
        "macro_f1_ci95": [round(float(np.percentile(macros, 2.5)), 4), round(float(np.percentile(macros, 97.5)), 4)] if macros else None,
        "f1_excluded_0": round(float(f1s[0]), 4),
        "f1_diff_1": round(float(f1s[1]), 4),
        "f1_conf_2": round(float(f1s[2]), 4),
        "val_premature_conf_rate": round(float(np.mean(pred[y_te == 1] == 2)), 4) if np.any(y_te == 1) else None,
        "val_missed_update_rate": round(float(np.mean(pred[y_te != 1] == 1)), 4) if np.any(y_te != 1) else None,
    }
    for k in ("best_epochs", "tune_macro_f1", "selected_strategy", "selected_weight", "real_only_tune_macro_f1", "contribution_note"):
        if k in meta:
            res[k] = meta[k]
    return res


def score_t2b_neural(meta, probs, val_rows, seeds=SEEDS, n_boot=2000):
    if meta.get("empty") or not val_rows:
        return {"n_train": meta.get("n_train", 0), "n_val": len(val_rows), "auroc": None}
    y_te = np.array([r[1] for r in val_rows], dtype=int)
    sil_te = np.array([bool(r[2]) for r in val_rows], dtype=bool)
    non_sil = ~sil_te
    if meta.get("single_class") or probs is None or len(np.unique(y_te)) < 2:
        return {
            "n_train": meta["n_train"],
            "n_val": len(val_rows),
            "train_pos": meta.get("train_pos", 0),
            "val_pos": int(np.sum(y_te)),
            "val_mentioned": int(np.sum(non_sil)),
            "auroc": None,
            "mentioned_only_auroc": None,
            "note": "single_class",
        }
    g_all = [r[3] for r in val_rows]
    g_ment = [r[3] for i, r in enumerate(val_rows) if non_sil[i]]
    full_boot = cluster_bootstrap_binary(y_te, probs, g_all, n_boot=n_boot, seed=seeds[0])

    has_m2 = int(np.sum(non_sil)) > 0 and len(np.unique(y_te[non_sil])) >= 2
    ment_boot = cluster_bootstrap_binary(y_te[non_sil], probs[non_sil], g_ment, n_boot=n_boot, seed=seeds[0]) if has_m2 else {}

    p05 = (probs >= 0.5).astype(int)
    t_cal = float(meta.get("thr_cal", 0.5))
    pcal = (probs >= t_cal).astype(int)
    prec05, rec05, f1_05 = np_prf1(y_te, p05)
    preccal, reccal, f1_cal = np_prf1(y_te, pcal)
    sil_lens = [len(r[0]) for i, r in enumerate(val_rows) if sil_te[i]]
    men_lens = [len(r[0]) for i, r in enumerate(val_rows) if non_sil[i]]
    triv_auc = np_roc_auc(y_te, non_sil.astype(float))

    res = {
        "n_train": meta["n_train"],
        "train_pos": meta["train_pos"],
        "train_silent_pos": meta["train_silent_pos"],
        "n_val": len(val_rows),
        "val_pos": int(np.sum(y_te)),
        "val_silent": int(np.sum(sil_te)),
        "val_mentioned": int(np.sum(non_sil)),
        "silent_mean_chars": round(float(np.mean(sil_lens)), 1) if sil_lens else None,
        "mentioned_mean_chars": round(float(np.mean(men_lens)), 1) if men_lens else None,
        "trivial_mention_only_full_auroc": round(triv_auc, 4) if triv_auc is not None else None,
        "auroc": full_boot["auroc"],
        "auroc_ci95": full_boot.get("auroc_ci95"),
        "auprc": full_boot["auprc"],
        "auprc_ci95": full_boot.get("auprc_ci95"),
        "brier": full_boot.get("brier"),
        "ece": full_boot.get("ece"),
        "mentioned_only_auroc": ment_boot.get("auroc"),
        "mentioned_only_auroc_ci95": ment_boot.get("auroc_ci95"),
        "mentioned_only_auprc": ment_boot.get("auprc"),
        "mentioned_only_auprc_ci95": ment_boot.get("auprc_ci95"),
        "mentioned_only_auroc_null": 0.5,
        "mentioned_only_auprc_null": round(float(np.mean(y_te[non_sil])), 4) if int(np.sum(non_sil)) > 0 else None,
        "precision_05": round(prec05, 4),
        "recall_05": round(rec05, 4),
        "f1_05": round(f1_05, 4),
        "silent_false_update_rate": round(float(np.mean(p05[sil_te] == 1)), 4) if np.any(sil_te) else None,
        "calibrated_threshold": round(t_cal, 4),
        "thr_cal_source": meta.get("thr_cal_source"),
        "precision_cal": round(preccal, 4),
        "recall_cal": round(reccal, 4),
        "f1_cal": round(f1_cal, 4),
        "silent_false_update_rate_cal": round(float(np.mean(pcal[sil_te] == 1)), 4) if np.any(sil_te) else None,
    }
    for k in ("best_epochs", "tune_score", "selected_strategy", "selected_weight", "real_only_tune_score", "contribution_note"):
        if k in meta:
            res[k] = meta[k]
    return res


def score_t2_human_gold_neural(pred_probs_by_key, pred_bin_by_key, gold_sheet_jsonl, sealed_key_json):
    """Delegate single-arm or multi-arm HT clinician gold scoring to shared strict bijection scorer."""
    from t123_bench import score_t2_human_gold

    return score_t2_human_gold(pred_probs_by_key, pred_bin_by_key, gold_sheet_jsonl, sealed_key_json)


def encode_head_tail_ids(texts, tokenizer, max_len):
    """Fast head+tail truncation keeping equal token budget for document start and tail."""
    cap = max(2, int(max_len) - 2)
    head_cap = cap // 2
    tail_cap = cap - head_cap
    cls_id = tokenizer.cls_token_id if getattr(tokenizer, "cls_token_id", None) is not None else 0
    sep_id = tokenizer.sep_token_id if getattr(tokenizer, "sep_token_id", None) is not None else 0
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    norm_texts = [t if isinstance(t, str) else (t or "") for t in texts]
    if not norm_texts:
        return [], 2, pad_id
    try:
        batch_ids = tokenizer(norm_texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    except Exception:
        batch_ids = [tokenizer.encode(t, add_special_tokens=False) for t in norm_texts]
    seqs = []
    max_seen = 2
    for ids in batch_ids:
        if len(ids) > cap:
            ids = ids[:head_cap] + ids[-tail_cap:]
        row = [cls_id] + list(ids) + [sep_id]
        seqs.append(row)
        if len(row) > max_seen:
            max_seen = len(row)
    return seqs, max_seen, pad_id


def encode_head_tail_batch(texts, tokenizer, max_len):
    """Tokenize texts with head+tail truncation into CPU int64 (input_ids, attention_mask)."""
    seqs, max_seen, pad_id = encode_head_tail_ids(texts, tokenizer, max_len)
    n = len(seqs)
    width = min(int(max_len), max_seen)
    input_ids = torch.full((n, width), pad_id, dtype=torch.int64)
    attention_mask = torch.zeros((n, width), dtype=torch.int64)
    for i, row in enumerate(seqs):
        ln = len(row)
        input_ids[i, :ln] = torch.tensor(row, dtype=torch.int64)
        attention_mask[i, :ln] = 1
    return input_ids, attention_mask


def _is_embedding_key(name):
    return ".embeddings." in name or name.startswith("bert.embeddings.") or name.startswith("embeddings.")


def _extract_trainable_fp16(model):
    return {
        k: v.detach().cpu().half()
        for k, v in model.state_dict().items()
        if not _is_embedding_key(k)
    }


def _strip_classifier_keys(state_dict):
    return {k: v for k, v in state_dict.items() if not (k.startswith("classifier.") or ".classifier." in k)}


def _build_model(encoder_path, num_labels, init_trainable_state=None, device="cpu"):
    cfg = AutoConfig.from_pretrained(encoder_path, local_files_only=True)
    cfg.num_labels = int(num_labels)
    cfg.id2label = {i: f"L{i}" for i in range(int(num_labels))}
    cfg.label2id = {f"L{i}": i for i in range(int(num_labels))}
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_path, config=cfg, local_files_only=True, ignore_mismatched_sizes=True
    )
    for name, param in model.named_parameters():
        if _is_embedding_key(name):
            param.requires_grad = False
    if init_trainable_state is not None:
        loaded = {k: v.float() for k, v in init_trainable_state.items()}
        model.load_state_dict(loaded, strict=False)
    model.to(device)
    return model


def _autocast_ctx(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _predict_logits(model, enc_pair, device, batch_size=64):
    input_ids, attention_mask = enc_pair
    n = input_ids.shape[0]
    if n == 0:
        return np.zeros((0, model.config.num_labels), dtype=np.float32)
    model.eval()
    chunks = []
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            hi = min(n, lo + batch_size)
            mb = attention_mask[lo:hi]
            mx = max(1, int(mb.sum(dim=1).max().item()))
            ib = input_ids[lo:hi, :mx].to(device, non_blocking=(device.type == "cuda"))
            mb = mb[:, :mx].to(device, non_blocking=(device.type == "cuda"))
            with _autocast_ctx(device):
                logits = model(input_ids=ib, attention_mask=mb).logits
            chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _train_one_seed(
    encoder_path,
    num_labels,
    tr_enc,
    y_tensor,
    loss_fn,
    device,
    lr,
    max_epochs,
    seed,
    init_trunk_state=None,
    epoch_hook=None,
):
    """Fine-tune one seed end-to-end with frozen embeddings and return optional hook results."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = _build_model(encoder_path, num_labels, init_trainable_state=init_trunk_state, device=device)
    tr_ids, tr_mask = tr_enc
    n_train = tr_ids.shape[0]
    batch_size = 16 if n_train < 160 else 32
    steps_per_ep = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = max(1, steps_per_ep * max_epochs)
    warmup_steps = max(1, int(0.10 * total_steps))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    lr_lambda = lambda st: (st + 1) / warmup_steps if st < warmup_steps else max(0.0, (total_steps - st) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    gen = torch.Generator()
    gen.manual_seed(seed)

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for lo in range(0, n_train, batch_size):
            idx = perm[lo:lo + batch_size]
            mb = tr_mask[idx]
            mx = max(1, int(mb.sum(dim=1).max().item()))
            ib = tr_ids[idx, :mx].to(device, non_blocking=(device.type == "cuda"))
            mb = mb[:, :mx].to(device, non_blocking=(device.type == "cuda"))
            yb = y_tensor[idx].to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device):
                logits = model(input_ids=ib, attention_mask=mb).logits
            loss = loss_fn(logits.float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

        if epoch_hook is not None:
            epoch_hook(ep, model)

    final_state = _extract_trainable_fp16(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_state


def _save_pt(bundle_dir, rel_name, trainable_state):
    abs_p = os.path.join(bundle_dir, rel_name)
    os.makedirs(os.path.dirname(abs_p), exist_ok=True)
    torch.save(trainable_state, abs_p)
    return rel_name


def _logits_to_t1_probs(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -30.0, 30.0)))


def _logits_to_t2b_probs(logits):
    v = logits[:, 0] if logits.ndim == 2 else logits
    return 1.0 / (1.0 + np.exp(-np.clip(v.astype(np.float64), -30.0, 30.0)))


def _logits_to_t2a_probs(logits):
    x = logits.astype(np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -30.0, 30.0))
    return ex / np.sum(ex, axis=1, keepdims=True)


def _eval_pt_files(ckpt_dir, rel_paths, encoder_path, num_labels, enc_pair, prob_fn, device):
    probs_per_seed = []
    for rel in rel_paths:
        state = torch.load(os.path.join(ckpt_dir, rel), map_location="cpu", weights_only=True)
        model = _build_model(encoder_path, num_labels, init_trainable_state=state, device=device)
        logits = _predict_logits(model, enc_pair, device)
        probs_per_seed.append(prob_fn(logits))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return probs_per_seed


class NeuralBenchRunner:
    """Encapsulates tokenization cache, local HF backbone fitting/freezing, and read-only eval."""

    def __init__(self, encoder_path, device=None, max_epochs_override=None):
        self.encoder_path = encoder_path
        self.max_epochs_override = max_epochs_override
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
        self._enc_cache = {}
        self._tune_pred_cache = {}

    def encode_cached(self, key, texts, max_len):
        if key not in self._enc_cache:
            self._enc_cache[key] = encode_head_tail_batch(texts, self.tokenizer, max_len)
        return self._enc_cache[key]

    def fit_t1_arm(self, bundle_dir, arm_key, train_rows, tune_rows, tune_key="t1_tune", seeds=SEEDS):
        y_a = np.array([r[1] for r in train_rows], dtype=int)
        y_b = np.array([r[2] for r in train_rows], dtype=int)
        train_pos = {"T1a": int(np.sum(y_a)), "T1b": int(np.sum(y_b))}
        act_a = len(np.unique(y_a)) >= 2
        act_b = len(np.unique(y_b)) >= 2
        if not act_a and not act_b:
            return {"kind": "t1_single", "train_n": len(train_rows), "train_pos": train_pos, "targets": {"T1a": {"single_class": True}, "T1b": {"single_class": True}}}

        tr_enc = encode_head_tail_batch([r[0] for r in train_rows], self.tokenizer, 512)
        tu_enc = self.encode_cached(tune_key, [r[0] for r in tune_rows], 512)
        y_tu_a = np.array([r[1] for r in tune_rows], dtype=int)
        y_tu_b = np.array([r[2] for r in tune_rows], dtype=int)
        y_tr_t = torch.tensor(np.stack([y_a, y_b], axis=1), dtype=torch.float32)

        pw_a = min(8.0, max(1.0, (len(y_a) - int(np.sum(y_a))) / max(1, int(np.sum(y_a)))))
        pw_b = min(8.0, max(1.0, (len(y_b) - int(np.sum(y_b))) / max(1, int(np.sum(y_b)))))
        pos_w = torch.tensor([pw_a, pw_b], dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        max_ep = self.max_epochs_override or (5 if len(train_rows) <= 600 else 4)
        lr_records = {}
        for lr in (2e-5, 5e-5):
            snaps = {}
            best_a = (-1.0, 1, None)
            best_b = (-1.0, 1, None)

            def _hook0(ep, model, lr_val=lr):
                nonlocal best_a, best_b
                st = _extract_trainable_fp16(model)
                snaps[ep] = st
                tu_p = _logits_to_t1_probs(_predict_logits(model, tu_enc, self.device))
                aa = np_roc_auc(y_tu_a, tu_p[:, 0]) if (act_a and len(np.unique(y_tu_a)) >= 2) else None
                ab = np_roc_auc(y_tu_b, tu_p[:, 1]) if (act_b and len(np.unique(y_tu_b)) >= 2) else None
                va = aa if aa is not None else 0.0
                vb = ab if ab is not None else 0.0
                if va > best_a[0]:
                    best_a = (va, ep, tu_p[:, 0])
                if vb > best_b[0]:
                    best_b = (vb, ep, tu_p[:, 1])

            _train_one_seed(
                self.encoder_path, 2, tr_enc, y_tr_t, loss_fn, self.device, lr, max_ep, seeds[0], epoch_hook=_hook0
            )
            lr_records[lr] = (best_a, best_b, snaps)

        best_lr = max(
            (2e-5, 5e-5),
            key=lambda lr: (lr_records[lr][0][0] if act_a else lr_records[lr][1][0], lr_records[lr][1][0]),
        )
        best_a, best_b, snaps0 = lr_records[best_lr]
        ep_a, ep_b = int(best_a[1]), int(best_b[1])
        auc_tu_a, auc_tu_b = best_a[0], best_b[0]

        safe_arm = arm_key.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace("|", "_")
        seed_paths_a, seed_paths_b = [], []
        tu_preds_a, tu_preds_b = [best_a[2]], [best_b[2]]

        p0_a = _save_pt(bundle_dir, f"ckpts/t1_{safe_arm}_s0_T1a.pt", snaps0[ep_a])
        seed_paths_a.append(p0_a)
        if ep_b == ep_a:
            seed_paths_b.append(p0_a)
        else:
            seed_paths_b.append(_save_pt(bundle_dir, f"ckpts/t1_{safe_arm}_s0_T1b.pt", snaps0[ep_b]))

        top_ep = max(ep_a, ep_b)
        for s_idx, s in enumerate(seeds[1:], start=1):
            snaps_s = {}
            tpa = tpb = None

            def _hook_s(ep, model):
                nonlocal tpa, tpb
                if ep == ep_a or ep == ep_b:
                    snaps_s[ep] = _extract_trainable_fp16(model)
                    tu_p = _logits_to_t1_probs(_predict_logits(model, tu_enc, self.device))
                    if ep == ep_a:
                        tpa = tu_p[:, 0]
                    if ep == ep_b:
                        tpb = tu_p[:, 1]

            _train_one_seed(
                self.encoder_path, 2, tr_enc, y_tr_t, loss_fn, self.device, best_lr, top_ep, s, epoch_hook=_hook_s
            )
            pa_rel = _save_pt(bundle_dir, f"ckpts/t1_{safe_arm}_s{s_idx}_T1a.pt", snaps_s[ep_a])
            seed_paths_a.append(pa_rel)
            if ep_b == ep_a:
                seed_paths_b.append(pa_rel)
            else:
                seed_paths_b.append(_save_pt(bundle_dir, f"ckpts/t1_{safe_arm}_s{s_idx}_T1b.pt", snaps_s[ep_b]))
            tu_preds_a.append(tpa)
            tu_preds_b.append(tpb)

        self._tune_pred_cache[("T1", arm_key)] = (
            np.mean(tu_preds_a, axis=0) if all(x is not None for x in tu_preds_a) else None,
            np.mean(tu_preds_b, axis=0) if all(x is not None for x in tu_preds_b) else None,
        )
        targets = {
            "T1a": {
                "single_class": not act_a,
                "best_lr": best_lr,
                "best_epochs": ep_a,
                "tune_auroc": round(float(auc_tu_a), 4) if act_a else None,
                "ckpts": seed_paths_a if act_a else [],
            },
            "T1b": {
                "single_class": not act_b,
                "best_lr": best_lr,
                "best_epochs": ep_b,
                "tune_auroc": round(float(auc_tu_b), 4) if act_b else None,
                "ckpts": seed_paths_b if act_b else [],
            },
        }
        return {"kind": "t1_single", "train_n": len(train_rows), "train_pos": train_pos, "targets": targets}

    def fit_t1_augment(self, bundle_dir, aug_key, real_key, synth_key, real_rows, synth_rows, real_meta, synth_meta, tune_rows, tune_key="t1_tune", seeds=SEEDS):
        r_pos = real_meta["train_pos"]
        s_pos = synth_meta["train_pos"]
        p_r_a, p_r_b = self._tune_pred_cache.get(("T1", real_key), (None, None))
        p_s_a, p_s_b = self._tune_pred_cache.get(("T1", synth_key), (None, None))

        y_r_a = np.array([r[1] for r in real_rows], dtype=int)
        y_r_b = np.array([r[2] for r in real_rows], dtype=int)
        act_a = len(np.unique(y_r_a)) >= 2
        act_b = len(np.unique(y_r_b)) >= 2
        y_tu_a = np.array([r[1] for r in tune_rows], dtype=int)
        y_tu_b = np.array([r[2] for r in tune_rows], dtype=int)
        tu_enc = self.encode_cached(tune_key, [r[0] for r in tune_rows], 512)

        # Two-stage neural transfer candidate: Synth trunk -> Real fine-tune (3 epochs @ 2e-5)
        ptft_ep = self.max_epochs_override or 3
        ptft_a, ptft_b = [], []
        ptft_snaps_a, ptft_snaps_b = [], []
        if act_a and not synth_meta["targets"]["T1a"].get("single_class"):
            tr_enc = encode_head_tail_batch([r[0] for r in real_rows], self.tokenizer, 512)
            y_tr_t = torch.tensor(np.stack([y_r_a, y_r_b], axis=1), dtype=torch.float32)
            pw_a = min(8.0, max(1.0, (len(y_r_a) - int(np.sum(y_r_a))) / max(1, int(np.sum(y_r_a)))))
            pw_b = min(8.0, max(1.0, (len(y_r_b) - int(np.sum(y_r_b))) / max(1, int(np.sum(y_r_b)))))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_a, pw_b], dtype=torch.float32, device=self.device))

            ep_pt_a = ep_pt_b = 1
            for s_idx, s in enumerate(seeds):
                init_raw = torch.load(os.path.join(bundle_dir, synth_meta["targets"]["T1a"]["ckpts"][s_idx]), map_location="cpu", weights_only=True)
                init_trunk = _strip_classifier_keys(init_raw)
                snaps_s = {}
                scores_s = {}

                def _hook_pt(ep, model):
                    snaps_s[ep] = _extract_trainable_fp16(model)
                    tu_p = _logits_to_t1_probs(_predict_logits(model, tu_enc, self.device))
                    aa = np_roc_auc(y_tu_a, tu_p[:, 0]) if len(np.unique(y_tu_a)) >= 2 else 0.0
                    ab = np_roc_auc(y_tu_b, tu_p[:, 1]) if (act_b and len(np.unique(y_tu_b)) >= 2) else 0.0
                    scores_s[ep] = (aa or 0.0, ab or 0.0, tu_p[:, 0], tu_p[:, 1])

                _train_one_seed(
                    self.encoder_path, 2, tr_enc, y_tr_t, loss_fn, self.device, 2e-5, ptft_ep, s + 100,
                    init_trunk_state=init_trunk, epoch_hook=_hook_pt,
                )
                if s_idx == 0:
                    ep_pt_a = max(scores_s.keys(), key=lambda e: scores_s[e][0])
                    ep_pt_b = max(scores_s.keys(), key=lambda e: scores_s[e][1])
                ptft_snaps_a.append(snaps_s[ep_pt_a])
                ptft_snaps_b.append(snaps_s[ep_pt_b])
                ptft_a.append(scores_s[ep_pt_a][2])
                ptft_b.append(scores_s[ep_pt_b][3])

        p_ptft_a = np.mean(ptft_a, axis=0) if ptft_a else None
        p_ptft_b = np.mean(ptft_b, axis=0) if ptft_b else None
        safe_aug = aug_key.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")

        targets = {}
        for label, act, y_tu, p_r, p_s, p_pt, pt_snaps, pt_best_ep in (
            ("T1a", act_a, y_tu_a, p_r_a, p_s_a, p_ptft_a, ptft_snaps_a, ep_pt_a if ptft_a else 1),
            ("T1b", act_b, y_tu_b, p_r_b, p_s_b, p_ptft_b, ptft_snaps_b, ep_pt_b if ptft_b else 1),
        ):
            if not act or real_meta["targets"][label].get("single_class"):
                targets[label] = {"single_class": True}
                continue
            base_auc = np_roc_auc(y_tu, p_r) if (p_r is not None and len(np.unique(y_tu)) >= 2) else 0.5
            best_auc = base_auc
            best_strat, best_w = "prob_blend", 0.0
            if p_s is not None and len(np.unique(y_tu)) >= 2:
                for alpha in BLEND_ALPHAS:
                    cand = np_roc_auc(y_tu, (1.0 - alpha) * p_r + alpha * p_s)
                    if cand is not None and cand > best_auc + 1e-12:
                        best_auc, best_strat, best_w = cand, "prob_blend", alpha
            if p_pt is not None and len(np.unique(y_tu)) >= 2:
                for beta in PTFT_BETAS:
                    cand = np_roc_auc(y_tu, (1.0 - beta) * p_r + beta * p_pt)
                    if cand is not None and cand > best_auc + 1e-12:
                        best_auc, best_strat, best_w = cand, "pt_ft_blend", beta

            ckpts_pt = []
            if best_strat == "pt_ft_blend":
                for s_idx, snap in enumerate(pt_snaps):
                    ckpts_pt.append(_save_pt(bundle_dir, f"ckpts/t1_{safe_aug}_s{s_idx}_{label}.pt", snap))

            targets[label] = {
                "single_class": False,
                "selected_strategy": best_strat,
                "selected_weight": float(best_w),
                "pt_ft_epochs": int(pt_best_ep) if best_strat == "pt_ft_blend" else None,
                "tune_auroc": round(float(best_auc), 4),
                "real_only_tune_auroc": round(float(base_auc), 4),
                "contribution_note": "no_synthetic_gain_weight_zero" if best_w == 0.0 else "positive_tune_gain",
                "real_ckpts": real_meta["targets"][label]["ckpts"],
                "synth_ckpts": synth_meta["targets"][label].get("ckpts", []) if best_strat == "prob_blend" else [],
                "pt_ft_ckpts": ckpts_pt,
            }
        return {
            "kind": "t1_augment",
            "real_train_n": len(real_rows),
            "synth_train_n": len(synth_rows),
            "train_n": len(real_rows) + len(synth_rows),
            "r_pos": r_pos,
            "s_pos": s_pos,
            "targets": targets,
        }

    def fit_t2a_arm(self, bundle_dir, arm_key, train_rows, tune_rows, tune_key="t2a_tune", seeds=SEEDS, init_ckpts=None):
        if not train_rows:
            return {"kind": "t2a", "empty": True, "n_train": 0}
        y_tr = np.array([r[1] for r in train_rows], dtype=int)
        dist = {str(k): v for k, v in sorted(collections.Counter(y_tr.tolist()).items())}
        present = sorted(set(y_tr.tolist()))
        if len(present) < 2:
            return {"kind": "t2a", "empty": False, "single_class": True, "n_train": len(train_rows), "train_dist": dist}

        tr_enc = encode_head_tail_batch([r[0] for r in train_rows], self.tokenizer, 256)
        tu_enc = self.encode_cached(tune_key, [r[0] for r in tune_rows], 256)
        y_tu = np.array([r[1] for r in tune_rows], dtype=int)
        counts = collections.Counter(y_tr.tolist())
        w_cls = [
            float(np.clip(len(y_tr) / (len(present) * counts[c]), 0.5, 8.0)) if counts.get(c, 0) > 0 else 1.0
            for c in range(3)
        ]
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w_cls, dtype=torch.float32, device=self.device))
        y_tr_t = torch.tensor(y_tr, dtype=torch.int64)

        lr = 2e-5 if init_ckpts else 3e-5
        max_ep = self.max_epochs_override or (3 if init_ckpts else (4 if len(train_rows) <= 800 else 3))
        safe_arm = arm_key.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace("|", "_")

        best_score, best_ep = -1.0, 1
        snaps0 = {}
        tu_preds = []
        tu_p0_best = None
        init0 = _strip_classifier_keys(torch.load(os.path.join(bundle_dir, init_ckpts[0]), map_location="cpu", weights_only=True)) if init_ckpts else None

        def _hook0(ep, model):
            nonlocal best_score, best_ep, tu_p0_best
            snaps0[ep] = _extract_trainable_fp16(model)
            tp = _logits_to_t2a_probs(_predict_logits(model, tu_enc, self.device))
            pred_b = np.argmax(tp, axis=1)
            pres_tu = sorted(set(present) | set(y_tu.tolist()))
            mf1 = float(np.mean(np_f1_per_class(y_tu, pred_b, labels=pres_tu)))
            if mf1 > best_score:
                best_score, best_ep, tu_p0_best = mf1, ep, tp

        _train_one_seed(
            self.encoder_path, 3, tr_enc, y_tr_t, loss_fn, self.device, lr, max_ep, seeds[0] + (100 if init_ckpts else 0),
            init_trunk_state=init0, epoch_hook=_hook0,
        )
        tu_preds.append(tu_p0_best)
        ckpts = [_save_pt(bundle_dir, f"ckpts/t2a_{safe_arm}_s0.pt", snaps0[best_ep])]

        for s_idx, s in enumerate(seeds[1:], start=1):
            init_s = _strip_classifier_keys(torch.load(os.path.join(bundle_dir, init_ckpts[s_idx]), map_location="cpu", weights_only=True)) if init_ckpts else None
            fin = _train_one_seed(
                self.encoder_path, 3, tr_enc, y_tr_t, loss_fn, self.device, lr, best_ep, s + (100 if init_ckpts else 0),
                init_trunk_state=init_s,
            )
            rel = _save_pt(bundle_dir, f"ckpts/t2a_{safe_arm}_s{s_idx}.pt", fin)
            ckpts.append(rel)
            model = _build_model(self.encoder_path, 3, init_trainable_state=fin, device=self.device)
            tu_preds.append(_logits_to_t2a_probs(_predict_logits(model, tu_enc, self.device)))
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        mean_tu = np.mean(tu_preds, axis=0)
        self._tune_pred_cache[("T2a", arm_key)] = mean_tu
        pres_tu = sorted(set(present) | set(y_tu.tolist()))
        tu_mf1 = float(np.mean(np_f1_per_class(y_tu, np.argmax(mean_tu, axis=1), labels=pres_tu)))
        return {
            "kind": "t2a",
            "empty": False,
            "single_class": False,
            "n_train": len(train_rows),
            "train_dist": dist,
            "best_epochs": int(best_ep),
            "tune_macro_f1": round(tu_mf1, 4),
            "ckpts": ckpts,
        }

    def _cleanup_ptft_ckpts(self, bundle_dir, ptft_meta):
        for rel in ptft_meta.get("ckpts", ()):
            abs_p = os.path.join(bundle_dir, rel)
            if os.path.exists(abs_p):
                os.remove(abs_p)

    def fit_t2a_augment(self, bundle_dir, aug_key, real_key, synth_key, real_rows, synth_rows, real_meta, synth_meta, tune_rows, tune_key="t2a_tune", seeds=SEEDS):
        if real_meta.get("empty") or real_meta.get("single_class"):
            return real_meta
        p_r = self._tune_pred_cache.get(("T2a", real_key))
        p_s = self._tune_pred_cache.get(("T2a", synth_key))
        if p_r is None or p_s is None or synth_meta.get("single_class") or not synth_meta.get("ckpts"):
            return {
                "kind": "t2a_augment",
                "empty": False,
                "single_class": False,
                "n_train": len(real_rows),
                "train_dist": real_meta["train_dist"],
                "selected_strategy": "prob_blend",
                "selected_weight": 0.0,
                "contribution_note": "no_synthetic_gain_weight_zero",
                "real_ckpts": real_meta["ckpts"],
                "synth_ckpts": [],
                "pt_ft_ckpts": [],
            }
        y_tu = np.array([r[1] for r in tune_rows], dtype=int)
        pres = sorted(set(int(k) for k in real_meta["train_dist"]) | set(y_tu.tolist()))
        base_mf1 = float(np.mean(np_f1_per_class(y_tu, np.argmax(p_r, axis=1), labels=pres)))

        ptft_meta = self.fit_t2a_arm(
            bundle_dir, f"{aug_key}_ptft_cand", real_rows, tune_rows, tune_key=tune_key, seeds=seeds, init_ckpts=synth_meta["ckpts"],
        )
        p_pt = self._tune_pred_cache.pop(("T2a", f"{aug_key}_ptft_cand"), None)

        best_score, best_strat, best_w = base_mf1, "prob_blend", 0.0
        for alpha in BLEND_ALPHAS:
            sc = float(np.mean(np_f1_per_class(y_tu, np.argmax((1.0 - alpha) * p_r + alpha * p_s, axis=1), labels=pres)))
            if sc > best_score + 1e-12:
                best_score, best_strat, best_w = sc, "prob_blend", alpha
        if p_pt is not None:
            for beta in PTFT_BETAS:
                sc = float(np.mean(np_f1_per_class(y_tu, np.argmax((1.0 - beta) * p_r + beta * p_pt, axis=1), labels=pres)))
                if sc > best_score + 1e-12:
                    best_score, best_strat, best_w = sc, "pt_ft_blend", beta

        if best_strat != "pt_ft_blend":
            self._cleanup_ptft_ckpts(bundle_dir, ptft_meta)
            pt_ckpts = []
        else:
            pt_ckpts = ptft_meta["ckpts"]

        return {
            "kind": "t2a_augment",
            "empty": False,
            "single_class": False,
            "n_train": len(real_rows) + len(synth_rows),
            "train_dist": real_meta["train_dist"],
            "selected_strategy": best_strat,
            "selected_weight": float(best_w),
            "tune_macro_f1": round(best_score, 4),
            "real_only_tune_macro_f1": round(base_mf1, 4),
            "contribution_note": "no_synthetic_gain_weight_zero" if best_w == 0.0 else "positive_tune_gain",
            "real_ckpts": real_meta["ckpts"],
            "synth_ckpts": synth_meta["ckpts"] if best_strat == "prob_blend" else [],
            "pt_ft_ckpts": pt_ckpts,
        }

    def _calibrate_t2b_thr(self, probs, tune_rows):
        y_cal = np.array([r[1] for r in tune_rows], dtype=int)
        if len(y_cal) < 10 or len(np.unique(y_cal)) < 2:
            return 0.5, "default_0.5"
        best_f, cal_thr = -1.0, 0.5
        for t in T2B_THR_GRID:
            _, _, f = np_prf1(y_cal, (probs >= t).astype(int))
            if f > best_f:
                best_f, cal_thr = f, t
        return float(cal_thr), "tune_fold"

    def _t2b_tune_metric(self, probs, tune_rows):
        y_tu = np.array([r[1] for r in tune_rows], dtype=int)
        non_sil = ~np.array([bool(r[2]) for r in tune_rows], dtype=bool)
        if len(np.unique(y_tu)) < 2:
            return 0.0
        full_auc = np_roc_auc(y_tu, probs) or 0.0
        if np.any(non_sil) and len(np.unique(y_tu[non_sil])) >= 2:
            ment_auc = np_roc_auc(y_tu[non_sil], probs[non_sil]) or 0.0
            return 0.5 * full_auc + 0.5 * ment_auc
        return full_auc

    def fit_t2b_arm(self, bundle_dir, arm_key, train_rows, tune_rows, tune_key="t2b_tune", seeds=SEEDS, init_ckpts=None):
        if not train_rows:
            return {"kind": "t2b", "empty": True, "n_train": 0}
        y_tr = np.array([r[1] for r in train_rows], dtype=int)
        t_pos = int(np.sum(y_tr))
        t_sil_pos = int(sum(1 for r in train_rows if r[1] == 1 and r[2]))
        if len(np.unique(y_tr)) < 2:
            return {"kind": "t2b", "empty": False, "single_class": True, "n_train": len(train_rows), "train_pos": t_pos, "train_silent_pos": t_sil_pos}

        tr_enc = encode_head_tail_batch([r[0] for r in train_rows], self.tokenizer, 384)
        tu_enc = self.encode_cached(tune_key, [r[0] for r in tune_rows], 384)
        pw = min(8.0, max(1.0, (len(y_tr) - t_pos) / max(1, t_pos)))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32, device=self.device))
        y_tr_t = torch.tensor(y_tr[:, None], dtype=torch.float32)

        lr = 2e-5 if init_ckpts else 3e-5
        max_ep = self.max_epochs_override or (3 if init_ckpts else (4 if len(train_rows) <= 800 else 3))
        safe_arm = arm_key.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace("|", "_")

        best_score, best_ep = -1.0, 1
        snaps0 = {}
        tu_preds = []
        tu_p0_best = None
        init0 = _strip_classifier_keys(torch.load(os.path.join(bundle_dir, init_ckpts[0]), map_location="cpu", weights_only=True)) if init_ckpts else None

        def _hook0(ep, model):
            nonlocal best_score, best_ep, tu_p0_best
            snaps0[ep] = _extract_trainable_fp16(model)
            tp = _logits_to_t2b_probs(_predict_logits(model, tu_enc, self.device))
            sc = self._t2b_tune_metric(tp, tune_rows)
            if sc > best_score:
                best_score, best_ep, tu_p0_best = sc, ep, tp

        _train_one_seed(
            self.encoder_path, 1, tr_enc, y_tr_t, loss_fn, self.device, lr, max_ep, seeds[0] + (100 if init_ckpts else 0),
            init_trunk_state=init0, epoch_hook=_hook0,
        )
        tu_preds.append(tu_p0_best)
        ckpts = [_save_pt(bundle_dir, f"ckpts/t2b_{safe_arm}_s0.pt", snaps0[best_ep])]

        for s_idx, s in enumerate(seeds[1:], start=1):
            init_s = _strip_classifier_keys(torch.load(os.path.join(bundle_dir, init_ckpts[s_idx]), map_location="cpu", weights_only=True)) if init_ckpts else None
            fin = _train_one_seed(
                self.encoder_path, 1, tr_enc, y_tr_t, loss_fn, self.device, lr, best_ep, s + (100 if init_ckpts else 0),
                init_trunk_state=init_s,
            )
            rel = _save_pt(bundle_dir, f"ckpts/t2b_{safe_arm}_s{s_idx}.pt", fin)
            ckpts.append(rel)
            model = _build_model(self.encoder_path, 1, init_trainable_state=fin, device=self.device)
            tu_preds.append(_logits_to_t2b_probs(_predict_logits(model, tu_enc, self.device)))
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        mean_tu = np.mean(tu_preds, axis=0)
        self._tune_pred_cache[("T2b", arm_key)] = mean_tu
        cal_thr, cal_src = self._calibrate_t2b_thr(mean_tu, tune_rows)
        return {
            "kind": "t2b",
            "empty": False,
            "single_class": False,
            "n_train": len(train_rows),
            "train_pos": t_pos,
            "train_silent_pos": t_sil_pos,
            "best_epochs": int(best_ep),
            "tune_score": round(self._t2b_tune_metric(mean_tu, tune_rows), 4),
            "thr_cal": cal_thr,
            "thr_cal_source": cal_src,
            "ckpts": ckpts,
        }

    def fit_t2b_augment(self, bundle_dir, aug_key, real_key, synth_key, real_rows, synth_rows, real_meta, synth_meta, tune_rows, tune_key="t2b_tune", seeds=SEEDS):
        if real_meta.get("empty") or real_meta.get("single_class"):
            return real_meta
        p_r = self._tune_pred_cache.get(("T2b", real_key))
        p_s = self._tune_pred_cache.get(("T2b", synth_key))
        if p_r is None or p_s is None or synth_meta.get("single_class") or not synth_meta.get("ckpts"):
            return {
                "kind": "t2b_augment",
                "empty": False,
                "single_class": False,
                "n_train": len(real_rows),
                "train_pos": real_meta.get("train_pos", 0),
                "train_silent_pos": real_meta.get("train_silent_pos", 0),
                "selected_strategy": "prob_blend",
                "selected_weight": 0.0,
                "contribution_note": "no_synthetic_gain_weight_zero",
                "thr_cal": real_meta.get("thr_cal", 0.5),
                "thr_cal_source": real_meta.get("thr_cal_source", "default_0.5"),
                "real_ckpts": real_meta["ckpts"],
                "synth_ckpts": [],
                "pt_ft_ckpts": [],
            }
        base_sc = self._t2b_tune_metric(p_r, tune_rows)

        ptft_meta = self.fit_t2b_arm(
            bundle_dir, f"{aug_key}_ptft_cand", real_rows, tune_rows, tune_key=tune_key, seeds=seeds, init_ckpts=synth_meta["ckpts"],
        )
        p_pt = self._tune_pred_cache.pop(("T2b", f"{aug_key}_ptft_cand"), None)

        best_sc, best_strat, best_w, best_p = base_sc, "prob_blend", 0.0, p_r
        for alpha in BLEND_ALPHAS:
            pb = (1.0 - alpha) * p_r + alpha * p_s
            sc = self._t2b_tune_metric(pb, tune_rows)
            if sc > best_sc + 1e-12:
                best_sc, best_strat, best_w, best_p = sc, "prob_blend", alpha, pb
        if p_pt is not None:
            for beta in PTFT_BETAS:
                pb = (1.0 - beta) * p_r + beta * p_pt
                sc = self._t2b_tune_metric(pb, tune_rows)
                if sc > best_sc + 1e-12:
                    best_sc, best_strat, best_w, best_p = sc, "pt_ft_blend", beta, pb

        if best_strat != "pt_ft_blend":
            self._cleanup_ptft_ckpts(bundle_dir, ptft_meta)
            pt_ckpts = []
        else:
            pt_ckpts = ptft_meta["ckpts"]

        cal_thr, cal_src = self._calibrate_t2b_thr(best_p, tune_rows)
        return {
            "kind": "t2b_augment",
            "empty": False,
            "single_class": False,
            "n_train": len(real_rows) + len(synth_rows),
            "train_pos": real_meta["train_pos"] + synth_meta["train_pos"],
            "train_silent_pos": real_meta["train_silent_pos"] + synth_meta["train_silent_pos"],
            "selected_strategy": best_strat,
            "selected_weight": float(best_w),
            "tune_score": round(best_sc, 4),
            "real_only_tune_score": round(base_sc, 4),
            "contribution_note": "no_synthetic_gain_weight_zero" if best_w == 0.0 else "positive_tune_gain",
            "thr_cal": cal_thr,
            "thr_cal_source": cal_src,
            "real_ckpts": real_meta["ckpts"],
            "synth_ckpts": synth_meta["ckpts"] if best_strat == "prob_blend" else [],
            "pt_ft_ckpts": pt_ckpts,
        }


def fit_t1_neural_augment(
    bundle_dir,
    real_rows,
    synth_rows,
    tune_rows,
    encoder_path=DEFAULT_ENCODER_PATH,
    aug_key="Augment",
    seeds=SEEDS,
    max_epochs_override=None,
):
    """Fit Real, Synth, and tune-calibrated Augment fusion end-to-end (API helper & anti-dilution test entry)."""
    runner = NeuralBenchRunner(encoder_path, max_epochs_override=max_epochs_override)
    real_art = runner.fit_t1_arm(bundle_dir, f"{aug_key}_real", real_rows, tune_rows, seeds=seeds)
    synth_art = runner.fit_t1_arm(bundle_dir, f"{aug_key}_synth", synth_rows, tune_rows, seeds=seeds)
    aug_art = runner.fit_t1_augment(
        bundle_dir, aug_key, f"{aug_key}_real", f"{aug_key}_synth", real_rows, synth_rows, real_art, synth_art, tune_rows, seeds=seeds
    )
    return aug_art, real_art, synth_art


def _collect_arm_sources(args):
    if getattr(args, "real_folds", None):
        real_folds_27 = [p for p in args.real_folds if p and os.path.exists(p)]
    else:
        real_folds_27 = [f"/home/weify/pyt/data/deid_run9_fold{k}.jsonl" for k in range(2, 8) if os.path.exists(f"/home/weify/pyt/data/deid_run9_fold{k}.jsonl")]
    direct_paths = _resolve_direct32b_paths(args.s5_root, args.direct32b_jsonl)
    exported_paths = _resolve_exported_synth_path(args.s5_root, getattr(args, "exported_synth_jsonl", None))
    fa, fj = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400)
    ffa, ffj = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=400, adm_file="first_try_admissions.jsonl")
    va, vj = _completed_shard_paths(args.s5_root, "s5_1", "vis_only_shard", min_kept=400)
    vfa, vfj = _completed_shard_paths(args.s5_root, "s5_1", "vis_only_shard", min_kept=400, adm_file="first_try_admissions.jsonl")
    oa, oj = _completed_shard_paths(args.s5_root, "s5_1", "oracle_leak_shard", min_kept=400)

    t1_src = collections.OrderedDict()
    if real_folds_27:
        t1_src["Real-50"] = (real_folds_27, None, 50, None)
        t1_src["Real-100"] = (real_folds_27, None, 100, None)
        t1_src["Real-Small (250)"] = (real_folds_27, None, 250, None)
        t1_src["Real-Full (2000)"] = (real_folds_27, None, 2000, None)
    if fa:
        t1_src["Synth-Full"] = (fa, fj, None, None)
        t1_src["Synth-Full (Matched N=600)"] = (fa, fj, None, 600)
    if exported_paths:
        t1_src["Synth-Exported (Release N=901)"] = (exported_paths, fj, None, None)
    if va:
        t1_src["Synth-VisibilityOnly"] = (va, vj, None, None)
    if oa:
        t1_src["Synth-OracleLeak"] = (oa, oj, None, None)
    if direct_paths:
        t1_src["DIRECT-32B"] = (direct_paths, None, None, None)

    t2_src = collections.OrderedDict()
    if real_folds_27:
        t2_src["Real-Small (250)"] = (real_folds_27, None, 250, None)
        t2_src["Real-Full (2000)"] = (real_folds_27, None, 2000, None)
    if fa:
        t2_src["Synth-Full (PostGate)"] = (fa, fj, None, None)
        t2_src["Synth-Full (Matched N=600)"] = (fa, fj, None, 600)
    if exported_paths:
        t2_src["Synth-Exported (Release N=901)"] = (exported_paths, fj, None, None)
    if ffa:
        t2_src["Synth-Full (FirstTry)"] = (ffa, ffj, None, None)
    if va:
        t2_src["Synth-VisibilityOnly (PostGate)"] = (va, vj, None, None)
    if vfa:
        t2_src["Synth-VisibilityOnly (FirstTry)"] = (vfa, vfj, None, None)
    if oa:
        t2_src["Synth-OracleLeak"] = (oa, oj, None, None)
    if direct_paths:
        t2_src["DIRECT-32B"] = (direct_paths, None, None, None)

    return t1_src, t2_src, {"full": fa, "vis_only": va, "oracle_leak": oa, "direct_32b": direct_paths, "exported_901": exported_paths}, (fa, fj, va, vj, direct_paths, exported_paths)


def _encoder_weight_hashes(encoder_path):
    hashes = {}
    for fn in ("config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json", "vocab.txt"):
        p = os.path.join(encoder_path, fn)
        if os.path.exists(p):
            hashes[fn] = _sha256_file(p)
    return hashes


def _collect_ckpt_hashes(bundle_dir):
    hashes = {}
    ckpt_root = os.path.join(bundle_dir, "ckpts")
    if os.path.isdir(ckpt_root):
        for fn in sorted(os.listdir(ckpt_root)):
            if fn.endswith(".pt"):
                rel = f"ckpts/{fn}"
                hashes[rel] = _sha256_file(os.path.join(bundle_dir, rel))
    return hashes


def fit_and_freeze_neural(args):
    if not HAS_TORCH_HF:
        raise RuntimeError("PyTorch and transformers are required for neural student fitting")
    assert_stage1_fit_allowed(args.val_fold, args.eval_fold)
    os.makedirs(args.fit_and_freeze, exist_ok=True)
    os.makedirs(os.path.join(args.fit_and_freeze, "ckpts"), exist_ok=True)

    if args.rehearsal_split:
        tune_ids, eval_ids = _load_or_make_rehearsal_split(args.val_fold, split_json=args.rehearsal_split_json)
    else:
        tune_ids = {str(a["admission_id"]) for a in _read_jsonl(args.val_fold)}
        eval_ids = None

    t1_tune = _filter_rows_aid(build_t1_dataset([args.val_fold], gt2_only=False), tune_ids, 3)
    t2a_tune = _filter_rows_aid(build_t2a_clause_dataset([args.val_fold]), tune_ids, 2)
    t2b_tune = _filter_rows_aid(build_t2b_update_dataset([args.val_fold]), tune_ids, 3)

    t1_src, t2_src, shard_lists, (fa, fj, va, vj, direct_paths, exported_paths) = _collect_arm_sources(args)
    runner = NeuralBenchRunner(args.encoder_path, max_epochs_override=args.max_epochs_override)

    meta = {
        "T1": {},
        "T1_low_resource_curve": {},
        "T2a": {},
        "T2b": {},
        "T2_2x2_supervision_ablation": {
            "T2a": {},
            "T2b": {},
            "T2b_dynamic_stop_supplementary": {},
            "fixed_frame_audit": {},
        },
    }

    t1_rows_cache = {}
    for k, (apths, jpth, mx, mr) in t1_src.items():
        if not apths:
            continue
        rows = build_t1_dataset(apths, jpth, max_n=mx, max_raw_stays=mr, gt2_only=False)
        t1_rows_cache[k] = rows
        art = runner.fit_t1_arm(args.fit_and_freeze, k, rows, t1_tune)
        if k in ("Real-50", "Real-100", "Real-Small (250)"):
            curve_k = "Real-250" if k == "Real-Small (250)" else k
            meta["T1_low_resource_curve"][curve_k] = art
        if k not in ("Real-50", "Real-100"):
            meta["T1"][k] = art

    for s_lbl, s_key in (("SynthFull", "Synth-Full"), ("SynthExported", "Synth-Exported (Release N=901)"), ("DIRECT32B", "DIRECT-32B")):
        if s_key not in t1_rows_cache:
            continue
        s_rows = t1_rows_cache[s_key]
        s_art = meta["T1"][s_key]
        for n_r, r_key in ((50, "Real-50"), (100, "Real-100"), (250, "Real-Small (250)")):
            if r_key not in t1_rows_cache:
                continue
            r_rows = t1_rows_cache[r_key]
            r_art = meta["T1"][r_key] if n_r == 250 else meta["T1_low_resource_curve"][r_key]
            aug_k = f"Augment (Real{n_r}+{s_lbl})"
            aug_art = runner.fit_t1_augment(
                args.fit_and_freeze, aug_k, r_key, s_key, r_rows, s_rows, r_art, s_art, t1_tune
            )
            meta["T1_low_resource_curve"][aug_k] = aug_art
            if n_r == 250:
                meta["T1"][aug_k] = aug_art
    t1_rows_cache.clear()
    print(f"NEURAL_FIT_T1 arms={len(meta['T1'])} curve={len(meta['T1_low_resource_curve'])}", flush=True)

    t2a_rows_cache = {}
    for k, (apths, jpth, mx, mr) in t2_src.items():
        if not apths:
            continue
        rows = build_t2a_clause_dataset(apths, jpth, max_stays=mx, max_raw_stays=mr)
        if k in ("Real-Small (250)", "Synth-Full (PostGate)", "Synth-Exported (Release N=901)"):
            t2a_rows_cache[k] = rows
        meta["T2a"][k] = runner.fit_t2a_arm(args.fit_and_freeze, k, rows, t2a_tune)
    for s_lbl, s_key in (("SynthFull", "Synth-Full (PostGate)"), ("SynthExported", "Synth-Exported (Release N=901)")):
        if "Real-Small (250)" in t2a_rows_cache and s_key in t2a_rows_cache:
            aug_k = f"Augment (Real250+{s_lbl})"
            meta["T2a"][aug_k] = runner.fit_t2a_augment(
                args.fit_and_freeze, aug_k, "Real-Small (250)", s_key,
                t2a_rows_cache["Real-Small (250)"], t2a_rows_cache[s_key],
                meta["T2a"]["Real-Small (250)"], meta["T2a"][s_key], t2a_tune,
            )
    t2a_rows_cache.clear()
    if fa and va:
        ff_t2a = build_t2a_fixed_frame_2x2(fa, fj, va, vj)
        for cell_k, cell_rows in ff_t2a.items():
            meta["T2_2x2_supervision_ablation"]["T2a"][cell_k] = runner.fit_t2a_arm(
                args.fit_and_freeze, f"2x2_ff_{cell_k}", cell_rows, t2a_tune
            )
    print(f"NEURAL_FIT_T2A arms={len(meta['T2a'])}", flush=True)

    t2b_rows_cache = {}
    for k, (apths, jpth, mx, mr) in t2_src.items():
        if not apths:
            continue
        rows = build_t2b_update_dataset(apths, jpth, max_stays=mx, max_raw_stays=mr)
        if k in ("Real-Small (250)", "Synth-Full (PostGate)", "Synth-Exported (Release N=901)"):
            t2b_rows_cache[k] = rows
        meta["T2b"][k] = runner.fit_t2b_arm(args.fit_and_freeze, k, rows, t2b_tune)
    for s_lbl, s_key in (("SynthFull", "Synth-Full (PostGate)"), ("SynthExported", "Synth-Exported (Release N=901)")):
        if "Real-Small (250)" in t2b_rows_cache and s_key in t2b_rows_cache:
            aug_k = f"Augment (Real250+{s_lbl})"
            meta["T2b"][aug_k] = runner.fit_t2b_augment(
                args.fit_and_freeze, aug_k, "Real-Small (250)", s_key,
                t2b_rows_cache["Real-Small (250)"], t2b_rows_cache[s_key],
                meta["T2b"]["Real-Small (250)"], meta["T2b"][s_key], t2b_tune,
            )
    t2b_rows_cache.clear()
    if fa and va:
        ff_t2b, ff_audit = build_t2b_fixed_frame_2x2(fa, fj, va, vj, with_context=False)
        meta["T2_2x2_supervision_ablation"]["fixed_frame_audit"] = ff_audit
        for cell_k, cell_rows in ff_t2b.items():
            meta["T2_2x2_supervision_ablation"]["T2b"][cell_k] = runner.fit_t2b_arm(
                args.fit_and_freeze, f"2x2_ff_{cell_k}", cell_rows, t2b_tune
            )
        for arm_lbl, pg_key, ap, jp in (
            ("Synth-Full", "Synth-Full (PostGate)", fa, fj),
            ("Synth-VisibilityOnly", "Synth-VisibilityOnly (PostGate)", va, vj),
        ):
            meta["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][f"{arm_lbl} | dynamic_planner"] = meta["T2b"][pg_key]
            dyn_tr = build_t2b_update_dataset(ap, jp, label_source="text_rule")
            dyn_k = f"{arm_lbl} | dynamic_text_rule"
            meta["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][dyn_k] = runner.fit_t2b_arm(
                args.fit_and_freeze, f"2x2_dyn_{dyn_k}", dyn_tr, t2b_tune
            )
    print(f"NEURAL_FIT_T2B arms={len(meta['T2b'])}", flush=True)

    bundle_meta = {
        "version": "v1.6-neural",
        "encoder_path": args.encoder_path,
        "rehearsal_split": bool(args.rehearsal_split),
        "tune_fold": args.val_fold,
        "tune_patient_count": len(tune_ids),
        "rehearsal_eval_ids": sorted(eval_ids) if eval_ids is not None else None,
        "expected_t1_keys": sorted(meta["T1"].keys()),
        "expected_t2_keys": sorted(meta["T2b"].keys()),
        "completed_shards": {k: len(v) for k, v in shard_lists.items()},
        "artifacts": meta,
    }
    meta_p = os.path.join(args.fit_and_freeze, "bundle_meta.json")
    with open(meta_p, "w", encoding="utf-8") as bf:
        json.dump(bundle_meta, bf, ensure_ascii=False, indent=2)

    source_hashes = {}
    for p in [os.path.abspath(__file__), args.val_fold, *fa, *va, *shard_lists["oracle_leak"], *direct_paths, *exported_paths]:
        if p and os.path.exists(p):
            source_hashes[p] = _sha256_file(p)
    manifest = {
        "version": "v1.6-neural",
        "encoder_path": args.encoder_path,
        "encoder_hashes": _encoder_weight_hashes(args.encoder_path),
        "bundle_meta_sha256": _sha256_file(meta_p),
        "checkpoint_hashes": _collect_ckpt_hashes(args.fit_and_freeze),
        "rehearsal_split": bool(args.rehearsal_split),
        "completed_shards": bundle_meta["completed_shards"],
        "expected_t1_keys": bundle_meta["expected_t1_keys"],
        "expected_t2_keys": bundle_meta["expected_t2_keys"],
        "source_hashes": source_hashes,
    }
    man_p = os.path.join(args.fit_and_freeze, "freeze_manifest.json")
    with open(man_p, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    print("NEURAL_FIT_FREEZE_DONE", json.dumps({"bundle_dir": args.fit_and_freeze, "bundle_meta_sha256": manifest["bundle_meta_sha256"], "n_ckpts": len(manifest["checkpoint_hashes"])}, ensure_ascii=False), flush=True)


def _predict_t1_meta(bundle_dir, arm_meta, eval_enc, encoder_path, device):
    preds = {}
    cache = {}

    def _run_ckpts(ckpts, col_idx):
        key = tuple(ckpts)
        if key not in cache:
            cache[key] = _eval_pt_files(bundle_dir, ckpts, encoder_path, 2, eval_enc, _logits_to_t1_probs, device)
        return [arr[:, col_idx] for arr in cache[key]]

    if arm_meta["kind"] == "t1_single":
        for idx, label in ((0, "T1a"), (1, "T1b")):
            info = arm_meta["targets"][label]
            if info.get("single_class"):
                preds[label] = None
            else:
                sp = _run_ckpts(info["ckpts"], idx)
                preds[label] = (np.mean(sp, axis=0), sp)
        return preds

    for idx, label in ((0, "T1a"), (1, "T1b")):
        info = arm_meta["targets"][label]
        if info.get("single_class"):
            preds[label] = None
            continue
        sp_r = _run_ckpts(info["real_ckpts"], idx)
        w = float(info["selected_weight"])
        if w > 0.0:
            sec_ckpts = info["pt_ft_ckpts"] if info["selected_strategy"] == "pt_ft_blend" else info["synth_ckpts"]
            sp_s = _run_ckpts(sec_ckpts, idx)
            sp = [(1.0 - w) * a + w * b for a, b in zip(sp_r, sp_s)]
        else:
            sp = sp_r
        preds[label] = (np.mean(sp, axis=0), sp)
    return preds


def _predict_t2a_meta(bundle_dir, arm_meta, eval_enc, encoder_path, device, single_cache):
    if arm_meta.get("empty") or arm_meta.get("single_class"):
        return None
    if arm_meta["kind"] == "t2a":
        ck = tuple(arm_meta["ckpts"])
        if ck not in single_cache:
            single_cache[ck] = _eval_pt_files(bundle_dir, arm_meta["ckpts"], encoder_path, 3, eval_enc, _logits_to_t2a_probs, device)
        probs = np.mean(single_cache[ck], axis=0)
        return np.argmax(probs, axis=1), probs
    rk = tuple(arm_meta["real_ckpts"])
    if rk not in single_cache:
        single_cache[rk] = _eval_pt_files(bundle_dir, list(rk), encoder_path, 3, eval_enc, _logits_to_t2a_probs, device)
    pr = np.mean(single_cache[rk], axis=0)
    w = float(arm_meta["selected_weight"])
    if w > 0.0:
        sec_k = tuple(arm_meta["pt_ft_ckpts"] if arm_meta["selected_strategy"] == "pt_ft_blend" else arm_meta["synth_ckpts"])
        if sec_k not in single_cache:
            single_cache[sec_k] = _eval_pt_files(bundle_dir, list(sec_k), encoder_path, 3, eval_enc, _logits_to_t2a_probs, device)
        ps = np.mean(single_cache[sec_k], axis=0)
        probs = (1.0 - w) * pr + w * ps
    else:
        probs = pr
    return np.argmax(probs, axis=1), probs


def _predict_t2b_meta(bundle_dir, arm_meta, eval_enc, encoder_path, device, single_cache):
    if arm_meta.get("empty") or arm_meta.get("single_class"):
        return None
    if arm_meta["kind"] == "t2b":
        ck = tuple(arm_meta["ckpts"])
        if ck not in single_cache:
            single_cache[ck] = _eval_pt_files(bundle_dir, arm_meta["ckpts"], encoder_path, 1, eval_enc, _logits_to_t2b_probs, device)
        return np.mean(single_cache[ck], axis=0)
    rk = tuple(arm_meta["real_ckpts"])
    if rk not in single_cache:
        single_cache[rk] = _eval_pt_files(bundle_dir, list(rk), encoder_path, 1, eval_enc, _logits_to_t2b_probs, device)
    pr = np.mean(single_cache[rk], axis=0)
    w = float(arm_meta["selected_weight"])
    if w > 0.0:
        sec_k = tuple(arm_meta["pt_ft_ckpts"] if arm_meta["selected_strategy"] == "pt_ft_blend" else arm_meta["synth_ckpts"])
        if sec_k not in single_cache:
            single_cache[sec_k] = _eval_pt_files(bundle_dir, list(sec_k), encoder_path, 1, eval_enc, _logits_to_t2b_probs, device)
        ps = np.mean(single_cache[sec_k], axis=0)
        return (1.0 - w) * pr + w * ps
    return pr


def eval_frozen_neural(args):
    if not HAS_TORCH_HF:
        raise RuntimeError("PyTorch and transformers are required for neural student evaluation")
    if not args.out:
        raise ValueError("--eval-frozen requires --out")

    man_p = os.path.join(args.eval_frozen, "freeze_manifest.json")
    meta_p = os.path.join(args.eval_frozen, "bundle_meta.json")
    if not (os.path.exists(man_p) and os.path.exists(meta_p)):
        raise FileNotFoundError(f"Missing frozen neural manifest or bundle_meta in {args.eval_frozen}")

    with open(man_p, encoding="utf-8") as mf:
        manifest = json.load(mf)
    if _sha256_file(meta_p) != manifest["bundle_meta_sha256"]:
        raise RuntimeError("FROZEN_NEURAL_HASH_MISMATCH: bundle_meta.json")
    for rel, expected_sha in manifest["checkpoint_hashes"].items():
        abs_p = os.path.join(args.eval_frozen, rel)
        if not os.path.exists(abs_p) or _sha256_file(abs_p) != expected_sha:
            raise RuntimeError(f"FROZEN_NEURAL_HASH_MISMATCH: {rel}")

    with open(meta_p, encoding="utf-8") as bf:
        bundle_meta = json.load(bf)
    artifacts = bundle_meta["artifacts"]
    for req_k in manifest.get("expected_t1_keys", ()):
        if req_k not in artifacts["T1"]:
            raise KeyError(f"Missing frozen neural T1 key: {req_k}")
    for req_k in manifest.get("expected_t2_keys", ()):
        if req_k not in artifacts["T2b"]:
            raise KeyError(f"Missing frozen neural T2b key: {req_k}")

    encoder_path = args.encoder_path or bundle_meta["encoder_path"]
    tok = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_target = args.eval_fold or bundle_meta["tune_fold"]
    eval_aid_filter = set(bundle_meta["rehearsal_eval_ids"]) if (bundle_meta.get("rehearsal_eval_ids") and not args.eval_fold) else None

    t1_eval = build_t1_dataset([eval_target], gt2_only=False)
    t2a_val = build_t2a_clause_dataset([eval_target], max_stays=(1600 if eval_aid_filter is None else None))
    t2b_val = build_t2b_update_dataset([eval_target], max_stays=(1600 if eval_aid_filter is None else None))
    if eval_aid_filter is not None:
        t1_eval = _filter_rows_aid(t1_eval, eval_aid_filter, 3)
        t2a_val = _filter_rows_aid(t2a_val, eval_aid_filter, 2)
        t2b_val = _filter_rows_aid(t2b_val, eval_aid_filter, 3)

    t1_enc = encode_head_tail_batch([r[0] for r in t1_eval], tok, 512)
    t2a_enc = encode_head_tail_batch([r[0] for r in t2a_val], tok, 256)
    t2b_enc = encode_head_tail_batch([r[0] for r in t2b_val], tok, 384)

    out = {
        "T1": {},
        "T1_low_resource_curve": {},
        "T2a": {},
        "T2b": {},
        "T2_2x2_supervision_ablation": {
            "T2a": {},
            "T2b": {},
            "T2b_dynamic_stop_supplementary": {},
            "fixed_frame_audit": artifacts["T2_2x2_supervision_ablation"].get("fixed_frame_audit", {}),
        },
        "paired_comparisons": {},
    }

    t1_p, lr_p = {}, {}
    for k, ameta in artifacts["T1"].items():
        pr = _predict_t1_meta(args.eval_frozen, ameta, t1_enc, encoder_path, device)
        t1_p[k] = pr
        out["T1"][k] = score_t1_neural(ameta, pr, t1_eval, n_boot=args.n_boot)
    for k, ameta in artifacts["T1_low_resource_curve"].items():
        if k == "Real-250" and "Real-Small (250)" in out["T1"]:
            lr_p[k] = t1_p["Real-Small (250)"]
            out["T1_low_resource_curve"][k] = out["T1"]["Real-Small (250)"]
        elif k in out["T1"]:
            lr_p[k] = t1_p[k]
            out["T1_low_resource_curve"][k] = out["T1"][k]
        else:
            pr = _predict_t1_meta(args.eval_frozen, ameta, t1_enc, encoder_path, device)
            lr_p[k] = pr
            out["T1_low_resource_curve"][k] = score_t1_neural(ameta, pr, t1_eval, n_boot=args.n_boot)
    print(f"NEURAL_EVAL_T1 n_eval={len(t1_eval)}", flush=True)

    t2a_cache = {}
    for k, ameta in artifacts["T2a"].items():
        pr = _predict_t2a_meta(args.eval_frozen, ameta, t2a_enc, encoder_path, device, t2a_cache)
        out["T2a"][k] = score_t2a_neural(ameta, pr, t2a_val, n_boot=args.n_boot)
    for k, ameta in artifacts["T2_2x2_supervision_ablation"]["T2a"].items():
        pr = _predict_t2a_meta(args.eval_frozen, ameta, t2a_enc, encoder_path, device, t2a_cache)
        out["T2_2x2_supervision_ablation"]["T2a"][k] = score_t2a_neural(ameta, pr, t2a_val, n_boot=args.n_boot)

    t2b_cache, t2b_p, t2b_ff_p, t2b_dyn_p = {}, {}, {}, {}
    for k, ameta in artifacts["T2b"].items():
        pr = _predict_t2b_meta(args.eval_frozen, ameta, t2b_enc, encoder_path, device, t2b_cache)
        t2b_p[k] = pr
        out["T2b"][k] = score_t2b_neural(ameta, pr, t2b_val, n_boot=args.n_boot)
    for k, ameta in artifacts["T2_2x2_supervision_ablation"]["T2b"].items():
        pr = _predict_t2b_meta(args.eval_frozen, ameta, t2b_enc, encoder_path, device, t2b_cache)
        t2b_ff_p[k] = pr
        out["T2_2x2_supervision_ablation"]["T2b"][k] = score_t2b_neural(ameta, pr, t2b_val, n_boot=args.n_boot)
    for k, ameta in artifacts["T2_2x2_supervision_ablation"].get("T2b_dynamic_stop_supplementary", {}).items():
        if k == "Synth-Full | dynamic_planner" and "Synth-Full (PostGate)" in out["T2b"]:
            t2b_dyn_p[k] = t2b_p["Synth-Full (PostGate)"]
            out["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][k] = out["T2b"]["Synth-Full (PostGate)"]
        elif k == "Synth-VisibilityOnly | dynamic_planner" and "Synth-VisibilityOnly (PostGate)" in out["T2b"]:
            t2b_dyn_p[k] = t2b_p["Synth-VisibilityOnly (PostGate)"]
            out["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][k] = out["T2b"]["Synth-VisibilityOnly (PostGate)"]
        else:
            pr = _predict_t2b_meta(args.eval_frozen, ameta, t2b_enc, encoder_path, device, t2b_cache)
            t2b_dyn_p[k] = pr
            out["T2_2x2_supervision_ablation"]["T2b_dynamic_stop_supplementary"][k] = score_t2b_neural(ameta, pr, t2b_val, n_boot=args.n_boot)
    print(f"NEURAL_EVAL_T2 n_t2a={len(t2a_val)} n_t2b={len(t2b_val)}", flush=True)

    # Paired cluster-bootstrap comparisons matching t123_bench
    all_t1_p = {**lr_p, **t1_p}
    g_t1 = [r[3] for r in t1_eval]
    for y_idx, t_lbl in ((1, "T1a"), (2, "T1b")):
        y_te = np.array([r[y_idx] for r in t1_eval], dtype=int)
        diffs = {}
        for d_name, a_arm, b_arm in (
            ("Augment_SynthFull_minus_Real250", "Augment (Real250+SynthFull)", "Real-Small (250)"),
            ("Augment_SynthExported_minus_Real250", "Augment (Real250+SynthExported)", "Real-Small (250)"),
            ("Augment_DIRECT32B_minus_Real250", "Augment (Real250+DIRECT32B)", "Real-Small (250)"),
            ("SynthFull_minus_VisOnly", "Synth-Full", "Synth-VisibilityOnly"),
            ("SynthFull600_minus_DIRECT32B", "Synth-Full (Matched N=600)", "DIRECT-32B"),
            ("SynthExported_minus_DIRECT32B", "Synth-Exported (Release N=901)", "DIRECT-32B"),
            ("SynthFull_minus_SynthExported", "Synth-Full", "Synth-Exported (Release N=901)"),
            ("Augment_Real50_SynthFull_minus_Real50", "Augment (Real50+SynthFull)", "Real-50"),
            ("Augment_Real100_SynthFull_minus_Real100", "Augment (Real100+SynthFull)", "Real-100"),
            ("Augment_Real50_SynthExported_minus_Real50", "Augment (Real50+SynthExported)", "Real-50"),
            ("Augment_Real100_SynthExported_minus_Real100", "Augment (Real100+SynthExported)", "Real-100"),
        ):
            pa = (all_t1_p.get(a_arm) or {}).get(t_lbl)
            pb = (all_t1_p.get(b_arm) or {}).get(t_lbl)
            if pa is not None and pb is not None:
                diffs[d_name] = paired_bootstrap_diff(y_te, pa[0], pb[0], g_t1, n_boot=args.n_boot)
        out["paired_comparisons"][t_lbl] = diffs

    non_sil = ~np.array([bool(r[2]) for r in t2b_val], dtype=bool)
    if np.any(non_sil):
        y_m = np.array([r[1] for r in t2b_val], dtype=int)[non_sil]
        g_m = [r[3] for i, r in enumerate(t2b_val) if non_sil[i]]
        ment_diffs = {}
        for d_name, a_arm, b_arm in (
            ("SynthFull_PostGate_minus_VisOnly_PostGate", "Synth-Full (PostGate)", "Synth-VisibilityOnly (PostGate)"),
            ("SynthFull_FirstTry_minus_VisOnly_FirstTry", "Synth-Full (FirstTry)", "Synth-VisibilityOnly (FirstTry)"),
            ("SynthFull600_minus_DIRECT32B", "Synth-Full (Matched N=600)", "DIRECT-32B"),
            ("Augment_SynthFull_minus_Real250", "Augment (Real250+SynthFull)", "Real-Small (250)"),
            ("SynthExported_minus_VisOnly_PostGate", "Synth-Exported (Release N=901)", "Synth-VisibilityOnly (PostGate)"),
            ("SynthExported_minus_DIRECT32B", "Synth-Exported (Release N=901)", "DIRECT-32B"),
            ("Augment_SynthExported_minus_Real250", "Augment (Real250+SynthExported)", "Real-Small (250)"),
            ("SynthFull_minus_SynthExported", "Synth-Full (PostGate)", "Synth-Exported (Release N=901)"),
        ):
            pa, pb = t2b_p.get(a_arm), t2b_p.get(b_arm)
            if pa is not None and pb is not None:
                ment_diffs[d_name] = paired_bootstrap_diff(y_m, pa[non_sil], pb[non_sil], g_m, n_boot=args.n_boot)
        out["paired_comparisons"]["T2b_mentioned_only"] = ment_diffs

        diffs_ff = {}
        for d_name, a_arm, b_arm in (
            ("Full_fixed_textrule_minus_VisOnly_fixed_textrule", "Synth-Full | fixed_text_rule", "Synth-VisibilityOnly | fixed_text_rule"),
            ("Full_fixed_planner_minus_Full_fixed_textrule", "Synth-Full | fixed_planner_tk", "Synth-Full | fixed_text_rule"),
            ("Full_fixed_planner_minus_VisOnly_fixed_planner", "Synth-Full | fixed_planner_tk", "Synth-VisibilityOnly | fixed_planner_tk"),
        ):
            pa, pb = t2b_ff_p.get(a_arm), t2b_ff_p.get(b_arm)
            if pa is not None and pb is not None:
                diffs_ff[d_name] = paired_bootstrap_diff(y_m, pa[non_sil], pb[non_sil], g_m, n_boot=args.n_boot)
        out["paired_comparisons"]["T2b_fixed_frame_2x2"] = diffs_ff

        if t2b_dyn_p:
            diffs_dyn = {}
            for d_name, a_arm, b_arm in (
                ("Full_dyn_textrule_minus_VisOnly_dyn_textrule", "Synth-Full | dynamic_text_rule", "Synth-VisibilityOnly | dynamic_text_rule"),
                ("Full_dyn_planner_minus_Full_dyn_textrule", "Synth-Full | dynamic_planner", "Synth-Full | dynamic_text_rule"),
            ):
                pa, pb = t2b_dyn_p.get(a_arm), t2b_dyn_p.get(b_arm)
                if pa is not None and pb is not None:
                    diffs_dyn[d_name] = paired_bootstrap_diff(y_m, pa[non_sil], pb[non_sil], g_m, n_boot=args.n_boot)
            out["paired_comparisons"]["T2b_dynamic_stop_2x2"] = diffs_dyn

    t2_key_path = args.t2_sealed_key
    if not t2_key_path and (not args.eval_fold or "fold1" in args.eval_fold.lower()):
        cand_key = os.path.join(args.s5_root, "blind_pools", "t2_fold1_v15", "T2_SEALED_SAMPLING_KEY.json")
        if os.path.exists(cand_key):
            t2_key_path = cand_key

    t2_blind_preds = {}
    blind_audit = {"status": "no_sealed_key_provided"}
    if t2_key_path and os.path.exists(t2_key_path) and os.path.exists(eval_target):
        blind_rows, blind_key_rows = build_blind_window_eval_rows(eval_target, t2_key_path, with_context=False)
        blind_enc = encode_head_tail_batch([r[0] for r in blind_rows], tok, 384)
        expected_wkeys = [km["window_key"] for km in blind_key_rows]
        per_stratum = dict(collections.Counter(km["auto_stratum"] for km in blind_key_rows))
        blind_cache = {}
        for arm_nm, ameta in artifacts["T2b"].items():
            pr = _predict_t2b_meta(args.eval_frozen, ameta, blind_enc, encoder_path, device, blind_cache)
            if pr is None or len(pr) != len(expected_wkeys):
                continue
            t_cal = float(ameta.get("thr_cal", 0.5))
            t2_blind_preds[arm_nm] = (
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

    human_gold_rep = {"status": "pending_clinician_annotation"}
    if args.t2_human_gold and t2_key_path and os.path.exists(args.t2_human_gold) and t2_blind_preds:
        human_gold_rep = score_t2_human_gold_multiarm(t2_blind_preds, args.t2_human_gold, t2_key_path)

    results = {
        "mode": "eval_frozen",
        "version": "v1.6-neural",
        "encoder_path": encoder_path,
        "bundle_dir": args.eval_frozen,
        "bundle_meta_sha256": manifest["bundle_meta_sha256"],
        "eval_fold": args.eval_fold or "fold1_rehearsal_split_eval",
        "n_boot": args.n_boot,
        "completed_shards": manifest.get("completed_shards", {}),
        "data_audit": {"eval_fold": audit_fold_docs(eval_target)} if os.path.exists(eval_target) else {},
        **out,
        "blind_pack_prediction_audit": blind_audit,
        "T2_human_gold": human_gold_rep,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("NEURAL_EVAL_FROZEN_DONE", json.dumps({"out": args.out, "bundle_meta_sha256": manifest["bundle_meta_sha256"]}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-and-freeze", default=None, help="Fit and freeze Stage-1 neural checkpoints")
    ap.add_argument("--rehearsal-split", action="store_true", help="Hold out 50%% fold1 patients for rehearsal eval")
    ap.add_argument("--rehearsal-split-json", default=None, help="Optional JSON containing exact tune_ids and eval_ids")
    ap.add_argument("--eval-frozen", default=None, help="Load frozen neural bundle for strict read-only eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--encoder-path", default=DEFAULT_ENCODER_PATH)
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r1")
    ap.add_argument("--val-fold", default="/home/weify/pyt/data/deid_run9_fold1.jsonl")
    ap.add_argument("--eval-fold", default=None)
    ap.add_argument("--real-folds", nargs="*", default=None, help="Optional real training fold JSONL paths")
    ap.add_argument("--t2-human-gold", default=None)
    ap.add_argument("--t2-sealed-key", default=None)
    ap.add_argument("--exported-synth-jsonl", default=None)
    ap.add_argument("--direct32b-jsonl", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-epochs-override", type=int, default=None, help="Override epochs for fast unit test smoke")
    args = ap.parse_args()

    if args.fit_and_freeze:
        fit_and_freeze_neural(args)
    elif args.eval_frozen:
        eval_frozen_neural(args)
    else:
        raise ValueError("Specify --fit-and-freeze or --eval-frozen")


if __name__ == "__main__":
    main()
