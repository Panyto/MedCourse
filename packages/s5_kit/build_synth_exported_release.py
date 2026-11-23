# -*- coding: utf-8 -*-
"""Build and verify the zero-rare-30gram Synth-Exported Stage-1 release artifact.

Strictly scans only allowed reference real folds (default folds 2-7) and hard-refuses fold 9.
Reuses s6_mia_audit.build_full_exported_artifact so manifest fields and SHA256 lock match S6-1.
"""
import argparse
import glob
import json
import os

from s6_mia_audit import assert_no_fold9, build_full_exported_artifact, _read_jsonl
from t123_bench import _completed_shard_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s5-root", default="/home/weify/pyt/product/s5_r2")
    ap.add_argument(
        "--real-folds",
        nargs="+",
        default=[f"/home/weify/pyt/data/deid_run9_fold{k}.jsonl" for k in range(2, 8)],
    )
    ap.add_argument("--full-adms", nargs="*", default=None, help="Optional explicit full admissions JSONL paths")
    ap.add_argument("--min-kept", type=int, default=400)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    expanded_real = []
    for pat in args.real_folds:
        hits = sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[]") else [pat]
        expanded_real.extend(hits or [pat])

    assert_no_fold9(expanded_real + [args.out_dir, args.s5_root], "stage1_cal")
    for bad in ("fold1", "fold8", "fold9"):
        if any(bad in os.path.basename(p).lower() for p in expanded_real):
            raise ValueError(f"REF_CORPUS_MUST_NOT_CONTAIN_{bad.upper()}: {expanded_real}")

    if args.full_adms:
        full_paths = list(args.full_adms)
    else:
        full_paths, _ = _completed_shard_paths(args.s5_root, "s5_1", "full_shard", min_kept=args.min_kept)
    if not full_paths:
        raise FileNotFoundError(f"No completed full_shard admissions found under {args.s5_root}")

    synth_full = [a for p in full_paths for a in _read_jsonl(p)]
    kept, manifest = build_full_exported_artifact(synth_full, expanded_real, args.out_dir)
    if int(manifest.get("after_g30_candidates_with_rare", -1)) != 0:
        raise RuntimeError(f"Post-filter rare 30-gram verification failed: {manifest}")
    print("BUILD_SYNTH_EXPORTED_DONE", json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
