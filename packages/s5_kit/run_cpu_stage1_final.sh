#!/bin/bash
set -uo pipefail
umask 077

BASE=/home/weify/pyt
ROOT=$BASE/product/s5_r1
PY="$BASE/env/bin/python"
ENC="$BASE/models/bge-small-zh-v1.5"

source "$BASE/gates/S0/v02/code/env.sh"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/code:$BASE/product/v031_s3/medcourse_core:$BASE/gates/S0/bench_core:$BASE/gates/S0:$BASE/gates/G3/g2_core:$BASE/gates/Q45b/g45_core"

mkdir -p "$ROOT/tables" "$ROOT/logs" "$ROOT/frozen_stage1_v16" "$ROOT/frozen_rehearsal_v16" "$ROOT/frozen_mia_stage1_v16"
rm -rf "$ROOT/frozen_rehearsal_1k_v15"
echo "[$(date '+%F %T')] CPU_STAGE1_FINAL_V16_WAITING"

WAITED=0
while true; do
  main_alive=0
  extra_alive=0
  if pgrep -u "$(id -un)" -f "run_s5_gpu_daemon\.sh" >/dev/null 2>&1; then
    main_alive=1
  fi
  if pgrep -u "$(id -un)" -f "run_extra_gpu_worker\.sh" >/dev/null 2>&1; then
    extra_alive=1
  fi
  if [ "$main_alive" -eq 0 ] && [ "$extra_alive" -eq 0 ]; then
    break
  fi
  if [ "$WAITED" -ge 57600 ]; then
    echo "[$(date '+%F %T')] CPU_WAIT_TIMEOUT"
    exit 1
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

echo "[$(date '+%F %T')] GPU_WORKERS_FINISHED -> running v1.6 Stage-1 final lock & rehearsal"

# Ensure de-identified blind pools exist without mutating sealed keys once created
T2_POOL_ARG=()
T1_POOL_ARG=()
if [ ! -f "$ROOT/blind_pools/t2_fold1_v15/T2_SEALED_SAMPLING_KEY.json" ]; then
  T2_POOL_ARG=(--export-t2-pool "$ROOT/blind_pools/t2_fold1_v15")
fi
if [ ! -f "$ROOT/blind_pools/t1_fold1_v15/T1_SEALED_AUDIT_KEY.json" ]; then
  T1_POOL_ARG=(--export-t1-pool "$ROOT/blind_pools/t1_fold1_v15")
fi

# 1. Rehearsal on fold 1 (50% tune -> freeze -> 50% read-only eval + 210-window blind bijection + private CONSORT IDs, B=2000)
"$PY" -B "$ROOT/code/t123_bench.py" \
  --s5-root "$ROOT" \
  --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
  --fit-and-freeze "$ROOT/frozen_rehearsal_v16" \
  --rehearsal-split \
  "${T2_POOL_ARG[@]}" \
  "${T1_POOL_ARG[@]}" || exit 1

"$PY" -B "$ROOT/code/t123_bench.py" \
  --s5-root "$ROOT" \
  --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
  --t2-sealed-key "$ROOT/blind_pools/t2_fold1_v15/T2_SEALED_SAMPLING_KEY.json" \
  --export-consort-ids-private "$ROOT/tables/fold1_t1_consort_ids_PRIVATE.json" \
  --eval-frozen "$ROOT/frozen_rehearsal_v16" \
  --n-boot 2000 \
  --out "$ROOT/tables/rehearsal_t123_fold1_final2k_v16.json" || exit 1

# 2. Freeze final 100%-fold1 bundle ready for Stage-2 fold9 --eval-frozen (ZERO fold9 contact)
"$PY" -B "$ROOT/code/t123_bench.py" \
  --s5-root "$ROOT" \
  --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
  --fit-and-freeze "$ROOT/frozen_stage1_v16" || exit 1

# 3. Unified text-feature 17-pathway probe (ge2 + gt2 + within-pathway c-stat)
"$PY" -B "$ROOT/code/audit_s5_2_17d.py" \
  --s5-root "$ROOT" \
  --out "$ROOT/tables/s5_2_17d_text_probe_final_v16.json" || exit 1

# 4. Intrinsic audit across all Stage-1 corpora
"$PY" -B "$ROOT/code/audit_s5_intrinsic.py" \
  --s5-root "$ROOT" \
  --out-json "$ROOT/tables/intrinsic_stage1_final2k_v16.json" \
  --out-csv "$ROOT/tables/intrinsic_stage1_final2k_v16.csv" || exit 1

# 5. Lock MIA Stage-1 calibration scorers + bidirectional 1% FPR thresholds + composition shift audit (B=2000)
"$PY" -B "$ROOT/code/s6_mia_audit.py" \
  --mode stage1_cal \
  --s5-root "$ROOT" \
  --n-boot 2000 \
  --exported-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
  --freeze-cal-dir "$ROOT/frozen_mia_stage1_v16" \
  --out "$ROOT/tables/mia_stage1_cal_final2k_v16.json" || exit 1

{
  echo "# Stage-1 v1.6 Locked SHA256 ($(date -u '+%F %T UTC'))"
  sha256sum "$ROOT"/code/*.py "$ROOT"/code/*.sh
  sha256sum "$ROOT"/frozen_stage1_v16/frozen_bundle.pkl "$ROOT"/frozen_stage1_v16/freeze_manifest.json
  sha256sum "$ROOT"/frozen_mia_stage1_v16/mia_frozen_cal.pkl "$ROOT"/frozen_mia_stage1_v16/freeze_manifest.json
  sha256sum "$ROOT"/blind_pools/t1_fold1_v15/T1_SEALED_AUDIT_KEY.json "$ROOT"/blind_pools/t2_fold1_v15/T2_SEALED_SAMPLING_KEY.json
  sha256sum "$ROOT"/tables/fold1_t1_consort_ids_PRIVATE.json
  sha256sum "$ROOT"/generated/s5_1/*_shard*/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/s5_2/*/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/s5_ext/direct_32b/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/synth_exported_final2k_v15/admissions.jsonl 2>/dev/null || true
} > "$ROOT/tables/STAGE1_LOCKED_SHA256.txt"

echo "[$(date '+%F %T')] CPU_STAGE1_FINAL_V16_DONE"
