#!/bin/bash
set -uo pipefail
umask 077

BASE=/home/weify/pyt
ROOT=$BASE/product/s5_r1
PY_CPU="$BASE/env/bin/python"
PY_GPU="$BASE/env/vllm032/bin/python"
ENC="$BASE/models/bge-small-zh-v1.5"
SPLIT_JSON="$ROOT/frozen_rehearsal_v16/rehearsal_split_fold1.json"

source "$BASE/gates/S0/v02/code/env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/code:$BASE/product/v031_s3/medcourse_core:$BASE/gates/S0/bench_core:$BASE/gates/S0:$BASE/gates/G3/g2_core:$BASE/gates/Q45b/g45_core"

mkdir -p "$ROOT/tables" "$ROOT/logs" "$ROOT/frozen_neural_rehearsal_v16" "$ROOT/frozen_neural_stage1_v16"
echo "[$(date '+%F %T')] NEURAL_STAGE1_FINAL_V16_START"

# 1. Export exact fold1 50/50 split matching frozen_rehearsal_v16
"$PY_CPU" -B -c '
import json, pickle, t123_bench
tune_ids, eval_ids = t123_bench._rehearsal_patient_split("/home/weify/pyt/data/deid_run9_fold1.jsonl")
with open("/home/weify/pyt/product/s5_r1/frozen_rehearsal_v16/frozen_bundle.pkl", "rb") as f:
    b = pickle.load(f)
assert sorted(eval_ids) == b["rehearsal_eval_ids"], "rehearsal_eval_ids mismatch"
with open("/home/weify/pyt/product/s5_r1/frozen_rehearsal_v16/rehearsal_split_fold1.json", "w", encoding="utf-8") as out:
    json.dump({"tune_ids": sorted(tune_ids), "eval_ids": sorted(eval_ids)}, out, indent=2)
print(f"SPLIT_EXPORT_OK tune={len(tune_ids)} eval={len(eval_ids)}", flush=True)
' || exit 1

# 2. Pick up to 2 completely free GPUs among 1..7 (NEVER GPU 0)
mapfile -t IDLE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '
    { idx=$1+0; mem=$2+0; if (idx != 0 && mem < 500) print idx }
  ' | head -n 2
)

if [ "${#IDLE_GPUS[@]}" -lt 1 ]; then
  echo "NO_IDLE_GPU_IN_1_7"
  exit 1
fi

GPU_A="${IDLE_GPUS[0]}"
GPU_B="${IDLE_GPUS[1]:-}"
echo "[$(date '+%F %T')] SELECTED_GPUS gpu_a=${GPU_A} gpu_b=${GPU_B:-NONE}"

run_rehearsal() {
  local g="$1"
  CUDA_VISIBLE_DEVICES="$g" "$PY_GPU" -B "$ROOT/code/t12_neural_bench.py" \
    --s5-root "$ROOT" \
    --encoder-path "$ENC" \
    --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
    --fit-and-freeze "$ROOT/frozen_neural_rehearsal_v16" \
    --rehearsal-split \
    --rehearsal-split-json "$SPLIT_JSON" \
    > "$ROOT/logs/neural_rehearsal_v16.log" 2>&1 || return 1
  CUDA_VISIBLE_DEVICES="$g" "$PY_GPU" -B "$ROOT/code/t12_neural_bench.py" \
    --s5-root "$ROOT" \
    --encoder-path "$ENC" \
    --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
    --t2-sealed-key "$ROOT/blind_pools/t2_fold1_v15/T2_SEALED_SAMPLING_KEY.json" \
    --eval-frozen "$ROOT/frozen_neural_rehearsal_v16" \
    --n-boot 2000 \
    --out "$ROOT/tables/rehearsal_neural_fold1_final2k_v16.json" \
    >> "$ROOT/logs/neural_rehearsal_v16.log" 2>&1 || return 1
  echo "[$(date '+%F %T')] WORKER_REHEARSAL_DONE" >> "$ROOT/logs/neural_rehearsal_v16.log"
}

run_stage1_lock() {
  local g="$1"
  CUDA_VISIBLE_DEVICES="$g" "$PY_GPU" -B "$ROOT/code/t12_neural_bench.py" \
    --s5-root "$ROOT" \
    --encoder-path "$ENC" \
    --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v15/admissions.jsonl" \
    --fit-and-freeze "$ROOT/frozen_neural_stage1_v16" \
    > "$ROOT/logs/neural_stage1_lock_v16.log" 2>&1 || return 1
  echo "[$(date '+%F %T')] WORKER_STAGE1_LOCK_DONE" >> "$ROOT/logs/neural_stage1_lock_v16.log"
}

if [ -n "${GPU_B}" ]; then
  run_rehearsal "$GPU_A" &
  PID_A=$!
  run_stage1_lock "$GPU_B" &
  PID_B=$!
  wait "$PID_A"
  RC_A=$?
  wait "$PID_B"
  RC_B=$?
  echo "[$(date '+%F %T')] PARALLEL_WORKERS_DONE rc_a=$RC_A rc_b=$RC_B"
  cat "$ROOT/logs/neural_rehearsal_v16.log"
  cat "$ROOT/logs/neural_stage1_lock_v16.log"
  if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then
    exit 1
  fi
else
  run_rehearsal "$GPU_A" || { cat "$ROOT/logs/neural_rehearsal_v16.log"; exit 1; }
  run_stage1_lock "$GPU_A" || { cat "$ROOT/logs/neural_stage1_lock_v16.log"; exit 1; }
  cat "$ROOT/logs/neural_rehearsal_v16.log"
  cat "$ROOT/logs/neural_stage1_lock_v16.log"
fi

{
  echo "# Stage-1 v1.6 Locked SHA256 ($(date -u '+%F %T UTC'))"
  sha256sum "$ROOT"/code/*.py "$ROOT"/code/*.sh
  sha256sum "$ROOT"/frozen_stage1_v16/frozen_bundle.pkl "$ROOT"/frozen_stage1_v16/freeze_manifest.json
  sha256sum "$ROOT"/frozen_neural_stage1_v16/bundle_meta.json "$ROOT"/frozen_neural_stage1_v16/freeze_manifest.json
  sha256sum "$ROOT"/frozen_mia_stage1_v16/mia_frozen_cal.pkl "$ROOT"/frozen_mia_stage1_v16/freeze_manifest.json
  sha256sum "$ENC/pytorch_model.bin"
  sha256sum "$ROOT"/blind_pools/t1_fold1_v15/T1_SEALED_AUDIT_KEY.json "$ROOT"/blind_pools/t2_fold1_v15/T2_SEALED_SAMPLING_KEY.json
  sha256sum "$ROOT"/tables/fold1_t1_consort_ids_PRIVATE.json
  sha256sum "$ROOT"/generated/s5_1/*_shard*/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/s5_2/*/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/s5_ext/direct_32b/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/synth_exported_final2k_v15/admissions.jsonl 2>/dev/null || true
} > "$ROOT/tables/STAGE1_LOCKED_SHA256.txt"

echo "[$(date '+%F %T')] NEURAL_STAGE1_FINAL_V16_DONE"
