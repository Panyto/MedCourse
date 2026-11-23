#!/bin/bash
set -uo pipefail
umask 077

BASE=/home/weify/pyt
ROOT=$BASE/product/s5_r2
PORT=8106
SERVE_LOG="$ROOT/logs/vllm_${PORT}.log"
PY="$BASE/env/bin/python"
PY_GPU="$BASE/env/vllm032/bin/python"
VL="$BASE/env/vllm032/bin/vllm"
ENC="$BASE/models/bge-small-zh-v1.5"
MODEL_DIR="$BASE/models/Qwen3-32B-AWQ"
SERVED_MODEL="Qwen3-32B-AWQ"

source "$BASE/.secrets/llm.env"
source "$BASE/gates/S0/v02/code/env.sh"
export MEDCOURSE_CORE_DIR="$BASE/product/v037_s5r2/medcourse_core"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/.cache"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/code:$MEDCOURSE_CORE_DIR:$BASE/gates/S0/bench_core:$BASE/gates/S0:$BASE/gates/G3/g2_core:$BASE/gates/Q45b/g45_core"

mkdir -p "$ROOT/logs" "$ROOT/generated/s5_2" "$ROOT/generated/s5_1" "$ROOT/generated/s5_ext/direct_32b" \
         "$ROOT/tables" "$ROOT/blind_pools" "$ROOT/frozen_rehearsal_v17" "$ROOT/frozen_stage1_v17" \
         "$ROOT/frozen_mia_stage1_v17" "$ROOT/frozen_neural_rehearsal_v17" "$ROOT/frozen_neural_stage1_v17" \
         "$BASE/tmp" "$BASE/.cache"

cleanup_vllm() {
  for p in $(pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" || true); do
    pkill -TERM -P "$p" 2>/dev/null || true
    kill -TERM "$p" 2>/dev/null || true
  done
  for _ in $(seq 1 15); do
    pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" >/dev/null 2>&1 || break
    sleep 1
  done
  for p in $(pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" || true); do
    pkill -KILL -P "$p" 2>/dev/null || true
    kill -KILL "$p" 2>/dev/null || true
  done
  sleep 2
}

trap cleanup_vllm EXIT INT TERM

probe_free_2gpus() {
  "$PY" - <<'PY'
import os, subprocess, sys

base = os.path.realpath("/home/weify/pyt")
my_uid = os.getuid()

def is_ours(pid):
    try:
        if os.stat(f"/proc/{pid}").st_uid != my_uid:
            return False
    except OSError:
        return False
    for link in ("exe", "cwd"):
        try:
            r = os.path.realpath(f"/proc/{pid}/{link}")
            if r == base or r.startswith(base + "/"):
                return True
        except OSError:
            pass
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore")
        if base in cmd:
            return True
    except OSError:
        pass
    return False

gpus = {}
for line in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
).splitlines():
    ps = [p.strip() for p in line.split(",")]
    if len(ps) == 4 and ps[0].isdigit():
        gpus[int(ps[0])] = {"uuid": ps[1], "mem": int(ps[2]), "util": int(ps[3])}

busy, mine = set(), set()
for line in subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
).splitlines():
    ps = [p.strip() for p in line.split(",")]
    if len(ps) >= 2 and ps[0].startswith("GPU-") and ps[1].isdigit():
        busy.add(ps[0])
        if is_ours(int(ps[1])):
            mine.add(ps[0])

avail = [
    g for g in (6, 7, 4, 5, 3, 2, 1)
    if g != 0 and g in gpus and gpus[g]["uuid"] not in busy and gpus[g]["mem"] < 500 and gpus[g]["util"] == 0
]
cap = 2 - len(mine)
if cap < 2 or len(avail) < 2:
    sys.exit(1)
print(f"{avail[0]},{avail[1]}")
PY
}

served_ok() {
  "$PY" - <<'PY' 2>/dev/null
import json, os, urllib.request
req = urllib.request.Request(
    "http://127.0.0.1:8106/v1/models",
    headers={"Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")},
)
try:
    print(",".join(m["id"] for m in json.load(urllib.request.urlopen(req, timeout=5))["data"]))
except Exception:
    pass
PY
}

job_done() {
  local out_dir="$1"
  local min_kept="$2"
  "$PY" - "$out_dir/counts.json" "$out_dir/admissions.jsonl" "$min_kept" <<'PY' 2>/dev/null
import json, os, sys
cp, ap, mk = sys.argv[1], sys.argv[2], int(sys.argv[3])
if not (os.path.exists(cp) and os.path.exists(ap)):
    sys.exit(1)
c = json.load(open(cp, encoding="utf-8"))
if int(c.get("llm_errors", 0)) == 0 and int(c.get("kept", 0)) >= mk:
    sys.exit(0)
sys.exit(1)
PY
}

reuse_direct32b_if_ready() {
  if job_done "$ROOT/generated/s5_ext/direct_32b" 580; then
    echo "[$(date '+%F %T')] DIRECT32B_ALREADY_PRESENT"
    return 0
  fi
  local src="$BASE/product/s5_r1/generated/s5_ext/direct_32b"
  if job_done "$src" 580; then
    echo "[$(date '+%F %T')] COPYING_ZERO_PRIOR_DIRECT32B_FROM_S5_R1"
    cp -a "$src"/. "$ROOT/generated/s5_ext/direct_32b"/
    return 0
  fi
  return 1
}

merge_direct_shards() {
  "$PY" - "$ROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
merged_dir = os.path.join(root, "generated", "s5_ext", "direct_32b")
os.makedirs(merged_dir, exist_ok=True)
kept = shards = 0
tmp_adm = os.path.join(merged_dir, "admissions.jsonl.tmp")
with open(tmp_adm, "w", encoding="utf-8") as out:
    for i in range(4):
        sdir = os.path.join(root, "generated", "s5_ext", f"direct_32b_shard{i}")
        cp = os.path.join(sdir, "counts.json")
        ap = os.path.join(sdir, "admissions.jsonl")
        if not (os.path.exists(cp) and os.path.exists(ap)):
            continue
        c = json.load(open(cp, encoding="utf-8"))
        if int(c.get("llm_errors", 0)) != 0 or int(c.get("kept", 0)) < 100:
            continue
        shards += 1
        with open(ap, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line)
                    kept += 1
os.replace(tmp_adm, os.path.join(merged_dir, "admissions.jsonl"))
with open(os.path.join(merged_dir, "counts.json"), "w", encoding="utf-8") as f:
    json.dump({"s5_arm": "direct_32b", "completed_shards": shards, "kept": kept, "llm_errors": 0}, f, indent=2)
PY
}

all_jobs_done() {
  job_done "$ROOT/generated/s5_ext/direct_32b" 580 || return 1
  job_done "$ROOT/generated/s5_2/route_b" 300 || return 1
  job_done "$ROOT/generated/s5_2/route_a_only" 300 || return 1
  job_done "$ROOT/generated/s5_2/route_ab" 300 || return 1
  for s in 0 1 2 3; do
    for arm in full vis_only oracle_leak; do
      job_done "$ROOT/generated/s5_1/${arm}_shard${s}" 450 || return 1
    done
  done
  return 0
}

run_smoke10_gate_if_needed() {
  local smoke_attempt="$ROOT/tables/v037_live_smoke10_last_attempt.json"
  local smoke_stamp="$ROOT/tables/v037_live_smoke10_passed.stamp.json"
  if "$PY" -B - "$MEDCOURSE_CORE_DIR" "$ROOT/planned/s5_1_shards/shard_0/journeys.jsonl" "$SERVED_MODEL" "TP2_8192_0.88" "$smoke_stamp" <<'PY'
import sys, s5_common
core_dir, pj_path, model, cfg, stamp_p = sys.argv[1:6]
fp = s5_common.compute_smoke10_fingerprint(core_dir, pj_path, model, cfg)
sys.exit(0 if s5_common.smoke10_stamp_valid(stamp_p, fp) else 1)
PY
  then
    echo "[$(date '+%F %T')] SMOKE10_ALREADY_PASSED"
    return 0
  fi
  if [ "$(served_ok)" != "$SERVED_MODEL" ]; then
    return 2
  fi
  echo "[$(date '+%F %T')] START_SMOKE10_GATE (10 stays shard_0 full)"
  rm -rf "$ROOT/tmp_smoke10_plan" "$ROOT/smoke_test_full10"
  mkdir -p "$ROOT/tmp_smoke10_plan"
  head -n 10 "$ROOT/planned/s5_1_shards/shard_0/journeys.jsonl" > "$ROOT/tmp_smoke10_plan/journeys.jsonl"
  "$PY" - "$ROOT" <<'PY' || return 1
import json, sys
r = sys.argv[1]
ids = {json.loads(l)["journey_id"] for l in open(f"{r}/tmp_smoke10_plan/journeys.jsonl", encoding="utf-8") if l.strip()}
for fn in ("specs.jsonl", "drafts.jsonl"):
    with open(f"{r}/tmp_smoke10_plan/{fn}", "w", encoding="utf-8") as out:
        for l in open(f"{r}/planned/s5_1_shards/shard_0/{fn}", encoding="utf-8"):
            if l.strip() and json.loads(l)["slot_id"] in ids:
                out.write(l)
PY

  "$PY" -B "$ROOT/code/gen_s5_shard.py" \
    --plan-dir "$ROOT/tmp_smoke10_plan" --arm full \
    --stats "$BASE/product/derived/m2_f27_v021/stats_v021.json" \
    --writer llm --base-url "http://127.0.0.1:${PORT}/v1" --model "$SERVED_MODEL" \
    --seed 20261210 --concurrency 16 --save-first-try --out "$ROOT/smoke_test_full10" || return 1

  "$PY" - "$ROOT" "$MEDCOURSE_CORE_DIR" "$SERVED_MODEL" "$smoke_attempt" "$smoke_stamp" <<'PY' || { rm -rf "$ROOT/tmp_smoke10_plan" "$ROOT/smoke_test_full10"; return 99; }
import json, sys
import s5_common
from medcourse import gate, planner
from g2_kit import oracle2

r, core_dir, served_model, attempt_p, stamp_p = sys.argv[1:6]
c = json.load(open(f"{r}/smoke_test_full10/counts.json", encoding="utf-8"))
adms = [json.loads(l) for l in open(f"{r}/smoke_test_full10/admissions.jsonl", encoding="utf-8") if l.strip()]
js = {j["journey_id"]: j for j in (json.loads(l) for l in open(f"{r}/smoke_test_full10/journeys.jsonl", encoding="utf-8") if l.strip())}
adm_fails = oracle_hits = 0
for a in adms:
    j = js[a["admission_id"]]
    if gate.check_admission(j, a["docs"]):
        adm_fails += 1
    odocs = [{"k": d.get("k", i), "role": d.get("role"), "day": int(d["day_offset"]), "text": d["text"]} for i, d in enumerate(a["docs"])]
    _, o_sum = oracle2.oracle_leaks_v2(planner.journey_to_draft(j), None, odocs)
    if o_sum.get("any"):
        oracle_hits += 1
cfg_tag = "TP2_8192_0.88"
c["smoke_mode_cfg"] = cfg_tag
c["delivered_adm_gate_fails"] = adm_fails
c["delivered_oracle_any_hits"] = oracle_hits
fp = s5_common.compute_smoke10_fingerprint(core_dir, f"{r}/planned/s5_1_shards/shard_0/journeys.jsonl", served_model, cfg_tag)
print("LIVE_SMOKE10_DONE", json.dumps(dict(c, input_fingerprint=fp), ensure_ascii=False), flush=True)
s5_common.record_smoke10_attempt_and_stamp(attempt_p, stamp_p, c, fp)
PY
  rm -rf "$ROOT/tmp_smoke10_plan" "$ROOT/smoke_test_full10"
  echo "[$(date '+%F %T')] SMOKE10_GATE_PASSED"
  return 0
}

run_one_job() {
  local plan_dir="$1"
  local arm="$2"
  local seed="$3"
  local out_dir="$4"
  local min_kept="$5"
  local extra_flag="$6"
  if job_done "$out_dir" "$min_kept"; then
    echo "[$(date '+%F %T')] SKIP_DONE $out_dir"
    return 0
  fi
  if [ "$(served_ok)" != "$SERVED_MODEL" ]; then
    echo "[$(date '+%F %T')] VLLM_UNHEALTHY_BEFORE $out_dir"
    return 2
  fi
  echo "[$(date '+%F %T')] START $arm -> $out_dir"
  rm -rf "$out_dir"
  "$PY" -B "$ROOT/code/gen_s5_shard.py" \
    --plan-dir "$plan_dir" --arm "$arm" \
    --stats "$BASE/product/derived/m2_f27_v021/stats_v021.json" \
    --writer llm --base-url "http://127.0.0.1:${PORT}/v1" --model "$SERVED_MODEL" \
    --seed "$seed" --concurrency 24 $extra_flag --out "$out_dir"
  local rc=$?
  echo "[$(date '+%F %T')] FINISH $arm rc=$rc out=$out_dir"
  return $rc
}

run_direct_if_needed() {
  if job_done "$ROOT/generated/s5_ext/direct_32b" 580; then
    return 0
  fi
  for i in 0 1 2 3; do
    local odir="$ROOT/generated/s5_ext/direct_32b_shard${i}"
    if ! job_done "$odir" 120; then
      if [ "$(served_ok)" != "$SERVED_MODEL" ]; then
        return 2
      fi
      rm -rf "$odir"
      "$PY" -B "$ROOT/code/gen_direct_32b.py" \
        --out "$odir" --n 150 --shard-idx "$i" --seed 20261230 \
        --writer llm --base-url "http://127.0.0.1:${PORT}/v1" --model "$SERVED_MODEL" \
        --concurrency 20 || return $?
    fi
  done
  merge_direct_shards
}

run_queue() {
  run_smoke10_gate_if_needed || return $?
  run_direct_if_needed || return $?
  run_one_job "$ROOT/planned/s5_2_17d/route_b" route_b 20261220 "$ROOT/generated/s5_2/route_b" 300 "--save-first-try" || return $?
  run_one_job "$ROOT/planned/s5_2_17d/route_a_only" route_a_only 20261220 "$ROOT/generated/s5_2/route_a_only" 300 "--save-first-try" || return $?
  run_one_job "$ROOT/planned/s5_2_17d/route_ab" route_ab 20261220 "$ROOT/generated/s5_2/route_ab" 300 "--save-first-try" || return $?

  local seeds=(20261210 20261211 20261212 20261213)
  for s in 0 1 2 3; do
    local sd="${seeds[$s]}"
    local pdir="$ROOT/planned/s5_1_shards/shard_${s}"
    run_one_job "$pdir" full "$sd" "$ROOT/generated/s5_1/full_shard${s}" 450 "--save-first-try" || return $?
    run_one_job "$pdir" vis_only "$sd" "$ROOT/generated/s5_1/vis_only_shard${s}" 450 "--save-first-try" || return $?
    run_one_job "$pdir" oracle_leak "$sd" "$ROOT/generated/s5_1/oracle_leak_shard${s}" 450 "" || return $?
  done
  return 0
}

echo "[$(date '+%F %T')] S5_R2_MASTER_STARTED pid=$$ root=$ROOT core=$MEDCOURSE_CORE_DIR"
reuse_direct32b_if_ready || true

while true; do
  if all_jobs_done; then
    echo "[$(date '+%F %T')] S5_R2_GPU_ALL_DONE"
    break
  fi

  if [ "$(served_ok)" = "$SERVED_MODEL" ]; then
    echo "[$(date '+%F %T')] ADOPTING_EXISTING_VLLM port=$PORT model=$SERVED_MODEL"
    run_queue
    QRC=$?
    echo "[$(date '+%F %T')] QUEUE_EXIT rc=$QRC -> releasing vLLM port=$PORT"
    cleanup_vllm
    if [ "$QRC" -eq 99 ]; then
      echo "[$(date '+%F %T')] SMOKE10_GATE_ABORTED_S5_R2"
      exit 1
    fi
    if [ "$QRC" -eq 0 ] && all_jobs_done; then
      echo "[$(date '+%F %T')] S5_R2_GPU_ALL_DONE"
      break
    fi
    sleep 60
    continue
  fi

  G1=$(probe_free_2gpus || true)
  if [ -z "$G1" ]; then
    echo "[$(date '+%F %T')] WAITING_FOR_2_FREE_GPUS (excluding GPU0; cap<=2)"
    sleep 120
    continue
  fi

  sleep 30
  G2=$(probe_free_2gpus || true)
  if [ "$G1" != "$G2" ] || [ -z "$G2" ]; then
    echo "[$(date '+%F %T')] GPU_PROBE_CHANGED ($G1 -> $G2), re-checking"
    sleep 60
    continue
  fi

  echo "[$(date '+%F %T')] ACQUIRED_GPUS=$G2 -> starting vLLM TP=2 on port $PORT"
  export CUDA_VISIBLE_DEVICES="$G2"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  rm -f "$SERVE_LOG"
  CUDA_VISIBLE_DEVICES="$G2" setsid "$VL" serve "$MODEL_DIR" \
    --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_MODEL" \
    --no-enable-log-requests --tensor-parallel-size 2 --max-model-len 8192 \
    --gpu-memory-utilization 0.88 --attention-backend FLASH_ATTN \
    > "$SERVE_LOG" 2>&1 < /dev/null &

  READY=0
  for _ in $(seq 1 36); do
    sleep 10
    if [ "$(served_ok)" = "$SERVED_MODEL" ]; then
      READY=1
      break
    fi
    if ! pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] VLLM_EXITED_EARLY"
      tail -n 30 "$SERVE_LOG" | cut -c1-200
      break
    fi
  done

  if [ "$READY" -ne 1 ]; then
    cleanup_vllm
    sleep 120
    continue
  fi

  echo "[$(date '+%F %T')] VLLM_READY gpus=$G2 port=$PORT"
  run_queue
  QRC=$?
  echo "[$(date '+%F %T')] QUEUE_EXIT rc=$QRC -> releasing vLLM port=$PORT"
  cleanup_vllm
  if [ "$QRC" -eq 99 ]; then
    echo "[$(date '+%F %T')] SMOKE10_GATE_ABORTED_S5_R2"
    exit 1
  fi
  if [ "$QRC" -eq 0 ] && all_jobs_done; then
    echo "[$(date '+%F %T')] S5_R2_GPU_ALL_DONE"
    break
  fi
  sleep 60
done

echo "[$(date '+%F %T')] S5_R2_POST_GPU_EVAL_START"

# 1. Build zero-rare-30gram Synth-Exported artifact across folds 2-7 (NEVER fold 9)
CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/build_synth_exported_release.py" \
  --s5-root "$ROOT" \
  --out-dir "$ROOT/generated/synth_exported_final2k_v17" || exit 1

# 2. Export fold 1 blind pools & fit rehearsal bundle (50/50 patient split)
T2_POOL_ARG=()
T1_POOL_ARG=()
if [ ! -f "$ROOT/blind_pools/t2_fold1_v17/T2_SEALED_SAMPLING_KEY.json" ]; then
  T2_POOL_ARG=(--export-t2-pool "$ROOT/blind_pools/t2_fold1_v17")
fi
if [ ! -f "$ROOT/blind_pools/t1_fold1_v17/T1_SEALED_AUDIT_KEY.json" ]; then
  T1_POOL_ARG=(--export-t1-pool "$ROOT/blind_pools/t1_fold1_v17")
fi

CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/t123_bench.py" \
  --s5-root "$ROOT" \
  --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
  --fit-and-freeze "$ROOT/frozen_rehearsal_v17" \
  --rehearsal-split \
  "${T2_POOL_ARG[@]}" \
  "${T1_POOL_ARG[@]}" || exit 1

if [ -f "$BASE/product/s5_r1/blind_pools/t2_fold1_v15/README_BLIND_ANNOTATION_GUIDE.md" ] && [ ! -f "$ROOT/blind_pools/t2_fold1_v17/README_BLIND_ANNOTATION_GUIDE.md" ]; then
  cp -f "$BASE/product/s5_r1/blind_pools/t2_fold1_v15/README_BLIND_ANNOTATION_GUIDE.md" "$ROOT/blind_pools/t2_fold1_v17/"
fi

# 3. Evaluate frozen rehearsal bundle read-only on held-out 50% fold 1 (B=2000)
CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/t123_bench.py" \
  --s5-root "$ROOT" \
  --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
  --t2-sealed-key "$ROOT/blind_pools/t2_fold1_v17/T2_SEALED_SAMPLING_KEY.json" \
  --t1-sealed-key "$ROOT/blind_pools/t1_fold1_v17/T1_SEALED_AUDIT_KEY.json" \
  --export-consort-ids-private "$ROOT/tables/fold1_t1_consort_ids_PRIVATE.json" \
  --eval-frozen "$ROOT/frozen_rehearsal_v17" \
  --n-boot 2000 \
  --out "$ROOT/tables/rehearsal_t123_fold1_final2k_v17.json" || exit 1

# 4. Freeze 100%-fold1 Stage-1 bundle ready for future Stage-2 fold 9 --eval-frozen
CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/t123_bench.py" \
  --s5-root "$ROOT" \
  --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
  --fit-and-freeze "$ROOT/frozen_stage1_v17" || exit 1

# 5. Unified 17-pathway probe + intrinsic audit
CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/audit_s5_2_17d.py" \
  --s5-root "$ROOT" \
  --out "$ROOT/tables/s5_2_17d_text_probe_final_v17.json" || exit 1

CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/audit_s5_intrinsic.py" \
  --s5-root "$ROOT" \
  --out-json "$ROOT/tables/intrinsic_stage1_final2k_v17.json" \
  --out-csv "$ROOT/tables/intrinsic_stage1_final2k_v17.csv" || exit 1

# 6. Lock MIA Stage-1 calibration scorers + bidirectional 1% FPR thresholds
CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/s6_mia_audit.py" \
  --mode stage1_cal \
  --s5-root "$ROOT" \
  --n-boot 2000 \
  --exported-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
  --freeze-cal-dir "$ROOT/frozen_mia_stage1_v17" \
  --out "$ROOT/tables/mia_stage1_cal_final2k_v17.json" || exit 1

# 7. Rebuild S6-2 60-case blinded pack on fold 8 (NEVER fold 9)
CUDA_VISIBLE_DEVICES="" "$PY" -B "$ROOT/code/build_s6_blind_pack.py" \
  --s5-root "$ROOT" \
  --real-fold "$BASE/data/deid_run9_fold8.jsonl" \
  --out-dir "$ROOT/blind_pools/s6_2_blind_pack_v17" || exit 1

# 8. Export exact fold 1 rehearsal split & run Stage-1 neural T1/T2b rehearsal + lock (non-GPU0 idle cards)
SPLIT_JSON="$ROOT/frozen_rehearsal_v17/rehearsal_split_fold1.json"
CUDA_VISIBLE_DEVICES="" "$PY" -B - "$ROOT/frozen_rehearsal_v17/frozen_bundle.pkl" "$SPLIT_JSON" <<'PY' || exit 1
import json, pickle, sys, t123_bench
bundle_p, out_p = sys.argv[1], sys.argv[2]
tune_ids, eval_ids = t123_bench._rehearsal_patient_split("/home/weify/pyt/data/deid_run9_fold1.jsonl")
with open(bundle_p, "rb") as f:
    b = pickle.load(f)
assert sorted(eval_ids) == b["rehearsal_eval_ids"], "rehearsal_eval_ids mismatch"
with open(out_p, "w", encoding="utf-8") as out:
    json.dump({"tune_ids": sorted(tune_ids), "eval_ids": sorted(eval_ids)}, out, indent=2)
print(f"SPLIT_EXPORT_OK tune={len(tune_ids)} eval={len(eval_ids)}", flush=True)
PY

mapfile -t IDLE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '
    { idx=$1+0; mem=$2+0; util=$3+0; if (idx != 0 && mem < 500 && util == 0) print idx }
  ' | head -n 2
)

if [ "${#IDLE_GPUS[@]}" -ge 1 ]; then
  GPU_A="${IDLE_GPUS[0]}"
  GPU_B="${IDLE_GPUS[1]:-$GPU_A}"
  echo "[$(date '+%F %T')] NEURAL_STAGE1_V17 gpu_a=$GPU_A gpu_b=$GPU_B"

  CUDA_VISIBLE_DEVICES="$GPU_A" "$PY_GPU" -B "$ROOT/code/t12_neural_bench.py" \
    --s5-root "$ROOT" \
    --encoder-path "$ENC" \
    --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
    --fit-and-freeze "$ROOT/frozen_neural_rehearsal_v17" \
    --rehearsal-split \
    --rehearsal-split-json "$SPLIT_JSON" \
    > "$ROOT/logs/neural_rehearsal_v17.log" 2>&1 || exit 1

  CUDA_VISIBLE_DEVICES="$GPU_A" "$PY_GPU" -B "$ROOT/code/t12_neural_bench.py" \
    --s5-root "$ROOT" \
    --encoder-path "$ENC" \
    --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
    --t2-sealed-key "$ROOT/blind_pools/t2_fold1_v17/T2_SEALED_SAMPLING_KEY.json" \
    --eval-frozen "$ROOT/frozen_neural_rehearsal_v17" \
    --n-boot 2000 \
    --out "$ROOT/tables/rehearsal_neural_fold1_final2k_v17.json" \
    >> "$ROOT/logs/neural_rehearsal_v17.log" 2>&1 || exit 1

  CUDA_VISIBLE_DEVICES="$GPU_B" "$PY_GPU" -B "$ROOT/code/t12_neural_bench.py" \
    --s5-root "$ROOT" \
    --encoder-path "$ENC" \
    --exported-synth-jsonl "$ROOT/generated/synth_exported_final2k_v17/admissions.jsonl" \
    --fit-and-freeze "$ROOT/frozen_neural_stage1_v17" \
    > "$ROOT/logs/neural_stage1_lock_v17.log" 2>&1 || exit 1
else
  echo "[$(date '+%F %T')] WARNING: no idle non-GPU0 cards for neural benchmark; CPU stage1 completed"
fi

{
  echo "# Stage-1 v1.9 (medcourse_core v0.3.7) Locked SHA256 ($(date -u '+%F %T UTC'))"
  sha256sum "$MEDCOURSE_CORE_DIR/MANIFEST.sha256" 2>/dev/null || true
  sha256sum "$ROOT/tables/v037_pre_gpu_audit.json" "$ROOT/tables/v037_item_num_parser_audit.json" 2>/dev/null || true
  sha256sum "$ROOT"/code/*.py "$ROOT"/code/*.sh
  sha256sum "$ROOT"/frozen_stage1_v17/frozen_bundle.pkl "$ROOT"/frozen_stage1_v17/freeze_manifest.json
  sha256sum "$ROOT"/frozen_mia_stage1_v17/mia_frozen_cal.pkl "$ROOT"/frozen_mia_stage1_v17/freeze_manifest.json
  sha256sum "$ROOT"/frozen_neural_stage1_v17/bundle_meta.json "$ROOT"/frozen_neural_stage1_v17/freeze_manifest.json 2>/dev/null || true
  sha256sum "$ROOT"/blind_pools/t1_fold1_v17/T1_SEALED_AUDIT_KEY.json "$ROOT"/blind_pools/t2_fold1_v17/T2_SEALED_SAMPLING_KEY.json
  sha256sum "$ROOT"/blind_pools/s6_2_blind_pack_v17/SEALED_KEY_DO_NOT_OPEN.json
  sha256sum "$ROOT"/tables/fold1_t1_consort_ids_PRIVATE.json
  sha256sum "$ROOT"/generated/s5_1/*_shard*/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/s5_2/*/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/s5_ext/direct_32b/admissions.jsonl 2>/dev/null || true
  sha256sum "$ROOT"/generated/synth_exported_final2k_v17/admissions.jsonl "$ROOT"/generated/synth_exported_final2k_v17/export_manifest.json 2>/dev/null || true
} > "$ROOT/tables/STAGE1_V17_LOCKED_SHA256.txt"

echo "[$(date '+%F %T')] S5_R2_MASTER_ALL_COMPLETE"
