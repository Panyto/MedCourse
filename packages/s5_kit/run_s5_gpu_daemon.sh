#!/bin/bash
set -uo pipefail
umask 077

BASE=/home/weify/pyt
ROOT=$BASE/product/s5_r1
PORT=8104
SERVE_LOG="$ROOT/logs/vllm_${PORT}.log"
PY="$BASE/env/bin/python"
VL="$BASE/env/vllm032/bin/vllm"
MODEL_DIR="$BASE/models/Qwen3-32B-AWQ"
SERVED_MODEL="Qwen3-32B-AWQ"

source "$BASE/.secrets/llm.env"
source "$BASE/gates/S0/v02/code/env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/.cache"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/code:$BASE/product/v031_s3/medcourse_core:$BASE/gates/S0/bench_core:$BASE/gates/S0:$BASE/gates/G3/g2_core:$BASE/gates/Q45b/g45_core"

mkdir -p "$ROOT/logs" "$ROOT/generated/s5_2" "$ROOT/generated/s5_1" "$ROOT/tables" "$BASE/tmp" "$BASE/.cache"

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
    g for g in (4, 5, 6, 7, 3, 2, 1)
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
    "http://127.0.0.1:8104/v1/models",
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
  "$PY" - "$out_dir/counts.json" "$min_kept" <<'PY' 2>/dev/null
import json, os, sys
p, mk = sys.argv[1], int(sys.argv[2])
if not os.path.exists(p):
    sys.exit(1)
c = json.load(open(p, encoding="utf-8"))
if int(c.get("llm_errors", 0)) == 0 and int(c.get("kept", 0)) >= mk:
    sys.exit(0)
sys.exit(1)
PY
}

all_jobs_done() {
  job_done "$ROOT/generated/s5_2/route_b" 320 || return 1
  job_done "$ROOT/generated/s5_2/route_a_only" 320 || return 1
  job_done "$ROOT/generated/s5_2/route_ab" 320 || return 1
  for s in 0 1 2 3; do
    for arm in full vis_only oracle_leak; do
      job_done "$ROOT/generated/s5_1/${arm}_shard${s}" 470 || return 1
    done
  done
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

run_queue() {
  run_one_job "$ROOT/planned/s5_2_17d/route_b" route_b 20261220 "$ROOT/generated/s5_2/route_b" 320 "--save-first-try" || return $?
  run_one_job "$ROOT/planned/s5_2_17d/route_a_only" route_a_only 20261220 "$ROOT/generated/s5_2/route_a_only" 320 "--save-first-try" || return $?
  run_one_job "$ROOT/planned/s5_2_17d/route_ab" route_ab 20261220 "$ROOT/generated/s5_2/route_ab" 320 "--save-first-try" || return $?

  local seeds=(20261210 20261211 20261212 20261213)
  for s in 0 1 2 3; do
    local sd="${seeds[$s]}"
    local pdir="$ROOT/planned/s5_1_shards/shard_${s}"
    run_one_job "$pdir" full "$sd" "$ROOT/generated/s5_1/full_shard${s}" 470 "--save-first-try" || return $?
    run_one_job "$pdir" vis_only "$sd" "$ROOT/generated/s5_1/vis_only_shard${s}" 470 "--save-first-try" || return $?
    run_one_job "$pdir" oracle_leak "$sd" "$ROOT/generated/s5_1/oracle_leak_shard${s}" 470 "" || return $?
  done
  return 0
}

echo "[$(date '+%F %T')] S5_DAEMON_STARTED pid=$$ root=$ROOT"

while true; do
  if all_jobs_done; then
    echo "[$(date '+%F %T')] S5_DAEMON_ALL_DONE"
    break
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

  echo "[$(date '+%F %T')] VLLM_READY gpus=$G2"
  run_queue
  QRC=$?
  echo "[$(date '+%F %T')] QUEUE_EXIT rc=$QRC -> releasing vLLM"
  cleanup_vllm
  if [ "$QRC" -eq 0 ] && all_jobs_done; then
    echo "[$(date '+%F %T')] S5_DAEMON_ALL_DONE"
    break
  fi
  sleep 60
done
