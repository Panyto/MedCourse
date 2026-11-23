#!/bin/bash
set -uo pipefail
umask 077

BASE=/home/weify/pyt
ROOT=$BASE/product/s5_r1
PORT=8105
SERVE_LOG="$ROOT/logs/vllm_${PORT}.log"
PY="$BASE/env/bin/python"
VL="$BASE/env/vllm032/bin/vllm"
MODEL_DIR="$BASE/models/Qwen3-32B-AWQ"
SERVED_MODEL="Qwen3-32B-AWQ"
DEADLINE_EPOCH=$(date -u -d '2026-11-22 22:25:00 UTC' +%s)

source "$BASE/.secrets/llm.env"
source "$BASE/gates/S0/v02/code/env.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/.cache"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/code:$BASE/product/v031_s3/medcourse_core:$BASE/gates/S0/bench_core:$BASE/gates/S0:$BASE/gates/G3/g2_core:$BASE/gates/Q45b/g45_core"

mkdir -p "$ROOT/logs" "$ROOT/generated/s5_ext/direct_32b" "$ROOT/generated/s5_1" "$BASE/tmp" "$BASE/.cache"

WATCHDOG_PID=""

cleanup_vllm_8105() {
  for p in $(pgrep -u "$(id -un)" -f "127\.0\.0\.1:${PORT}" || true); do
    kill -TERM "$p" 2>/dev/null || true
  done
  for p in $(pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" || true); do
    pkill -TERM -P "$p" 2>/dev/null || true
    kill -TERM "$p" 2>/dev/null || true
  done
  for _ in $(seq 1 12); do
    pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" >/dev/null 2>&1 || break
    sleep 1
  done
  for p in $(pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" || true); do
    pkill -KILL -P "$p" 2>/dev/null || true
    kill -KILL "$p" 2>/dev/null || true
  done
}

cleanup_all() {
  if [ -n "${WATCHDOG_PID:-}" ]; then
    kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
  fi
  cleanup_vllm_8105
  rm -rf "$ROOT/generated/s5_1"/.tmp8105_* 2>/dev/null || true
}

trap cleanup_all EXIT INT TERM

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
    g for g in (7, 4, 3, 2, 1)
    if g != 0 and g in gpus and gpus[g]["uuid"] not in busy and gpus[g]["mem"] < 500 and gpus[g]["util"] == 0
]
cap = 4 - len(mine)
if cap < 2 or len(avail) < 2:
    sys.exit(1)
print(f"{avail[0]},{avail[1]}")
PY
}

served_ok_8105() {
  "$PY" - <<'PY' 2>/dev/null
import json, os, urllib.request
req = urllib.request.Request(
    "http://127.0.0.1:8105/v1/models",
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

merge_direct_shards() {
  "$PY" - "$ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
merged_dir = os.path.join(root, "generated", "s5_ext", "direct_32b")
os.makedirs(merged_dir, exist_ok=True)
kept = 0
shards = 0
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

NOW_SEC=$(date -u +%s)
if [ "$NOW_SEC" -ge "$DEADLINE_EPOCH" ]; then
  echo "[$(date '+%F %T')] PAST_DEADLINE_BEFORE_START -> refusing to acquire extra GPUs"
  exit 0
fi

G1=$(probe_free_2gpus || true)
sleep 10
G2=$(probe_free_2gpus || true)
if [ -z "$G2" ] || [ "$G1" != "$G2" ]; then
  echo "[$(date '+%F %T')] NO_2_STABLE_FREE_GPUS ($G1 -> $G2)"
  exit 1
fi

PARENT_PID=$$
(
  trap - EXIT INT TERM
  while true; do
    now=$(date -u +%s)
    if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
      echo "[$(date '+%F %T')] DEADLINE_2225_UTC_RELEASING_EXTRA_GPU"
      for p in $(pgrep -u "$(id -un)" -f "127\.0\.0\.1:${PORT}" || true); do
        kill -TERM "$p" 2>/dev/null || true
      done
      for p in $(pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" || true); do
        pkill -TERM -P "$p" 2>/dev/null || true
        kill -TERM "$p" 2>/dev/null || true
      done
      kill -TERM "$PARENT_PID" 2>/dev/null || true
      exit 0
    fi
    sleep 30
  done
) &
WATCHDOG_PID=$!

echo "[$(date '+%F %T')] EXTRA_WORKER_ACQUIRED gpus=$G2 port=$PORT watchdog_deadline=22:25UTC"
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
  if [ "$(served_ok_8105)" = "$SERVED_MODEL" ]; then
    READY=1
    break
  fi
  if ! pgrep -u "$(id -un)" -f "vllm serve .*--port ${PORT}( |$)" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] VLLM_8105_EXITED_EARLY"
    tail -n 30 "$SERVE_LOG" | cut -c1-200
    exit 2
  fi
done

if [ "$READY" -ne 1 ]; then
  echo "[$(date '+%F %T')] VLLM_8105_READY_TIMEOUT"
  exit 3
fi
echo "[$(date '+%F %T')] VLLM_8105_READY gpus=$G2"

DIRECT_SEEDS=(20261230 20261231 20261232 20261233)
for s in 0 1 2 3; do
  rem=$((DEADLINE_EPOCH - $(date -u +%s)))
  if [ "$rem" -lt 600 ]; then
    echo "[$(date '+%F %T')] NEAR_DEADLINE stop direct_32b loop rem=${rem}s"
    break
  fi
  out_dir="$ROOT/generated/s5_ext/direct_32b_shard${s}"
  if job_done "$out_dir" 135; then
    echo "[$(date '+%F %T')] SKIP_DONE $out_dir"
    merge_direct_shards
    continue
  fi
  echo "[$(date '+%F %T')] START direct_32b shard=$s -> $out_dir"
  rm -rf "$out_dir"
  "$PY" -B "$ROOT/code/gen_direct_32b.py" \
    --out "$out_dir" --n 150 --shard-idx "$s" --seed "${DIRECT_SEEDS[$s]}" \
    --writer llm --base-url "http://127.0.0.1:${PORT}/v1" --model "$SERVED_MODEL" --concurrency 20
  rc=$?
  echo "[$(date '+%F %T')] FINISH direct_32b shard=$s rc=$rc"
  merge_direct_shards
  if [ "$rc" -ne 0 ]; then
    break
  fi
done

ASSIST_ARMS=("oracle_leak:6600:" "vis_only:4800:--save-first-try" "full:4800:--save-first-try")
for entry in "${ASSIST_ARMS[@]}"; do
  IFS=":" read -r arm min_sec extra_flag <<< "$entry"
  rem=$((DEADLINE_EPOCH - $(date -u +%s)))
  if [ "$rem" -lt "$min_sec" ]; then
    echo "[$(date '+%F %T')] SKIP_ASSIST_${arm}_shard3 rem=${rem}s < ${min_sec}s"
    continue
  fi
  final_dir="$ROOT/generated/s5_1/${arm}_shard3"
  if [ -e "$final_dir" ]; then
    echo "[$(date '+%F %T')] SKIP_ASSIST_${arm}_shard3 exists_or_active_on_8104"
    continue
  fi
  if pgrep -u "$(id -un)" -f "gen_s5_shard\.py.*--arm ${arm}.*shard3" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] SKIP_ASSIST_${arm}_shard3 active_on_8104"
    continue
  fi
  tmp_dir="$ROOT/generated/s5_1/.tmp8105_${arm}_shard3"
  rm -rf "$tmp_dir"
  echo "[$(date '+%F %T')] START_ASSIST ${arm}_shard3 -> $tmp_dir"
  "$PY" -B "$ROOT/code/gen_s5_shard.py" \
    --plan-dir "$ROOT/planned/s5_1_shards/shard_3" --arm "$arm" \
    --stats "$BASE/product/derived/m2_f27_v021/stats_v021.json" \
    --writer llm --base-url "http://127.0.0.1:${PORT}/v1" --model "$SERVED_MODEL" \
    --seed 20261213 --concurrency 24 $extra_flag --out "$tmp_dir"
  rc=$?
  if [ "$rc" -eq 0 ] && job_done "$tmp_dir" 470; then
    if [ ! -e "$final_dir" ] && ! pgrep -u "$(id -un)" -f "127\.0\.0\.1:8104.*${arm}_shard3" >/dev/null 2>&1; then
      mv "$tmp_dir" "$final_dir"
      echo "[$(date '+%F %T')] COMMIT_ASSIST $final_dir"
    else
      rm -rf "$tmp_dir"
      echo "[$(date '+%F %T')] DISCARD_ASSIST_COLLISION $final_dir"
    fi
  else
    rm -rf "$tmp_dir"
    echo "[$(date '+%F %T')] ASSIST_FAILED_OR_INCOMPLETE ${arm}_shard3 rc=$rc"
  fi
done

echo "[$(date '+%F %T')] EXTRA_WORKER_ALL_DONE -> releasing GPUs $G2"
