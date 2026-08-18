#!/usr/bin/env bash
# e5_hwdrag_lat parallel runner: 4 workers = (method x condition). Each CELL is its
# own Isaac process — a second gym.make in the same process dies on
# "A prim already exists at /World/envs/env_0/Tank" (measured 2026-08-11), which is
# why every experiment here has always run per-cell processes. Each worker keeps a
# private isaaclab/logs mount (docker/run.sh UWS_LOGS_MOUNT) so the acados codegen
# dir is not raced. Cells whose metrics file already exists are skipped, so the
# runner is resumable. Host-side logs: isaaclab/logs/e5_hwdrag_lat_run/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CFG=../marinelab/scripts/experiments/configs/e5_hwdrag_lat.yaml
LOGDIR="$REPO/isaaclab/logs/e5_hwdrag_lat_run"
RESULTS="$REPO/experimental_results/e5_hwdrag_lat/metrics"
SEEDS=(0 1 2 3 4)
mkdir -p "$LOGDIR"

worker() {  # $1 = worker index, $2 = method, $3 = condition
  local i=$1 method=$2 cond=$3 rc=0
  for s in "${SEEDS[@]}"; do
    if [ -f "$RESULTS/metrics_${method}_${cond}_s${s}.json" ]; then
      echo "[SKIP] ${method}_${cond}_s${s} (metrics exist)"
      continue
    fi
    if UWS_LOGS_MOUNT="$HOME/docker/isaac-sim/e5hl_w$i" "$REPO/docker/run.sh" \
        "./isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py $method --config $CFG --cond $cond --seed $s" \
        > "$LOGDIR/${method}_${cond}_s${s}.log" 2>&1; then
      echo "[DONE] ${method}_${cond}_s${s}"
    else
      echo "[FAIL] ${method}_${cond}_s${s} — see $LOGDIR/${method}_${cond}_s${s}.log"
      rc=1
    fi
  done
  return $rc
}

i=0
pids=()
names=()
for method in nominal ssi; do
  for cond in rti4_h20 rti4_h30; do
    worker "$i" "$method" "$cond" &
    pids+=($!)
    names+=("${method}_${cond}")
    i=$((i + 1))
    sleep 20   # stagger Isaac boots (shared kit cache)
  done
done

rc=0
for j in "${!pids[@]}"; do
  wait "${pids[$j]}" || { echo "[WORKER FAIL] ${names[$j]}"; rc=1; }
done
exit $rc
