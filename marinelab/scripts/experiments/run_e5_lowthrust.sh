#!/usr/bin/env bash
# e5_lowthrust parallel runner: 4 workers = (method x condition), each worker runs
# its 5 seeds inside ONE Isaac process (one boot per worker). Each worker gets a
# private isaaclab/logs mount (docker/run.sh UWS_LOGS_MOUNT) so the acados codegen
# dir is not raced. Host-side worker logs land in isaaclab/logs/e5_lowthrust_run/.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CFG=../marinelab/scripts/experiments/configs/e5_lowthrust.yaml
LOGDIR="$REPO/isaaclab/logs/e5_lowthrust_run"
mkdir -p "$LOGDIR"

i=0
pids=()
names=()
for method in nominal ssi; do
  for cond in lowthrust lowthrust_trim; do
    UWS_LOGS_MOUNT="$HOME/docker/isaac-sim/e5lt_w$i" \
      "$REPO/docker/run.sh" \
      "./isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py $method --config $CFG --cond $cond" \
      > "$LOGDIR/${method}_${cond}.log" 2>&1 &
    pids+=($!)
    names+=("${method}_${cond}")
    i=$((i + 1))
    sleep 20   # stagger Isaac boots (shared kit cache)
  done
done

rc=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[DONE] ${names[$j]}"
  else
    echo "[FAIL] ${names[$j]} — see $LOGDIR/${names[$j]}.log"
    rc=1
  fi
done
exit $rc
