#!/usr/bin/env bash
# Assemble mj_ws — the Jetson overlay workspace for the wallscan sim-to-real work.
#
# Contents:
#   src/pkrc_control          copied from hero_ws, with patch 0001 (wallscan auto
#                             mode) applied — teleop stays the sole CAN owner
#   src/pkrc_wallscan_bridge  copied from this repo (estimator/controller/mapper nodes)
#   marinelab/                self-contained pure-python marinelab bundle:
#                             marinelab/ (the package), config/*.json (incl. the
#                             field-calibrated pkrc_plant_hw2026.json),
#                             scripts/experiments/{bench_inference,hw_bag_replay_estimator}.py
#                             -> MARINELAB_ROOT=<ws>/marinelab, no repo clone needed
#   experimental_results/tuning/bo_nmpc/best_params.json   (BO weights ssi/bo load)
#   jetson_acados_build.md    acados aarch64 build procedure (bench prerequisite)
#   README.md                 build/run instructions incl. the small-pool scenarios
#
# Usage: ./make_mj_ws.sh <hero_ws-path> [out-path (default ~/mj_ws)]
set -euo pipefail

HERO_WS=${1:?usage: make_mj_ws.sh <hero_ws-path> [out-path]}
OUT=${2:-"$HOME/mj_ws"}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH="$REPO/marinelab/ros/hero_ws_patches/0001-teleop-wallscan-auto-mode.patch"

[ -d "$HERO_WS/src/pkrc_control" ] || { echo "no pkrc_control under $HERO_WS/src" >&2; exit 1; }
[ -e "$OUT" ] && { echo "$OUT already exists — remove it first (refusing to overwrite)" >&2; exit 1; }

mkdir -p "$OUT/src"
cp -a "$HERO_WS/src/pkrc_control" "$OUT/src/"
cp -a "$REPO/marinelab/ros/pkrc_wallscan_bridge" "$OUT/src/"

# strip caches, editor backups and stale build logs from the copies
find "$OUT/src" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$OUT/src" -name '*.bak-*' -delete
rm -rf "$OUT/src/pkrc_control/log"

# apply the wallscan auto-mode patch (paths in the patch are src/pkrc_control/...)
patch -p1 -d "$OUT" --dry-run < "$PATCH" > /dev/null
patch -p1 -d "$OUT" < "$PATCH"

# syntax check the patched teleop with whatever python3 is around
python3 -m py_compile "$OUT/src/pkrc_control/pkrc_control/keyboard_control_teleop.py"
find "$OUT/src" -name __pycache__ -type d -prune -exec rm -rf {} +

# ---- self-contained marinelab bundle (pure python — no isaaclab on the Jetson) ----
# Layout mirrors the repo so the scripts' relative-path logic works unchanged:
#   bench_inference.py  resolves REPO = <ws>, plant = <ws>/marinelab/config/...,
#                       BO weights = <ws>/experimental_results/tuning/bo_nmpc/...,
#                       output -> <ws>/experimental_results/e4_inference/
#   MARINELAB_ROOT for the ROS nodes = <ws>/marinelab
mkdir -p "$OUT/marinelab/scripts/experiments" "$OUT/experimental_results/tuning/bo_nmpc"
cp -a "$REPO/marinelab/marinelab" "$OUT/marinelab/marinelab"
# python only: the USD/mesh data under assets/ is ~145 MB and sim-only
find "$OUT/marinelab/marinelab" -type f ! -name '*.py' ! -name '*.json' -delete
find "$OUT/marinelab/marinelab" -type d -empty -delete
cp -a "$REPO/marinelab/config" "$OUT/marinelab/config"
cp "$REPO/marinelab/scripts/experiments/bench_inference.py" \
   "$REPO/marinelab/scripts/experiments/hw_bag_replay_estimator.py" \
   "$OUT/marinelab/scripts/experiments/"
cp "$REPO/experimental_results/tuning/bo_nmpc/best_params.json" \
   "$OUT/experimental_results/tuning/bo_nmpc/"
cp "$REPO/docs/experiments/sim-to-real/jetson_acados_build.md" "$OUT/"
find "$OUT/marinelab" -name __pycache__ -type d -prune -exec rm -rf {} +

# import smoke: the bundle's pure layers must import through the loader shim
MJWS_OUT="$OUT" python3 - <<'PY'
import os, sys
out = os.environ["MJWS_OUT"]
sys.path.insert(0, os.path.join(out, "src", "pkrc_wallscan_bridge", "pkrc_wallscan_bridge"))
from marinelab_loader import load_marinelab
load_marinelab(os.path.join(out, "marinelab"))
import marinelab.control.types  # noqa
import marinelab.control.hw_bridge  # noqa
try:
    from marinelab.tasks.pkrc_wallscan import scan_state_machine  # noqa — needs torch
    extra = " (+torch layers)"
except ModuleNotFoundError as e:
    extra = f" (torch layers skipped here: {e.name}; the Jetson needs torch — README §0)"
print("bundle import smoke OK" + extra)
PY

cp "$REPO/marinelab/ros/mj_ws_README.md" "$OUT/README.md"

echo "mj_ws assembled at $OUT (patch 0001 applied, teleop py_compile OK)"
echo "next: rsync -a $OUT/ <jetson>:~/mj_ws/  — then follow $OUT/README.md"
