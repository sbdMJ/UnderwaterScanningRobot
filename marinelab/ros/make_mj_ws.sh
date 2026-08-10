#!/usr/bin/env bash
# Assemble mj_ws — the Jetson overlay workspace for the wallscan sim-to-real work.
#
# Contents:
#   src/pkrc_control          copied from hero_ws, with patch 0001 (wallscan auto
#                             mode) applied — teleop stays the sole CAN owner
#   src/pkrc_wallscan_bridge  copied from this repo (estimator/controller/mapper
#                             nodes; running them needs a marinelab checkout +
#                             MARINELAB_ROOT — Phase D, not needed for calibration)
#   README.md                 build/run instructions for the Jetson
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

cp "$REPO/marinelab/ros/mj_ws_README.md" "$OUT/README.md"

echo "mj_ws assembled at $OUT (patch 0001 applied, teleop py_compile OK)"
echo "next: rsync -a $OUT/ <jetson>:~/mj_ws/  — then follow $OUT/README.md"
