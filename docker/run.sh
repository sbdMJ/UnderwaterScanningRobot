#!/usr/bin/env bash
# Launch the UnderwaterScanningRobot container.
#
#   ./docker/run.sh                       -> interactive bash shell (headless)
#   ./docker/run.sh '<cmd>'               -> run one command and exit
#   ./docker/run.sh --gui '<cmd>'         -> same, with X11 forwarded so Isaac Sim
#                                            can open its GUI window on the host
#
# Isaac Sim 5.1 needs glibc >= 2.35 (its `kit` binary needs GLIBC_2.34 symbols);
# this host is Ubuntu 20.04 (glibc 2.31), so the simulator only runs in here.
set -euo pipefail

IMAGE="${UWS_IMAGE:-underwater-scan:5.1}"
REPO_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_CTR=/workspace/UnderwaterScanningRobot
CACHE="$HOME/docker/isaac-sim"

GUI=0
if [ "${1:-}" = "--gui" ]; then GUI=1; shift; fi

# GPU access: native Linux with nvidia-container-toolkit registers an "nvidia"
# docker runtime; Docker Desktop (Windows/WSL2) has no such runtime and passes
# the GPU through --gpus alone. Detect the runtime instead of hardcoding it —
# an unconditional --runtime=nvidia hard-fails on Docker Desktop.
GPU_ARGS=(--gpus all)
if docker info --format '{{range $k, $v := .Runtimes}}{{$k}}{{"\n"}}{{end}}' 2>/dev/null | grep -qx nvidia; then
  GPU_ARGS=(--runtime=nvidia --gpus all)
fi

mkdir -p "$CACHE"/cache/{kit,ov,pip,glcache,computecache} "$CACHE"/{logs,data,documents}

# Only request a TTY when we actually have one, so this works from scripts too
# ("the input device is not a TTY" otherwise).
TTY_FLAGS=(-i)
[ -t 0 ] && TTY_FLAGS=(-i -t)

# acados (NMPC / Diff-WMPC): built on the host into $ACADOS_HOST and bind-mounted, so the
# 18GB image never has to be re-committed for it. Silently skipped when the dir is absent,
# which keeps the plain RL workflow working on a machine that has no acados.
ACADOS_HOST="${UWS_ACADOS:-$HOME/docker/acados}"
ACADOS_ARGS=()
if [ -d "$ACADOS_HOST" ]; then
  ACADOS_ARGS=(
    -v "$ACADOS_HOST":/opt/acados:rw
    -e ACADOS_SOURCE_DIR=/opt/acados
    -e "LD_LIBRARY_PATH=/opt/acados/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # acados_template + casadi live in the mount too (a --rm container's own
    # site-packages is thrown away, and re-committing the 18GB image for two
    # pure-python wheels is not worth it). numpy is deliberately NOT installed
    # there: it would shadow Isaac's 1.26 and break torch.
    -e "PYTHONPATH=/opt/acados/pysite${PYTHONPATH:+:$PYTHONPATH}"
  )
fi

GUI_ARGS=()
if [ "$GUI" = "1" ]; then
  : "${DISPLAY:?--gui needs DISPLAY set (run it from a desktop session, not a bare ssh shell)}"

  # Build a throwaway Xauthority whose entries are family-wildcard ("ffff"), so the
  # cookie is accepted from inside the container where the hostname differs. This is
  # what IsaacLab's own docker/x11.yaml does, and it avoids having to loosen host X
  # access control with `xhost +local:root`.
  XAUTH=/tmp/.uws.xauth
  rm -f "$XAUTH"; touch "$XAUTH"
  xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null
  chmod 644 "$XAUTH"

  GUI_ARGS=(
    -e "DISPLAY=$DISPLAY"
    -e "XAUTHORITY=$XAUTH"
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v "$XAUTH":"$XAUTH":rw
  )
fi

docker run --rm "${TTY_FLAGS[@]}" \
  --user 0:0 \
  "${GPU_ARGS[@]}" \
  --network=host \
  --shm-size=8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e "TERM=${UWS_TERM:-xterm}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  ${ACADOS_ARGS[@]+"${ACADOS_ARGS[@]}"} \
  ${GUI_ARGS[@]+"${GUI_ARGS[@]}"} \
  -v "$REPO_HOST":"$REPO_CTR":rw \
  -v "$CACHE/cache/kit":/isaac-sim/kit/cache:rw \
  -v "$CACHE/cache/ov":/root/.cache/ov:rw \
  -v "$CACHE/cache/pip":/root/.cache/pip:rw \
  -v "$CACHE/cache/glcache":/root/.cache/nvidia/GLCache:rw \
  -v "$CACHE/cache/computecache":/root/.nv/ComputeCache:rw \
  -v "$CACHE/logs":/root/.nvidia-omniverse/logs:rw \
  -v "$CACHE/data":/root/.local/share/ov/data:rw \
  -v "$CACHE/documents":/root/Documents:rw \
  -w "$REPO_CTR/isaaclab" \
  --entrypoint bash \
  "$IMAGE" ${1:+-lc "$*"}
