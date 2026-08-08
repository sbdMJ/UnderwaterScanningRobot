#!/usr/bin/env bash
# Runs INSIDE the isaac-sim:5.1.0 container AS ROOT. Installs Isaac Lab +
# marinelab into the container's bundled Python at /isaac-sim. Run once, then
# `docker commit` so the installs persist into the runtime image.
#
# Root is required because pip must write to /isaac-sim (owned by uid 1234 in
# the base image) *and* to the bind-mounted repo (owned by the host user).
# HOST_UID/HOST_GID are chowned back at the end so the host git tree stays usable.
set -euo pipefail

REPO=/workspace/UnderwaterScanningRobot
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"

echo "==> linking _isaac_sim -> /isaac-sim"
ln -sfn /isaac-sim "$REPO/isaaclab/_isaac_sim"

# rsl_rl imports GitPython at module load, which hard-fails without a `git`
# binary ("Bad git executable"). The isaac-sim base image ships no git.
echo "==> installing git (required by rsl_rl -> GitPython)"
apt-get update -qq && apt-get install -y -qq --no-install-recommends git

PY=/isaac-sim/python.sh

echo "==> pinning setuptools<81 (README: flatdict build fails on setuptools 81+)"
$PY -m pip install -q "setuptools<81"

# The README's flatdict warning has a subtlety: pinning the *runtime* setuptools
# is not enough. pip builds sdists in an ISOLATED env where it installs the
# newest setuptools, and flatdict-4.0.1/setup.py does `import pkg_resources`,
# which setuptools 81+ removed -> "Getting requirements to build wheel" fails.
# PIP_CONSTRAINT applies to those build envs too; the explicit no-isolation
# install of flatdict makes the failure impossible to hit at all.
echo "==> constraining build-isolation envs to setuptools<81"
echo 'setuptools<81' > /tmp/build-constraints.txt
export PIP_CONSTRAINT=/tmp/build-constraints.txt

echo "==> pre-installing flatdict==4.0.1 without build isolation"
$PY -m pip install --no-build-isolation "flatdict==4.0.1"

echo "==> installing Isaac Lab core + rsl_rl"
cd "$REPO/isaaclab"
./isaaclab.sh --install rsl_rl

echo "==> installing marinelab (editable)"
$PY -m pip install -e "$REPO/marinelab"

# Only modules that are importable BEFORE the Kit app starts are checked here.
# isaaclab_tasks / isaaclab.envs pull in `pxr` (USD python bindings), which Isaac
# Sim only puts on sys.path once AppLauncher has booted the app -- so importing
# them standalone fails by design. End-to-end verification is `play.py` instead.
# marinelab is special-cased: its __init__ eagerly imports tasks -> isaaclab.utils
# -> pxr, so pre-app it can only fail on pxr; any OTHER import error is a real
# install problem and still fails the build.
echo "==> verifying imports (pre-app-launch subset)"
$PY -c '
import importlib
for m in ("isaaclab", "isaaclab_rl", "rsl_rl", "flatdict", "marinelab.core.parameters"):
    try:
        importlib.import_module(m)
        print("ok:", m)
    except ModuleNotFoundError as e:
        if e.name != "pxr":
            raise
        print("ok (deferred - needs pxr, importable only after app launch):", m)
'

echo "==> restoring host ownership on the bind-mounted repo"
chown -R "$HOST_UID:$HOST_GID" "$REPO"

echo "==> setup complete"
