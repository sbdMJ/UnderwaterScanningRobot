# Installation

marinelab can be deployed two ways: locally on top of an existing isaaclab
editable install, or via Docker with isaaclab as the base image.

---

## Prerequisites

| Requirement | Why | Provided by |
|:---|:---|:---|
| Git LFS | marinelab stores UUV meshes (`.usd`, `.dae`, `.obj`) in Git LFS | `apt-get install git-lfs` before cloning |
| `isaaclab` + `isaaclab_assets` | **Not** pip dependencies of marinelab — do not list in `pyproject.toml` or install separately | Local: the isaaclab editable install already on the host. Docker: baked into the base image |

### Do not create a conda or venv environment

Isaac Sim ships its own Python interpreter. marinelab installs into that
interpreter, so **no environment should be created for this repo at all** — not
conda, not venv, not uv.

This matters because `isaaclab.sh` picks its interpreter from the environment
(`extract_python_exe`): it uses `$CONDA_PREFIX/bin/python` when `CONDA_PREFIX` is
set, `$VIRTUAL_ENV/bin/python` when `VIRTUAL_ENV` is set, and only otherwise the
Isaac Sim python. So an active environment silently redirects even the
`./isaaclab.sh -p -m pip install` command below, and marinelab lands in a
site-packages that Isaac Sim never reads. The symptom is a successful-looking
install followed by `ModuleNotFoundError: marinelab` (or `pxr`) at task launch.

Isaac Lab upstream does document a conda path (`isaaclab.sh --conda`). marinelab
does not use it — deactivate any environment before installing, and run
everything through `./isaaclab.sh -p` or, inside Docker, `/isaac-sim/python.sh`.

---

## Path A — Local install

Use this path when you already have a working isaaclab editable install on the
host machine.

### 1. Clone the repo and pull meshes

```bash
git clone <marinelab-url> /workspace/marinelab
cd /workspace/marinelab
git lfs install
git lfs pull
```

### 2. Install marinelab into the isaaclab environment

The isaaclab environment exposes its own Python interpreter via `isaaclab.sh -p`.
Use that launcher so the install lands in the same Python that Isaac Sim uses:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p -m pip install -e /workspace/marinelab
```

### 3. Verify

Verification must happen inside a launched Isaac Sim app because marinelab
transitively imports `pxr` (the USD runtime), which is only available once the
app is running. A bare `python -c "import marinelab"` will fail with
`No module named pxr` — that is expected outside the app.

Run the random agent instead:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

A clean exit (no import errors, no shape mismatches) confirms the install is
correct.

---

## Path B — Docker

Use this path for reproducible junior deployments or CI.

### Overview

```
isaaclab Docker base image   (built from the isaaclab clean fork)
        |  FROM
        v
marinelab Docker image       (marinelab/docker/Dockerfile)
        pip install -e marinelab
        git lfs pull
```

### 1. Build the isaaclab base image

Follow the build instructions in the isaaclab repository's `docker/` directory.
This produces a local image tagged `isaaclab:latest` (or a custom tag you
choose).

### 2. Build the marinelab image

```bash
cd /workspace/marinelab
docker compose -f docker/docker-compose.yaml build
```

To use a custom isaaclab image tag:

```bash
ISAACLAB_IMAGE=my-isaaclab:5.1.0 \
    docker compose -f docker/docker-compose.yaml build
```

### 3. Start the container

```bash
# Allow the container to open X11 windows on the host (if using GUI)
xhost +local:docker

docker compose -f docker/docker-compose.yaml run --rm marinelab bash
```

### 4. Verify inside the container

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

### Notes

- The `docker-compose.yaml` bind-mounts the marinelab source directory into the
  container so local edits are reflected immediately without a rebuild.
- GPU access uses the NVIDIA container runtime (`deploy.resources.reservations`).
  Ensure `nvidia-container-toolkit` is installed on the host.
- The exact Python launcher inside the isaaclab image is
  `${ISAACLAB_PATH}/isaaclab.sh -p -m pip` (or `/isaac-sim/python.sh -m pip`
  directly). The Dockerfile uses the direct path; if your isaaclab image uses a
  different layout, adjust accordingly.

---

## Path C — ARM64 (NVIDIA GB10 / DGX Spark, Grace-Blackwell)

Use this path on ARM64 (`aarch64`) machines such as the **NVIDIA GB10 / DGX
Spark**. The prebuilt Isaac Sim that Path A and Path B assume is **x86-only**, so
ARM64 needs Isaac Sim built from source. Everything downstream (Isaac Lab,
marinelab, the RL stack) then installs normally.

> **Why source build?** As of Isaac Sim 5.1 there is no pip wheel or container
> image for `linux/arm64`: `pip install isaacsim` only resolves a 6.0.x release
> whose `omniverse-kit` dependency fails to build on aarch64, and the 5.1 x86
> binaries are not runnable on ARM (`Exec format error`). NVIDIA's supported
> route on GB10 is the IsaacSim source build. Headless training, GUI viewport,
> and GPU rendering all work once it is built. `--livestream` does **not** work
> (the WebRTC livestream extension is not distributed for aarch64).

### 1. Build dependencies

The IsaacSim build requires **GCC 11** (newer Ubuntu defaults to GCC 13) plus the
usual X11/GL development headers:

```bash
sudo apt-get update
sudo apt-get install -y gcc-11 g++-11 \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
```

Verify `gcc --version` reports 11.x before building.

### 2. Clone and build Isaac Sim from source

```bash
git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
git lfs pull

# The build's EULA prompt is interactive and cannot be answered in a
# non-interactive shell. Pre-accept it by creating the marker file:
touch .eula_accepted

./build.sh
```

A successful build produces `_build/linux-aarch64/release` (~11 GB). The build
takes roughly 10–15 minutes.

### 3. Point Isaac Lab at the built Isaac Sim

Symlink the freshly built release into the Isaac Lab clone, then run its
installer:

```bash
ln -sfn "$(pwd)/_build/linux-aarch64/release" <isaaclab-path>/_isaac_sim

cd <isaaclab-path>
export TERM=xterm           # without this, isaaclab.sh aborts: 'ansi+tabs: unknown terminal type'
./isaaclab.sh --install
```

`--install` pulls Isaac Lab's dependencies but may **not** install the base
`isaaclab` package itself. If a later step fails with
`ModuleNotFoundError: No module named 'isaaclab'`, install it explicitly through
the bundled Python:

```bash
cd <isaaclab-path>
./isaaclab.sh -p -m pip install "setuptools<81" wheel   # newer setuptools drops pkg_resources, breaking the editable build
./isaaclab.sh -p -m pip install -e source/isaaclab
```

### 4. Install marinelab and the RL stack

Identical to Path A from here — the launcher is architecture-independent:

```bash
cd <isaaclab-path>
./isaaclab.sh -p -m pip install -e <marinelab-path>
```

### 5. Verify (headless)

```bash
cd <isaaclab-path>
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"   # avoids an OpenMP segfault at startup on aarch64
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

A clean exit confirms the ARM64 stack is working.

### Notes

- **GPU rendering with a GUI works** even when the X display is software-rendered
  (`llvmpipe`). Isaac Sim renders through the NVIDIA EGL vendor
  (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`), which is independent of the
  X11/GLX display path. To watch training in a viewport, drop `--headless` and
  point `DISPLAY` at a running VNC display.
- **`LD_PRELOAD` for `libgomp` is required**, not optional — without it the app
  segfaults during startup on aarch64.
- The bundled Python interpreter inside the build is **Python 3.11** even when the
  host OS ships Python 3.12; always go through `./isaaclab.sh -p` rather than the
  system Python.
- Replace `<isaaclab-path>` and `<marinelab-path>` with wherever you cloned each
  repo. Nothing in this path depends on a specific absolute location.
