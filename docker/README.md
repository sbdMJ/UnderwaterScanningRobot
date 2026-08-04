# Running UnderwaterScanningRobot on this machine (Docker route)

The top-level `README.md` assumes a native Isaac Sim 5.1 install. **That is not
possible on this host**, so everything runs in a container instead.

## Why the container is mandatory here

| | Required by Isaac Sim 5.1 | This host |
|:---|:---|:---|
| glibc | ≥ 2.35 (wheels are `manylinux_2_35`) | **2.31** (Ubuntu 20.04) |
| Python | 3.11 (`cp311` only) | 3.8 system default |

Isaac Sim 5.1 cannot be installed natively on Ubuntu 20.04 at all — not via pip,
not via the standalone bundle. The pre-existing `~/isaac410` is Isaac Sim 4.1.0,
too old for the Isaac Lab 0.54.0 tree vendored in `isaaclab/`.

The `nvcr.io/nvidia/isaac-sim:5.1.0` base image is Ubuntu 24.04, so the whole
stack is glibc-correct inside the container while the host stays untouched.
The NVIDIA driver (580.126.09) is new enough and is passed through by
`--runtime=nvidia`, so no host driver change is needed.

## One-time setup (already done)

```bash
docker pull nvcr.io/nvidia/isaac-sim:5.1.0     # anonymous pull, no NGC login needed
git lfs install && git lfs pull                # meshes/USD are LFS; git-lfs is at ~/.local/bin
# install Isaac Lab + marinelab into the image, then snapshot it
docker run --name uws-setup --user 0:0 -e TERM=xterm ... \
  nvcr.io/nvidia/isaac-sim:5.1.0 docker/setup_in_container.sh
docker commit uws-setup underwater-scan:5.1
```

The resulting `underwater-scan:5.1` image (18GB) already contains Isaac Lab,
rsl_rl and marinelab installed into `/isaac-sim`'s bundled Python 3.11.

## Daily use

```bash
./docker/run.sh                       # interactive shell, cwd = isaaclab/
./docker/run.sh '<command>'           # run one command and exit
./docker/run.sh --gui '<command>'     # same, with the Isaac Sim GUI on your screen
```

### GUI mode

`--gui` forwards X11 into the container, so dropping `--headless` opens the real
Isaac Sim 5.1 window on the host display. Verified working: RTX - Real-Time
renderer, viewport, Stage tree and the IsaacLab panel all render.

```bash
./docker/run.sh --gui './isaaclab.sh -p ../marinelab/scripts/play.py \
  --task Isaac-PKRC-WallScan-Eval-Direct-v0 --num_envs 4 \
  --checkpoint ../checkpoints/rb_train_model_7998.pt'
```

Requirements and notes:

- Needs `DISPLAY` set, i.e. run it from a desktop session (this host is X11 on
  `:1`). It will not work from a bare ssh shell.
- Rather than loosening host X access control with `xhost +local:root`, `run.sh`
  builds a throwaway `/tmp/.uws.xauth` whose entries are family-wildcarded
  (`sed 's/^..../ffff/'`) so the cookie is accepted from inside the container,
  where the hostname differs. Same trick as IsaacLab's `docker/x11.yaml`.
- GPU rendering works because the base image already sets
  `NVIDIA_DRIVER_CAPABILITIES=all`.
- GUI costs VRAM and speed: ~6GB used vs ~1.8GB headless, and it renders every
  frame. **Train with `--headless`**; use the GUI only to eyeball behaviour.
- Alternative if X11 is ever unavailable (e.g. remote/ssh): Isaac Lab's WebRTC
  livestream, `--livestream 2`, then connect with the Isaac Sim streaming client.

### Tests (84) — these also run natively, no Isaac Sim needed

The suite stubs out `isaaclab` in `marinelab/tests/conftest.py`, so it only
needs torch:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/mjkim/.conda/envs/acmpc_sim/bin/python -m pytest marinelab/tests/ -q
```

### Evaluate the trained policy

```bash
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/play.py \
  --task Isaac-PKRC-WallScan-Eval-Direct-v0 --num_envs 4 --headless \
  --checkpoint ../checkpoints/rb_train_model_7998.pt'
```

`play.py` loops until killed (`while simulation_app.is_running()`) unless you
pass `--video --video_length N`. It writes `checkpoints/exported/policy.{pt,onnx}`
right after loading the checkpoint — a quick way to confirm the load worked.

### Reproducing the README's results table

`--log_traj` runs a bounded rollout, logs the scan trajectory, computes the
scan-quality metrics and writes `results/{trajectory,metrics}_<tag>.{npz,json,png}`:

```bash
# nominal   (Stage3 = full cycle, no dynamics DR, no sensor bias, DORAEMON off)
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/play.py \
  --task Isaac-PKRC-WallScan-Stage3-Direct-v0 --num_envs 8 --headless \
  --checkpoint ../checkpoints/rb_train_model_7998.pt \
  --log_traj --eval_steps 18500 --score_episode 1 --tag nominal'

# stress DR (Eval = +-45 deg initial attitude, hydro +-50%, thrust +-30%)
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/play.py \
  --task Isaac-PKRC-WallScan-Eval-Direct-v0 --num_envs 8 --headless \
  --checkpoint ../checkpoints/rb_train_model_7998.pt \
  --log_traj --eval_steps 18500 --score_episode 1 --tag stress_dr'
```

**`--eval_steps 18500 --score_episode 1` is not optional.** `wallscan_env._reset_idx`
randomises `episode_length_buf` over `[0, max_episode_length)` on the initial full
reset — the standard Isaac Lab trick for decorrelating episode ends across envs — so
every env's **first** episode is a partial one, averaging `180 / 2 = 90 s`. Rate-like
metrics (tilt, speeds, crab, s_hat error) are unaffected, but `scan cycles completed`
is cumulative and comes out roughly halved (measured: 0.62 on episode 0 vs 2.00 on a
full episode 1). Logging 18500 steps covers one partial plus one full episode; the
metrics print a loud warning if the scored episode was truncated anyway.

The metric definitions live in `marinelab/marinelab/tasks/pkrc_wallscan/eval_metrics.py`
(pure torch/numpy, unit-tested in `marinelab/tests/test_eval_metrics.py`) and mirror the
env's own semantics — `tilt` is the same `arccos(up_z)` the reward penalises, and `crab`
is `|yaw - theta|`, which is exactly the heading error the reward tracks because
`_yaw_ref_cur = theta_gt` after the spin search.

### Train

```bash
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/train.py \
  --task Isaac-PKRC-WallScan-Stage3-Direct-v0 \
  --num_envs 2048 --headless --max_iterations 5000 --run_name stage3'
```

Note on `--num_envs`: the top-level README's recipe uses 4096, which was trained
on a larger GPU. This host has a 16GB RTX 4080 — start lower (2048) and raise it
only if memory allows, otherwise the reproduction recipe is unchanged.

Resume for the second leg exactly as the top-level README says — CLI
`--resume --load_run <folder>`, since `agent.resume=True` via hydra is overridden.

## Host-environment fixes that were needed

Three things fail on the stock image; all are handled in
`setup_in_container.sh`, but they are worth knowing if you rebuild:

1. **`TERM=dumb`** — the base image sets it, and `isaaclab.sh:16` runs `tabs 4`,
   which aborts with `'ansi+tabs': unknown terminal type`. Fixed by `-e TERM=xterm`.
2. **`flatdict==4.0.1` build failure** — the top-level README blames
   setuptools 81+, but pinning the *runtime* setuptools is not enough: pip builds
   sdists in an **isolated** env where it installs the newest setuptools, and
   flatdict's `setup.py` does `import pkg_resources` (removed in 81+). Fixed with
   `PIP_CONSTRAINT=setuptools<81` plus a `--no-build-isolation` pre-install.
   Without this, `isaaclab` core silently does not install.
3. **No `git` binary** — `rsl_rl` imports GitPython at module load and dies with
   `Bad git executable`. The image now has git installed.

### Root vs. host ownership

The base image's default user is uid 1234 (`isaac-sim`), which cannot write the
bind-mounted repo (owned by uid 1000), and uid 1000 cannot write `/isaac-sim`.
So containers run as root (`--user 0:0`), which is also what Isaac Lab's own
docker tooling does. Files root creates in the repo can be handed back with:

```bash
docker run --rm --user 0:0 -v /home/mjkim/UnderwaterScanningRobot:/repo \
  --entrypoint bash nvcr.io/nvidia/isaac-sim:5.1.0 -c 'chown -R 1000:1000 /repo'
```

`isaaclab/_isaac_sim` is a symlink to `/isaac-sim`, so it is **dangling on the
host** and only resolves inside the container. That is expected.
