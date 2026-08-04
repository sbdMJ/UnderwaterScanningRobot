# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

RL for a PKRC UUV (22.8 kg, 6× T200) that scans the wall of a cylindrical water tank
(R=6 m, H=10 m) with a single-beam Ping1D sonar. Three things live here:

- `marinelab/` — the Isaac Lab extension being developed (physics core + wallscan task). **This is the code you edit.**
- `isaaclab/` — a zero-touch vendored copy of upstream Isaac Lab (47aa161). Do not modify; it exists for version reproducibility and provides `isaaclab.sh`.
- `checkpoints/`, `results/` — trained policies (Git LFS-adjacent binaries) and evaluation artifacts.

Layering is strictly downward: `marinelab` imports `isaaclab`, never the reverse.
marinelab hooks in via the `isaaclab.extensions` entry point in `marinelab/pyproject.toml`,
so no file inside `isaaclab/` is ever patched.

## Running anything: this host requires Docker

Isaac Sim 5.1 needs glibc ≥ 2.35 / Python 3.11; this host is Ubuntu 20.04 (glibc 2.31).
The top-level `README.md` describes the native install path — **it does not work here**.
Use the container (`underwater-scan:5.1`, already built, contains Isaac Lab + rsl_rl +
marinelab installed into `/isaac-sim`'s bundled Python):

```bash
./docker/run.sh                      # interactive shell, cwd = <repo>/isaaclab
./docker/run.sh '<command>'          # one-shot
./docker/run.sh --gui '<command>'    # X11-forwarded Isaac Sim window (needs DISPLAY; not over ssh)
```

`docker/README.md` is the authoritative operational doc (image build, X11 cookie trick,
root-vs-host file ownership, the three stock-image bugs worked around in
`docker/setup_in_container.sh`). `isaaclab/_isaac_sim` is a **dangling symlink on the host**
by design — it resolves to `/isaac-sim` only inside the container.

Never create a conda/venv for marinelab: `isaaclab.sh -p` would switch to that interpreter
and the install would land where Isaac Sim never looks.

## Commands

### Tests (no Isaac Sim needed — run natively, fast)

`marinelab/tests/conftest.py` stubs `isaaclab` (with a real quaternion implementation) and
shims the `marinelab` package so its heavy `__init__` never runs. Only torch is required:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/mjkim/.conda/envs/acmpc_sim/bin/python \
  -m pytest marinelab/tests/ -q                          # 245 tests
# single file / single test
... -m pytest marinelab/tests/pkrc_wallscan/test_geometry.py -q
... -m pytest marinelab/tests/test_eval_metrics.py -q -k crab
```

The NMPC layer added `test_mpc_reference.py`, `test_wall_frame_ekf.py`, `test_diff_wmpc.py`,
`test_thruster_allocation.py`, `test_hydro_axis_consistency.py` — all still native-only, because
`mpc_reference`, `wall_frame_ekf`, `diff_wmpc` and `estimator_loop` hold no isaaclab import.
Only `mpc_controller.py` needs acados (hence the container), and it is deliberately thin.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required — third-party plugins in that env break collection.

### Train

```bash
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/train.py \
  --task Isaac-PKRC-WallScan-Stage3-Direct-v0 --num_envs 2048 --headless \
  --max_iterations 5000 --run_name stage3'
```

From-scratch reproduction is a two-leg chain: Stage3 (5000 iter) → Train (3000 iter, resumed).
`--num_envs 4096` is the README recipe but was trained on a bigger GPU; this host's 16 GB
RTX 4080 wants 2048.

Resume **must** go through the CLI — `--resume --load_run <run-folder>`. The hydra override
`agent.resume=True` is silently overwritten by the CLI default. Confirm via the
`Loading model checkpoint from:` log line.

Logs/checkpoints land in `<cwd>/logs/rsl_rl/pkrc_wallscan/<timestamp>_<run_name>/`, i.e.
`isaaclab/logs/...` under the normal working directory (gitignored).

### Play / evaluate

```bash
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/play.py \
  --task Isaac-PKRC-WallScan-Eval-Direct-v0 --num_envs 4 --headless \
  --checkpoint ../checkpoints/rb_train_model_7998.pt'
```

Plain `play.py` loops until killed. It writes `checkpoints/exported/policy.{pt,onnx}`
immediately after loading — a quick check that the checkpoint actually loaded.

`--log_traj` runs a bounded rollout, scores scan quality, and writes
`results/{trajectory,metrics}_<tag>.{npz,json,png}`:

```bash
... --log_traj --eval_steps 18500 --score_episode 1 --tag nominal
```

**`--eval_steps 18500 --score_episode 1` is not optional.** `_reset_idx` randomizes
`episode_length_buf` on the initial full reset (standard Isaac Lab episode-end
decorrelation), so every env's *first* episode averages 90 s of 180 s. Rate metrics are
unaffected, but cumulative `scan cycles completed` comes out ~halved (0.62 vs 2.00 measured).
Nominal = `Stage3` task, stress-DR = `Eval` task.

### acados toolchain setup

acados v0.5.3 + `acados_template` 0.5.1 + casadi 3.7.0 are built for the container and
**bind-mounted, not baked into the image** — `docker/run.sh` adds `/opt/acados` plus
`ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH` and `PYTHONPATH=/opt/acados/pysite` whenever
`~/docker/acados` exists, and silently skips them when it does not (so the plain RL
workflow still runs on a machine without acados). Rebuild after a fresh clone:

```bash
cp -a ~/acados ~/docker/acados && rm -rf ~/docker/acados/{build,lib} && mkdir ~/docker/acados/build
./docker/run.sh 'cd /opt/acados/build && cmake -DCMAKE_BUILD_TYPE=Release -DACADOS_WITH_HPIPM=ON \
  -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR=/opt/acados .. && make -j12 install'
./docker/run.sh 'git config --global --add safe.directory /opt/acados
  /isaac-sim/python.sh -m pip install --target /opt/acados/pysite "casadi==3.7.0"
  /isaac-sim/python.sh -m pip install --no-deps --no-build-isolation --target /opt/acados/pysite \
    /opt/acados/interfaces/acados_template Deprecated wrapt'
rm -rf ~/docker/acados/pysite/numpy*   # MUST go: it shadows Isaac's numpy 1.26 and breaks torch
```

Two traps that already cost time here: `pip --target` drags in numpy 2.x which shadows
Isaac's 1.26 via `PYTHONPATH` (delete it), and `acados_template`'s setuptools-scm build
fails on the mounted tree until `git config --global --add safe.directory /opt/acados`
(root-in-container vs uid-1000 files → "dubious ownership").

`isaaclab/logs/_probe_acados.py` (gitignored) is the smoke test: it builds the nominal +
sensitivity solver pair the way `DiffCylinderOrbitMPCController` does and checks
`eval_solution_sensitivity` against finite differences. Measured on this host: sensitivity
matches FD to 6-7 digits at an interior solution, ~0.8 ms/step for a toy N=20 OCP — and
**zero when the control saturates**, which is exactly why the learner skips saturated steps.
`isaaclab/logs/_probe_plant.py` dumps the PhysX plant parameters an MPC model must match.

### Run / train the NMPC controller

`scripts/run_wallscan_mpc.py` drives the *unmodified* `WallScanEnv` from `WallScanMPC` instead of
a policy, so `eval_metrics` numbers land on the same footing as `play.py --log_traj`:

```bash
# scoring run (headless). --tam fixed / --hydro z_slender are the MEASURED corrections and are
# not yet the cfg defaults, so they must be passed explicitly.
./docker/run.sh './isaaclab.sh -p -u ../marinelab/scripts/run_wallscan_mpc.py \
  --tam fixed --hydro z_slender --state ekf --sensors datasheet \
  --num_envs 8 --steps 9000 --seed 0 --tag nmpc'

# with learned cost weights instead of DEFAULT_WERR
  --policy_ckpt ../checkpoints/dw_ekf/policy_final.pt
# stress DR (the RL policy's published stress condition); pair with --sensors placeholder,
# whose bias magnitudes equal that task's sensor DR
  --task eval --sensors placeholder

# watch it: live window (local desktop only) or offscreen recording (works over ssh)
DISPLAY=:1 ./docker/run.sh --gui '... --render --cam overview'
./docker/run.sh '... --video --video_length 9000 --cam overview'
```

`-u` matters: without it the `t=` progress lines sit in the stdout buffer for the whole run.
`--cam` is not cosmetic — the stock viewer eye (7.5, 7.5, 7.5) is *inside* the R=6 m wall.

Diff-WMPC weight learning (`scripts/train_diff_wmpc_wallscan.py`, `--num_envs` is fixed at 1):

```bash
./docker/run.sh './isaaclab.sh -p -u ../marinelab/scripts/train_diff_wmpc_wallscan.py \
  --task stage3 --state ekf --sensors datasheet --tam fixed --hydro z_slender \
  --steps 40000 --seed 0 --ckpt_dir ../checkpoints/dw_ekf'
```

Native analysis tools (no container, seconds): `rescore_trajectories.py` recomputes every
`results/metrics_*.json` with current metric definitions; `plot_wall_distance_compare.py` draws
the standoff comparison; `replay_wall_frame_ekf.py` re-runs the filter off a saved trajectory.

### Lint

ruff (line length 120, isaaclab-style import sections) and pyright are configured in
`marinelab/pyproject.toml`; neither is installed in the local conda env. `ruff check marinelab/`
+ `ruff format marinelab/` are the pre-commit expectation per `marinelab/CONTRIBUTING.md`.

### Git LFS

`.usd`/`.dae`/`.obj` meshes are LFS-tracked. `git lfs pull` before running anything, or USD
loads fail at sim startup (git-lfs is at `~/.local/bin`).

## Architecture

### `marinelab/marinelab/core/` — the stable framework API

Vehicle- and task-agnostic marine physics, importable from `marinelab.core`
(`marinelab.physics` was removed): `HydrodynamicsModel` (Fossen 6-DOF: added mass,
linear/quadratic damping, buoyancy/restoring), `ThrusterModel` (first-order lag + 6×6
allocation matrix), `OceanCurrent` (injectable, shared across models), `HydroParams`.

Domain randomization goes through `set_parameters` (absolute) / `scale_parameters`
(`base * uniform(lo,hi)`), both reading an immutable base snapshot taken at construction —
repeated randomization never compounds. See `marinelab/README.md` for the full API reference
and `marinelab/docs/architecture.md` for the subpackage table.

`tasks/bluerov/` is the reference consumer (5 registered hover/attitude envs) and the
pattern to copy; `tasks/pkrc_wallscan/` is the actual research task.

### `tasks/pkrc_wallscan/` — the wallscan task

Five Gym IDs, one `WallScanEnv`, cfg-only differences forming a curriculum ladder
(`wallscan_env_cfg.py`):

| Task ID | vertical | sway | hard collision term | dynamics DR | sensor DR | DORAEMON |
|:---|:--|:--|:--|:--|:--|:--|
| `Isaac-PKRC-WallScan-Stage1-Direct-v0` | — | — | — | — | — | — |
| `…-Stage2-Direct-v0` | ✓ | — (`skip_sway`) | — | — | — | — |
| `…-Stage3-Direct-v0` | ✓ | ✓ | — | — | — | — |
| `…-Train-Direct-v0` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `…-Eval-Direct-v0` | ✓ | ✓ | ✓ | wider | wider | — |

`Eval`'s DR, spelled out because it is what breaks the NMPC: spawn roll/pitch ±45°, added mass /
linear / quadratic damping ×0.5–1.5, **volume ×0.85–1.15 (buoyancy ±34 N, vs 40 N per thruster)**,
thrust coefficient and thruster τ ×0.7–1.3, CoB/CoG offsets ±5 cm, inertia ×0.8–1.2, sonar mount
±8 cm / ±0.04 rad, per-episode sensor biases doubled.

Module split, and why it matters: `geometry.py`, `sensors.py`, `scan_state_machine.py`,
`eval_metrics.py` are **pure torch/numpy — no isaaclab or pxr imports**, which is what makes
them unit-testable without the sim app. `__init__.py` therefore resolves `WallScanEnv` and the
cfgs through a PEP 562 `__getattr__` lazy map; importing the package must not pull in isaaclab.
Keep new logic in this pure layer whenever it is testable there.

`wallscan_env.py` (`WallScanEnv(DirectRLEnv)`) is the body. Isaac Lab's fixed step order:

```
_pre_physics_step(actions) → [physics dt × decimation] → _get_dones → _get_rewards
  → (done envs) _reset_idx → _get_observations
```

Three invariants that the surrounding code depends on:

1. **Rewards read ground truth; observations read sensors.** `_read_state()` is the split
   point (the sim2real seam). `sensors.apply_sensors` adds noise + per-episode bias for obs;
   sonar obs is measured from the *DR'd* mount pose. Penalizing the policy for error it cannot
   observe breaks convergence — empirically twice.
2. The scan state machine latches `z_hold` and takes its sway input from **GT** (`z_latch`,
   `_s_gt`), so DVL bias drift cannot poison the reward reference.
3. In `_reset_idx`, DORAEMON's success evaluation must happen **before** `_cycles` is zeroed.

#### Observation vector — 31-D, assembled in `_get_observations` (`wallscan_env.py:302-321`)

Control rate is 50 Hz (sim dt 1/100 × decimation 2 → `step_dt` 0.02 s). Every channel is a
*sensor* reading: white noise resampled each step + a constant per-episode bias (DR stages only).

| Slice | Block | Dim | Sensor | noise std / per-episode bias knob |
|:--|:--|:--|:--|:--|
| 0:3 | `up_vec` (body +z in world) | 3 | INS (3DM-GV7) attitude | `ins_noise` 0.01 / `ins_att_bias_dr` |
| 3:5 | `heading_sc` = (sin, cos) | 2 | INS attitude | `ins_noise` / `ins_att_bias_dr` |
| 5:8 | `ang_vel` (body) | 3 | INS gyro | `ins_noise` / `ins_gyro_bias_dr` |
| 8:11 | `lin_vel` (body) | 3 | DVL-A50 | `dvl_noise` 0.02 / `dvl_bias_dr` |
| 11:12 | `sonar` wall range | 1 | Ping1D single-beam | `sonar_noise` 0.05 / `sonar_bias_dr` |
| 12:13 | `depth` | 1 | pressure depth | `depth_noise` 0.02 / `depth_bias_dr` |
| 13:15 | `ukfm_xy` | 2 | UKF-M (surface ArUco marker) | `ukfm_noise` 0.03 / `ukfm_bias_dr` |
| 15:16 | `ukfm_yaw` | 1 | UKF-M | same |
| 16:17 | `ukfm_valid` | 1 | visibility gate | — |
| 17:20 | `cmd_err` = (sonar−`d_ref`, depth−`z_ref`, ŝ−`s_ref`) | 3 | derived | — |
| 20:22 | `sin`/`cos(yaw_err)` | 2 | derived | — |
| 22:23 | `search_flag` | 1 | internal | — |
| 23:25 | `phase_sc` = (sin, cos) of phase/4 | 2 | internal state machine | — |
| 25:31 | `prev_action` | 6 | last applied action | — |

Sensor suite per `sensors.py:15`: Ping Sonar / pressure depth / 3DM-GV7 INS / DVL-A50 / UKF-M.
The noise magnitudes are explicitly flagged there as **conservative placeholders pending real
datasheets** — do not treat them as characterized values.

Three details that are easy to miss:

- The observed sonar is ray-cast from the **DR'd mount pose** (`wall_dist_meas`), while the
  reward uses the nominal mount (`wall_dist`) — `_read_state` computes both (`:252-257`).
  Nominal mount is body (0.10, 0.0) m; Train/Eval jitter it by ±0.08 m / ±0.04 rad per episode.
- `ukfm_*` is only valid when `|z| < ukfm_valid_max_depth` (8.0 m) **and** tilt < 0.5 rad
  (`sensors.py:90-95`); otherwise noise/bias are zeroed and `ukfm_valid` goes 0 to tell the
  policy the fix is stale.
- ŝ in `cmd_err[2]` is the **estimate** `_s`: DVL body-y dead reckoning corrected 0.2× toward the
  marker fix when visible, frozen during the spin search (`wallscan_env.py:420-433`). The true
  arc length `_s_gt` (GT angle-increment accumulation) is reward/phase-timing only.

Caveat worth knowing before any real-robot port: the *references* inside `cmd_err`/`yaw_err` are
env-side and GT-derived — `_yaw_ref_cur = theta_gt` (`:500`) and the state machine's sway timing
uses `_s_gt` (`:442`). The policy only sees the error, and the code documents the intent that a
deployed controller recomputes the same bearing from the UKF-M position (`:497-499`), but the
substitution is not implemented here.

Tank walls are **analytic**, not physics: `geometry.wall_distance` / `radial_clearance`
compute sonar range and collision; `tank.py`'s cylinder is visual-only with collision
disabled (a solid convex collider would depenetrate a robot spawned inside it on GPU PhysX).

Phase machine: `DESCEND(0) → SWAY_A(1) → ASCEND(2) → SWAY_B(3)` and wrap. A phase advances
when `|error| < reach_eps` holds for `reach_hold` consecutive steps; references *ramp* toward
phase targets at the real scan speed (`ref_step`, `ref_step_s`) rather than jumping.

Rewards (`_compute_reward_terms`) are ~13 terms: exp-tracking on wall distance / heading /
depth / sway, potential-based progress shaping (Ng 1999, so per-phase optimality is preserved),
waypoint bonus, collision, upright + linear tilt penalty, cross-velocity, overspeed cap,
action-rate/magnitude, alive. Diagnostics are logged as `Scan/end_phase`, `Scan/end_cycles`,
`Scan/term_*` and `DORAEMON/*`.

### DORAEMON adaptive DR (Train stage only)

`marinelab/algorithms/doraemon.py` is the engine (Tiboni et al., ICLR 2024: 13-D Beta
distribution, importance-sampled success-rate estimate, entropy maximization inside a KL trust
region). `tasks/pkrc_wallscan/doraemon_dr.py` is the glue: `PARAM_DEFS` (13 dynamics params),
`build_scheduler`, `apply_xi` (per-env injection through the core tensor paths). It only starts
working once episodes complete ≥1 scan cycle — which is why Stage3 exists. Known MVP gap:
scheduler state is **not** restored across `train.py --resume`; the DR distribution restarts at
`init_concentration`.

### PPO

Hyperparameters live in `tasks/pkrc_wallscan/agents/rsl_rl_ppo_cfg.py` (inheriting the bluerov
hover cfgs); the implementation is the Isaac-bundled `rsl_rl`. Actor/critic [128,128,64] ELU,
obs 31 → action 6, 48 steps/env, adaptive LR (desired_kl 0.01), γ=0.995.

### NMPC + Diff-WMPC (the second controller, alongside RL)

An acados NMPC that replaces the policy entirely, plus a small network that supplies its cost
weights (Diff-WMPC, ported from `~/Underwater-Actor-Critic-Model-Predictive-Control1`). Built to
attack two things the RL policy does badly: heading accuracy (a single echo sounder does not
measure wall distance once yaw rotates off the normal) and tilt during the sway legs.

| Module | Layer | Role |
|:--|:--|:--|
| `mpc_reference.py` | pure | 12-D error vector, cylinder-orbit mapping, ramp **preview** over the horizon |
| `mpc_controller.py` | acados | `PlantParams`, `build_ocp`, `WallScanMPC` (nominal + sensitivity solver pair) |
| `algorithms/diff_wmpc.py` | pure | `WeightPolicy`, `wallscan_loss`, `DiffWMPCLearner` |
| `wall_frame_ekf.py` | pure | EKF over wall-relative `(r, φ, s)` |
| `estimator_loop.py` | pure | `WallFrameEstimator` — sensor synthesis + EKF, **shared by runner and trainer** |

Three formulation choices that carry the result:

1. **Wall-relative state `(r, φ, s)`, not Cartesian.** On a cylinder the wall normal *is* the
   outward radial, so heading is closed-loop from position and the RL policy's ~10 s spin search
   is unnecessary (a 07-27 audit found that search's bearing landed 5–84° off the true normal).
2. **Reference preview.** The state machine's ramp is rolled forward over all N stages rather
   than presented as a frozen setpoint, so the solver decelerates *into* a phase endpoint.
3. **`d_ref` is the SONAR standoff, not the body standoff** (confirmed 2026-08-03), so
   `r_des = R − d_ref − sonar_mount_x` = 4.40 m. Getting this wrong left a 10 cm wall error.

Diff-WMPC specifics: cost weights → `p_global`, `eval_solution_sensitivity` gives `∂x*/∂p`, and
`gW = Sxᵀ ∂L/∂x + Suᵀ ∂L/∂u` is backpropagated into the weight network. The loss **must sum over
several horizon nodes** — see the traps below.

Nominal-condition results, 8 envs × 3 seeds = 24 spawns, settled window (first 20 s dropped;
the spawn hands up to 180° of heading error and correcting it is a transient, not scan quality):

| | RL policy (published) | **NMPC + EKF, learned weights** | NMPC + EKF, hand-tuned |
|:--|--:|--:|--:|
| crab (yaw − θ) | 1.50° | **0.247 ±0.002°** | 0.191° |
| tilt (heave / sway) | 0.90 / 2.20° | **0.183 / 0.170°** | 0.136 / 0.136° |
| wall standoff error | — | **1.10 cm** (RMS 1.26, p95 2.13, max 3.06) | 1.24 cm |
| scan speed (heave / sway) | 0.199 / 0.123 | 0.200 / 0.101 (target 0.20 / 0.10) | 0.200 / 0.101 |
| EKF `r` / `φ` / `s` RMSE | — | 1.24 cm / 0.32° / 11.6 cm | — |
| cycles, terminations | 2.0 | 2.00, 24/24 time-out (0 collided/oob/tilted) | same |
| solve time | — | 6.6 ms/step, QP fail 0.00%, saturated 1.6% | 6.5 ms, 0.00%, 0.01% |

The bolded column is `results/metrics_dw8_*.json`, which carries
`policy_ckpt=checkpoints/dw_ekf/policy_final.pt` — those are **learned** weights, not hand-tuned
ones; the column was mislabelled here until 2026-08-04. Hand-tuned (`metrics_hon_*.json`,
`DEFAULT_WERR` with `w_s = 2000`) is *better* on both attitude metrics and slightly worse on
standoff, so "Diff-WMPC beats hand tuning" is **not** supported at nominal conditions — the
comparison the learned weights do win is against the RL policy.

6–13× better than RL on the two target metrics either way, and the NMPC runs off *estimates* while
the RL policy's bearing reference is GT (`_yaw_ref_cur = theta_gt`). Solve is 3× inside the 20 ms
control period.

**Under stress DR the advantage disappears.** 2×2 decomposition (same 24 spawns, `settled`):

| cell | train | eval | crab | tilt (heave) | wall err | collided |
|:--|:--|:--|--:|--:|--:|--:|
| A | nominal | nominal | 0.247° | 0.183° | 1.10 cm | 0 |
| C | nominal | **DR** | 2.001° | **13.826°** | 5.52 cm | 0 |
| D | **DR** | **DR** | 3.884° | 13.342° | 13.95 cm | **1** |
| B | nominal | DR + placeholder sensors | 2.427° | 14.065° | 5.61 cm | 0 |

Paired per-seed effects: **DR (C−A) +13.64 ±2.73° tilt, decided**; sensor model (B−C) +0.24°,
i.e. 1.7% of the DR effect; **DR training (D−C) undecided on every metric** and it introduced the
only collision in 96 DR spawns. RL's published stress-DR tilt is 14.3° — statistically the same
as C's 13.83°, so under DR the two approaches tie.

The mechanism is **saturation, not solver failure**: QP fail stays 0.00% and solve time is
unchanged, but saturated steps go 1.6% → 39%. `Eval`'s `volume_scale=(0.85,1.15)` is ±34 N of
buoyancy against a 40 N thruster, and the MPC model is nominal — a pure NMPC has no integral
action, so a constant disturbance leaves a standing offset. This also explains why DR *training*
cannot help: sensitivity is exactly zero at saturation, so the learner skips those steps
(`skipped` went 39% → 87%, `updates` 3757 → 2415) and never sees the regime that is failing.
`w_z` stayed 1308 → 1347. **Fix the model (online buoyancy/mass estimation), not the weights.**

### That model fix was built and measured — and it found an actuation limit instead (2026-08-04)

`wrench_observer.py` (pure numpy, `--dobs` on `run_wallscan_mpc.py`, default **off**) estimates the
residual wrench `[force_world(3), moment_body(3)]` from the applied thrust and measured velocity,
and hands it to the solver through `mpc_reference.ND`, which grew 3 → 6 for it. The force slot had
been reserved in the model since the first NMPC commit; the **moment half was the addition that
mattered**, because a force resolved at the body origin makes no moment and the failing metric is
tilt. Stress DR, 8 envs × 3 seeds, paired per seed, every sign unanimous:

| | tilt heave | tilt sway | crab | wall err | saturated | QP fail |
|:--|--:|--:|--:|--:|--:|--:|
| observer off (`dwDRds`) | 13.83° | 13.27° | 2.00° | 5.52 cm | 39% | 0.00% |
| observer on (`dwDRobs`) | 11.44° | 11.09° | 3.01° | 9.64 cm | 67% | 0.00% |

Nominal is near-neutral (tilt 0.183 → 0.239°, standoff and saturation unchanged) — and that
+0.06° is the **unmodelled thruster lag** being fed back as disturbance: on a correct plant the
observer still reads |d_f| ≈ 2 N. So the observer works (estimates inside their physical bounds,
zero windup, `clipped` 0–4353 / 71968) but **is not a net win**, hence default off.

Channel ablation (`--dobs_channels`, 3 seeds each) says the force half is irrelevant: moments-only
11.59° / 9.64 cm, Fz+moments 11.49° / 9.52 cm, all-six 11.44° / 9.64 cm — **indistinguishable**.
The 16–22 N buoyancy residual on Fz changes nothing, because z tracking was never what DR broke
(weight 40 on a directly measured channel). A tempting hypothesis that measurement killed:
|Fx|, |Fy| ≈ 4–7 N really is damping DR resolved through a tilted vehicle (0.2 m/s heave at 10°
tilt × 97.79 N·s/m ≈ 3.4 N, magnitude matches), and exporting a velocity-dependent residual as a
horizon constant really is wrong — but withholding it moved the standoff by −0.12 cm. The radial
axis is servoed at weight 40; a small model bias cannot move that equilibrium. The +4.1 cm is
saturation stealing authority.

**Root cause of the residual 11.4°: under `PKRCThrusterCfgFixedTAM` the pitch row of the
allocation matrix is all zeros — `tau[4] ≡ 0`, pitch has no actuator.** The sway pair sits at
z = −0.09 m so it makes a *roll* moment, and the heave pair sits at x = 0. An unactuated moment is
balanced only by restoring, `B·z_cob·sin θ = 34.29·sin θ`:

| seed | \|Mx\| (actuated) | \|My\| (**unactuated**) | predicted pitch eq. | measured tilt (obs on) |
|:--|--:|--:|--:|--:|
| 0 | 6.96 N·m | 6.02 N·m | 10.12° | 8.49° |
| 1 | 4.06 | 9.28 | 15.71° | 12.07° |
| 2 | 8.27 | 9.81 | 16.62° | 13.76° |

Ordering matches, magnitude lands at 80–85% of prediction. **The stress-DR tilt is an
unactuated-axis equilibrium, not a control failure.** The observer buys 2.2–2.4° by trimming roll
(which *is* actuated) and pays for it in standoff and saturation, because roll comes from the sway
pair plus a heave differential — the same thrusters running both scan legs and the buoyancy offset.
This is also the mechanical reason `--w_roll` was never going to reach it: that flag raises
`werr[8]` and `werr[9]` together and the pitch weight has no actuator behind it.

Note the shipped `PKRCThrusterCfg` puts the arm on the pitch row instead, making pitch look
actuated. Per `pkrc.py` the fixed TAM is the physically correct one, so the shipped matrix granted
the controller authority the vehicle does not have.

**Do not spend more effort on estimation or weights for the DR tilt.** The next honest lever is
actuation: a thruster pair with a pitch arm, or accepting that Eval-level DR (±15% volume = ±34 N
against a 40 N thruster) is outside this vehicle's authority envelope and reporting the envelope.

Remaining known defects, in the order the measurements rank them: **pitch is unactuated** (above,
and it now ranks first); `s` drifts 60–91 cm under DR because UKF-M updates only `(r, φ)` and never
`s`; the sonar gate is σ-based so a good sensor rejects unmodeled mount error as an outlier (716
gated under DR with datasheet sensors vs 193 with the noisier placeholder model); `estimator_loop`
synthesizes DVL body x/y and gyro z but passes body `v_z` and `w_x`/`w_y` through from **ground
truth** — both instruments are 3-axis in reality, so this is under-modelling, and it means the
observer's force channel is partly GT-fed (immaterial to the conclusion above, since that channel
turned out not to matter).

## Traps that have already cost time

- **σ (exploration std) is chronically unstable with `clamp(-1,1)` actions.** `scripts/train.py`
  monkey-patches `ActorCritic._update_distribution` to clamp the scalar std into [0.1, 1.5]
  (both collapse and blowup — 0.4→5.4 over 8k iters — were measured). Do **not** replace this
  script with upstream `train.py`. Relatedly `entropy_coef` is forced to **0.0**: no positive
  entropy bonus proved safe long-term. Watch `Mean action noise std` in the logs; healthy is
  ~0.1–0.5, above 1.0 is a red flag.
- **Equilibrium vs. reach-band contention** (hit 3×): if a "zero-effort" equilibrium (tracking
  equilibrium, positive-buoyancy float at z≈9.5) sits just outside a reach band, the policy
  parks there forever. When moving targets or bands, first compute where the vehicle goes with
  no effort, and re-verify the clearance to the kill boundaries (z<0.15, z>10.2 — those are
  physics-runaway guards only; surface and floor are the real limits).
- **Exp-tracking pays ~0 far from target**, so transit speed needs explicit shaping.
- The dense numeric comments in `wallscan_env_cfg.py` / `rsl_rl_ppo_cfg.py` are a measured
  experiment log with dates and run names. Treat a constant with a comment as a conclusion, not
  a free knob; if you change it, append the evidence rather than deleting the history.
- Diagnose with instrumentation (`Scan/*` telemetry, deterministic-policy trajectory dumps),
  not blind knob turning.
- `train_and_push.sh` (train → commit checkpoint → push) has paths hardcoded to
  `/root/home/rl_ws/...`, which is **not** this host's container layout
  (`/workspace/UnderwaterScanningRobot`). Fix the paths before using it here.
- Note `README.md`'s "84 tests" is stale — the suite is 245.

### NMPC-specific traps (all measured)

- **One acados solver instance cannot serve N envs.** An RTI solver warm-starts from whatever
  state the *previous* `solve()` left it in. With one env that is the previous timestep (correct);
  with 8 envs in a loop it is a different vehicle 20 m away, and 8 RTI iterations do not recover.
  Measured: crab 0.24° → **23.41°**, wall error 0.47 → **116 cm**, saturated 0.79 → 13.6%, while
  QP fail stayed 0.00% — nothing *looked* broken. Fix: one `WallScanMPC` per env, `generate`/
  `build` only on the first. Same shape of bug applies to `WallFrameEstimator`, which holds one
  vehicle's filter/biases: one instance per env, and reset only the envs that actually reset.
  **Single-env runs hide this completely.**
- **Coarse RK4 diverges on the yaw axis**, not because of saturation or quaternion norm (both of
  which I wrongly blamed first). `rk4_substeps=5` fixed 100% `ACADOS_MINSTEP` QP failures.
- **The Diff-WMPC loss must sum over several horizon nodes.** At a single node 1.0 s out, radial
  drift barely moves inside the window, so the gradient drove `w_radial` to 2–3 and the vehicle
  crabbed 150.7°. `sens_nodes` defaults to `{N/6, N/3, 2N/3, N}`.
- **Sensitivity is exactly zero at saturation**, so the learner skips saturated steps. That is
  correct behaviour but it means Diff-WMPC is blind wherever control authority is the binding
  constraint (see the DR result above).
- **The `done` row already holds POST-reset state.** Isaac Lab resets inside `step()` and both
  loggers read afterwards, so `episode = cumsum(done)` labels the state and
  `ended_episode = cumsum(done) − done` labels what *finished*. `eval_metrics.episode_indices`
  returns both; using the wrong one lets one across-the-tank sonar reading (8.64 m vs a 1.5 m
  target) take standoff RMS from 0.7 to 11.0 cm. Means barely move, max-type statistics become
  meaningless. Reuse `eval_metrics`, do not reimplement the mask.
- **Leg completeness must be decided by boundary KIND, not run index.** "Drop the first and last
  run" lets a truncated leg through, because a timeout splits off a 1-row run. The old leg-speed
  definition averaged `|d/dt|` over every row of a phase, so a truncated leg's settling
  oscillation counted as forward progress and reported a 16% heave overshoot that did not exist.
- **`WeightPolicy` bounds must be buffers.** Left as plain attributes they are absent from
  `state_dict`, so the same network produced different cost weights after a reload — silently.
- Metrics are reported as **means**; a mean is robust to the outliers above but hides them. When
  a number looks too good, check RMS / p95 / max as well.
- Under `--task eval` the physical sonar mount is re-drawn every reset. The estimator synthesizes
  the range from the TRUE mount and its filter predicts from the nominal one; wiring both to
  nominal (the original code) silently deleted the entire mount DR.

## Coexisting-cfg convention (four of these are live)

Plant/sensor corrections found by measurement are added as **new cfg classes next to the legacy
ones**, with the evidence in the docstring, a guard test, and the legacy class left as the
default — so no previously-published number silently changes meaning. Currently:

| new class / flag | what it corrects | must be passed explicitly |
|:--|:--|:--|
| `PKRCThrusterCfgFixedTAM` | TAM put the sway moment arm on the pitch row where no actuator can cancel it; moving it to roll lets a heave differential trim the sway leg's parasitic moment (2.298° predicted = measured; compensated → 0.000°) | `--tam fixed` |
| `PKRCHydrodynamicsCfgZSlender` | shipped coefficients claim an x-slender hull; the USD mesh is z-elongated. Roughly halves heave drag — the wallscan's primary axis | `--hydro z_slender` |
| `SensorCfg.ukfm_gate="depth_below_surface"` | the legacy validity gate is inverted | `--ukfm_gate depth_below_surface` (already default in the NMPC scripts) |
| `SensorCfgDatasheet` | 3DM-GV7 + Ping1D + DVL-A50 published figures instead of the placeholder guesses; adds the 25° beam cone, the DVL scale-factor form and its 15 Hz zero-order hold | `--sensors datasheet` |

`marinelab/core/thruster.py:allocation_moment_residual` and
`marinelab/core/parameters.py:slender_axis_is_consistent` are the checks that catch these classes
of error; both are unit-tested.

## Reference docs in-repo

- `docker/README.md` — this host's runtime, GUI, reproduction commands
- `marinelab/docs/wallscan-training-code-guide.md` — code walkthrough in execution order (Korean)
- `marinelab/docs/wallscan-project-report.md` — results write-up
- `marinelab/docs/architecture.md`, `marinelab/docs/installation.md`, `marinelab/CONTRIBUTING.md`
- W&B training curves: https://wandb.ai/yju1121-postech/pkrc_wallscan
