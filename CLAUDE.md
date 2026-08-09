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
comparison the learned weights do win is against the RL policy. **This is a nominal-conditions
statement only, and it reverses under DR** (measured 2026-08-05: learned weights hold 5.52 cm of
standoff where the two hand-tuned points give 19–31 cm; see the pitch weight A/B section).

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
(**corrected 2026-08-07** — this line read "`skipped` went 39% → 87%", which reconciles with
nothing: `updates` 3757 → 2415 at `batch_size` 10 over 40k steps means **6.1% → 39.6%**, and the
checkpoints still on disk confirm both counts. The 39% appears to be the *evaluation* saturation
fraction mistaken for a training skip rate. The corrected DR figure landing on 39.6% ≈ the 39%
saturation it is caused by is the consistency check the wrong numbers lacked.)
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
saturation stealing authority. (**Qualified 2026-08-05**: a later A/B reaches 31 cm of standoff
error at *lower* saturation than this run, so saturation is not the dominant term — see the pitch
weight A/B below.)

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
`werr[8]` and `werr[9]` together and the pitch weight has no actuator behind it. (**Wrong on the
last clause, measured 2026-08-05**: splitting the pair shows `werr[9]` alone moves pitch by
3.6–5.0°. The flag's real defect is that it *couples* the two axes, not that its pitch half is
inert — see the pitch weight A/B below.)

Note the shipped `PKRCThrusterCfg` puts the arm on the pitch row instead, making pitch look
actuated. Per `pkrc.py` the fixed TAM is the physically correct one, so the shipped matrix granted
the controller authority the vehicle does not have.

**Do not spend more effort on estimation or weights for the DR tilt.** The next honest lever is
actuation: a thruster pair with a pitch arm, or accepting that Eval-level DR (±15% volume = ±34 N
against a 40 N thruster) is outside this vehicle's authority envelope and reporting the envelope.
(**Partly superseded 2026-08-05** — pitch turns out to respond to its cost weight after all, and
weights turn out to dominate standoff. See the next section; the *actuation* half of this
paragraph still stands, the *weights* half does not.)

### Pitch weight A/B: the zero TAM row does NOT make the pitch weight inert (2026-08-05)

Direct test of "weight on an unactuated axis can only perturb the QP": stress DR, 8 envs × 3
seeds, everything identical except `werr[9]` (pitch) and `werr[11]` (`w_y`), with `werr[8]`
(roll) pinned at 2000 in both arms. `results/metrics_p3a2_*` (pitch 2000) vs `p3b2_*`
(pitch 0.1). Reruns are **bit-identical** to the first pass (`max|Δtilt| = max|Δwall| = 0`), so
these runs are deterministic given the seed.

Splitting tilt into its actuated and unactuated halves is what settles it — mean |tilt| alone
cannot, which is why `run_wallscan_mpc.py` now logs `roll_deg` / `pitch_deg` into the npz:

| seed | Δ\|roll\| (B−A) | Δ\|pitch\| (B−A) | Δ\|tilt\| |
|:--|--:|--:|--:|
| 0 | −0.15° | **+3.64°** | +3.38° |
| 1 | +0.02° | **+4.99°** | +4.94° |
| 2 | −0.75° | **+3.73°** | +3.13° |

Roll does not move and its sign is not even consistent; **the whole effect is pitch**. So the
solver does suppress pitch by 30–35% when the weight asks it to, despite `tau[4] ≡ 0` — through
model coupling, not through a pitch actuator. Per-env numbers confirm the mechanism scales with
the disturbance: envs whose CoB draw leaves pitch near zero (s2 env1 at 1.4°, s0 env6 at 2.4°)
are *identical* in the two arms, while the large-pitch envs move the most.

The `asin(cob_x / z_cb)` equilibrium above survives, but as a **free-response upper bound**, not
a prediction of closed-loop tilt: arm B's pitch implies `cob_x` 1.9–4.1 cm (inside the ±5 cm
draw) i.e. it sits near free equilibrium, while arm A holds ~35% below it by spending authority.

Two corrections to the section above, both with the same root:

| config (roll = 2000 throughout) | tilt | wall err | saturated | collided |
|:--|--:|--:|--:|--:|
| hand, radial 40, pitch **2000** (`p3a2`) | 11.34° | **31.23 cm** | 24.4% | **1** |
| hand, radial 40, pitch **0.1** (`p3b2`) | 15.01° | 19.11 cm | **5.2%** | 0 |
| **learned Diff-WMPC weights** (`dwDRds`) | 13.83° | **5.52 cm** | 39.2% | 0 |

1. **Saturation is not what drives standoff.** The 24%-saturated arm is 6× *worse* on wall error
   than the 39%-saturated learned run. "The +4.1 cm is saturation stealing authority" is at best
   incomplete — weight balance dominates. Radial at 40 against roll/pitch at 2000 buys attitude
   with 30 cm of standoff, and it is the config that produced the only collision.
2. **Under DR the learned weights beat hand tuning by 3.5–5.7× on standoff**, at a tilt between
   the two hand-tuned points — i.e. they sit on a strictly better trade-off curve. This is the
   opposite of the nominal-condition finding (where hand tuning wins on attitude), and it was
   invisible until now only because the "DR + hand-tuned" cell had never been run.

Flooring the pitch weight is therefore **rejected as a fix**: it buys saturation and costs 4° of
pitch, and its standoff effect is undecided (seed 2 reverses).

### The Diff-WMPC DR result is a FIXED weight vector, not state-dependence (2026-08-05)

A `radial × pitch` factorial at `roll=2000` (`results/metrics_pf_r{400,2000}_p{2000,01}_*`, with
`p3a2`/`p3b2` as the `radial=40` row) does not reach the learned point:

| radial | pitch | tilt | \|pitch\| | wall err | sat | crab | collided |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 40 | 2000 | 11.34° | 9.93° | 31.23 cm | 24.4% | 10.92° | 1 |
| 40 | 0.1 | 15.01° | 14.05° | 19.11 cm | 5.2% | 13.26° | 0 |
| 400 | 2000 | 12.37° | 11.08° | **14.67 cm** | 28.0% | 12.90° | 0 |
| 400 | 0.1 | 15.01° | 14.04° | 18.84 cm | 6.6% | 13.58° | 0 |
| 2000 | 2000 | 13.32° | 12.00° | 17.87 cm | 26.3% | 14.65° | 1 |
| 2000 | 0.1 | 15.03° | 14.04° | 19.49 cm | 7.1% | 13.81° | 0 |
| learned | | 13.83° | | **5.52 cm** | 39.2% | **2.00°** | 0 |

**But this grid settles nothing, because it swept the wrong two axes** — recorded so the mistake
is not repeated. Probing `checkpoints/dw_ekf/policy_final.pt` directly shows why: the learned
vector is `radial 4265, z 1421, s 4820, v_rad 110, v_tan 1824, v_z 4984, head 1891/1749,
roll 4997, pitch 4530, w_x 117, w_y 0.9` with `wu ≈ 0.005`. Two ratios the grid held fixed at the
wrong value: `head/radial` is **0.44** learned vs **0.05** in the grid (`head` sat at DEFAULT's 20
throughout), and `werr:wu` is ~200× looser than DEFAULT. A single-beam echo sounder over-reads off
normal, so crab *is* standoff error — the grid's 13° crab is the 15–31 cm.

The decisive run: freeze that vector as a constant and run it **without** `--policy_ckpt`
(`metrics_fl_*` vs `dwDRds`, paired, 3 seeds):

| | tilt heave | tilt sway | crab | wall err | saturated |
|:--|--:|--:|--:|--:|--:|
| live policy (`dwDRds`) | 13.83° | 13.27° | 2.00° | 5.52 cm | 39.2% |
| frozen vector (`fl`) | 13.84° | 13.27° | 1.77° | 5.49 cm | 39.2% |
| paired Δ | +0.015° | +0.006° | −0.231° | −0.038 cm | +0.000 |

**Every metric undecided — the frozen vector is indistinguishable from the live policy.**

Why, stated correctly (an earlier revision of this line said "the weights are near-constant",
which came from a probe that varied the ERROR entries while holding the phase input fixed at
DESCEND — it measured the wrong axis; corrected 2026-08-06 with
`scripts/probe_weight_schedule.py`). Across the four scan phases the legacy policy's weights do
move: median spread 15%, and `w_x` swings 104%. **The variation is real but not load-bearing.**
It sits in entries that cannot move the closed loop — `w_x` ranges 57–171 against a radial weight
of 4265 — while every high-leverage entry is flat: `roll` 0%, `v_z` 0%, `s` 2%, `radial` 7%. So
freezing the vector changes nothing measurable, which is what the paired run above shows
directly. The frozen-vector conclusion rests on that closed-loop result, not on the probe.

Consequences, and they are large:

1. **The "weights-varying" premise is not carrying the DR result — *in this port*.** `dw_ekf`
   functions as a 13-D weight *optimizer*, not a *scheduler*. **This is a statement about our
   implementation, not about the method** (corrected 2026-08-06 after reading the source paper —
   an earlier revision of this section wrongly concluded that expanding the feature vector was
   unjustified). See the port-gap section below: the policy is missing the paper's actual
   mechanism, so the near-constant output is the expected outcome, not a finding about
   weights-varying MPC.
2. **What ships is 13 numbers, not a network.** No torch at inference, no checkpoint, deterministic.
3. **"Diff-WMPC beats hand tuning under DR" must be restated**: it found a fixed vector that hand
   tuning had not. Hand tuning *can* reach it — the search is 13-D and the two axes that matter
   (heading weight, effort ratio) are not the obvious ones.
4. Untouched by any of this: DR tilt stays ~13.8°, still the pitch equilibrium, still statistically
   equal to RL's 14.3°. The honest split is **standoff is won (5.5 cm vs RL's stress figures),
   tilt is a tie bounded by the missing allocation row.**

**Mechanism, pinned from both directions (2026-08-05).** Knock-out (`metrics_hk_*`): the learned
vector with `head_x`/`head_y` alone pulled back to DEFAULT's 20, paired against `fl`:

| | crab | wall err | tilt heave | saturated |
|:--|--:|--:|--:|--:|
| learned vector (`fl`) | 1.77° | 5.49 cm | 13.84° | 39.2% |
| … heading weight only → 20 (`hk`) | **17.72°** | **21.23 cm** | 13.88° | 44.8% |
| paired Δ | +17.94° | +15.74 cm | +0.04° (undecided) | +5.6pp |

Unanimous on all three seeds, and it reproduces the whole factorial-grid failure by changing one
entry. **Tilt does not move** — standoff and tilt are separately determined, which is the cleanest
statement of the DR split: heading weight owns standoff, the missing allocation row owns tilt.

Add-back (`metrics_ha_*`) is only *partial*: the best grid cell with `head` raised 20 → 176
(= 0.44 × radial, the learned ratio) and nothing else changed improves 2 of 3 seeds (10.73 → 9.71,
25.61 → 11.00 cm) but at seed 0 one env of eight diverged to 912 cm — a heading-loss into
across-the-tank sonar returns (~10.6 m in a 12 m tank), the failure `eval_metrics`' outlier note
warns about, here real divergence rather than a metric artifact (no termination fired). Over the
seven envs that did not diverge that seed still improved, 8.03 → 6.97 cm.

So the heading ratio is **necessary but not sufficient**: transplanted into a DEFAULT-scale vector
(`wu = 0.01`, i.e. ~200× tighter effort than learned) a high heading weight is not stable.

**The effort ratio is what stabilises it** (`metrics_haw_*` = `ha` plus `--wu 0.005`, one knob,
nothing else touched). The seed-0 env that diverged goes **912.3 → 7.9 cm**, and every metric
improves unanimously — tilt heave −1.54°, tilt sway −1.28°, crab −7.71°, wall −37.8 cm — paid for
with saturation +13.8pp (30.1% → 43.9%), QP failures still 0.00%.

Stress-DR summary, all 8 envs × 3 seeds, `settled`:

| config | tilt | crab | wall err | sat |
|:--|--:|--:|--:|--:|
| learned vector, frozen (`fl`) | 13.84° | **1.77°** | **5.48 cm** | 39.2% |
| head 0.44 + `wu` 0.005 (`haw`) | **10.89°** | 3.72° | 9.11 cm | 43.9% |
| head 0.44, `wu` 0.01 (`ha`) | 12.43° | 11.43° | 46.95 cm | 30.1% |
| grid best, head 20 (`pf_r400_p2000`) | 12.37° | 12.90° | 14.67 cm | 28.0% |
| learned minus heading weight (`hk`) | 13.88° | 19.72° | 21.23 cm | 44.8% |
| RL policy, published stress DR | 14.30° | — | — | — |

**Three ratios explain the whole DR picture; the other nine weights are free.**

1. `head/radial ≈ 0.44` — owns standoff. Drop it alone and 5.5 cm becomes 21 cm (`hk`).
2. `werr:wu` loose (~200× DEFAULT) — owns stability at that heading weight, and buys tilt and
   crab on top. Its price is saturation.
3. `pitch/radial` — slides along the tilt-vs-standoff frontier. Learned picks 1.06 (standoff end,
   5.48 cm at 13.84°); `haw` runs 5.0 (tilt end, 10.89° at 9.11 cm).

**This revises the "tilt is a tie with RL" line above.** `haw` reaches 10.89° against RL's
published 14.30° — the free pitch equilibrium is an upper bound the controller can be weighted
below, at a standoff cost. Both ends of the frontier are available; one vector cannot sit at both.

### Why our Diff-WMPC learns constant weights: the port drops the paper's mechanism (2026-08-06)

Source: Jahncke et al., *"Differentiable Weights-Varying Nonlinear MPC via Gradient-Based Policy
Learning: An Autonomous Vehicle Guidance Example"*, IEEE RA-L 11(3), March 2026
(`~/Downloads/Differentiable_Weights-Varying_...pdf`). Read it before touching `diff_wmpc.py`.

**The paper's policy input is a look-ahead over the FUTURE REFERENCE, and nothing else** — five
upcoming velocities and five upcoming curvatures over a 2.55 s preview (their Fig. 3). Current
tracking error is not an input at all. The advantage comes from *anticipation*: their Fig. 5 shows
the velocity-error weight rising on straights and falling in corners, and the steering-rate weight
tightening **before** turn-in. Their headline gain — 50% lower mean lateral deviation than the best
static-weight method — is measured **under model mismatch** (Sec. IV-E/F).

Our port inverts both halves:

| | paper | `train_diff_wmpc_wallscan.py` |
|:--|:--|:--|
| policy input | 5-point look-ahead on the reference, no current error | `[e_now(12), sin(phase), cos(phase)]` — all reactive, zero preview (`:325-329`) |
| training condition | includes model mismatch, where the gain is largest | `--task stage3` default = clean tank; `dw_ekf` (`n_updates=3757`) is a nominal run |

So a near-constant weight vector is the *expected* outcome of this port, not evidence about the
method. With only the current phase index the policy can learn a per-phase compromise at best;
with no mismatch in training there is nothing to adapt to.

The wallscan task does have the structure the mechanism needs: `DESCEND → SWAY_A → ASCEND →
SWAY_B` is the straights/corners analogue, and the phase transitions are exactly where anticipation
should pay. **The preview data already exists in the training loop** — `ref[...][:, stage]` is
computed per horizon stage and handed to the solver as `P`; it simply never reaches the policy.

Test order (nominal first, deliberately): saturation is 1.6% at nominal and 39% under DR, and
sensitivity is identically zero at a saturated bound, so a DR-trained learner skips 87% of its
steps. Train at nominal, **probe whether the weights now vary across the phase cycle at all** (the
Fig. 5 analogue — if they do not, the mechanism is still not implemented and a DR evaluation is
meaningless), and only then evaluate under DR against `fl`/`haw`, which are already-measured
best-static baselines of exactly the kind the paper benchmarks against.

### Look-ahead features implemented and measured: the paper's headline does not transfer (2026-08-07)

`--feat {error_phase, preview, both}` (see `diff_wmpc.FEAT_MODES`) ports the paper's policy input.
`preview` is its design — a 5-point look-ahead of `(v_z_des, v_tan_des)`, no current error;
`both` keeps the legacy error term too. Five learned configurations, all trained 40k steps at
nominal on the `dw_ekf` recipe, all evaluated zero-shot under stress DR on the same 3 seeds:

| config | feat | `l_u` | tilt | crab | wall err | sat |
|:--|:--|--:|--:|--:|--:|--:|
| **`fl` frozen legacy vector (best static)** | — | — | 13.84° | 1.77° | **5.48 cm** | 39.2% |
| `dw_ekf` live | error_phase | 0.001 | 13.83° | 2.00° | 5.52 cm | 39.2% |
| `dw_preview` | preview | 0.001 | 13.94° | 4.15° | 5.37 cm | 42.3% |
| `dw_both` | both | 0.001 | **12.71°** | 3.17° | 5.03 cm | 44.4% |
| `dw_preview_lu01` | preview | 0.01 | 13.43° | 4.10° | 6.23 cm | 45.3% |
| `dw_both_lu01` | both | 0.01 | 13.30° | 2.27° | 6.79 cm | 41.9% |
| `dw_both_lu1` | both | 0.1 | 15.19° | **1.22°** | 4.96 cm | **7.1%** |
| `haw` hand-made static | — | — | **10.89°** | 3.72° | 9.11 cm | 43.9% |

The mechanism is verifiably active — `scripts/probe_weight_schedule.py` measures anticipation
(same current leg, different *upcoming* leg) at 22–74% against **structurally zero** for
`error_phase`, and the high-leverage weights now move across phases where before only `w_x` did.

**No learned configuration beats the best static vector on standoff, the primary metric.** The two
that come closest (`dw_both` 5.03 cm, `dw_both_lu1` 4.96 cm) are undecided against `fl`'s 5.48 cm.
What the schedule does buy, unanimously across seeds: `dw_both` −1.13° tilt; `dw_both_lu1` −0.55°
crab and **saturation 39.2% → 7.1%**, a 5.5× cut at equal standoff, which is a real deployment win
(actuator headroom, thermal, wear) even though it is not the paper's claim.

Two hypotheses were tested and one was killed. **Ceiling parking is not the explanation**: raising
`l_u` removed it (`at_ceiling` 0.41 → 0.00) and raised anticipation (54% → 74%), yet standoff got
*worse*, unanimously (`dw_both` 5.03 → `dw_both_lu01` 6.79 cm). What remains is **task fit**: a
racetrack's curvature varies continuously, while this reference is four legs at piecewise-constant
speed, so the preview carries little beyond "a sway leg starts in 0.6 s" — and the solver already
has that, because `reference_preview` hands it the whole ramp over all N stages. The policy is
being shown information the MPC has already used.

**The `l_u=0.1` policy's frozen mean is the best static vector found (`flu1`, 2026-08-07).**
`run_wallscan_mpc.py` now records `mpc.werr_emitted_mean` — the weights a policy actually emits in
closed loop — because freezing a policy whose output swings 90% across phases at one hand-picked
probe point would not represent it. `dw_both_lu1` emits
`radial 2182, z 150, s 618, v_rad 0.8, v_tan 762, v_z 590, head 2123/1879, roll 1685, pitch 165,
w_x 14, w_y 0.1`, `wu 0.0315`. Note what it found on its own: **`pitch` at 165** (it gives up on the
unactuated axis — the move that failed when hand-applied in isolation as `--werr pitch=0.1`, because
the rest of the vector was not re-balanced with it), `head/radial = 0.97`, and an effort penalty 6.3×
`fl`'s.

Frozen and run without the network, paired against the previous best static:

| | tilt | crab | wall err | saturated |
|:--|--:|--:|--:|--:|
| `fl` (previous best static) | **13.84°** | 1.77° | 5.48 cm | 39.2% |
| **`flu1` (frozen `dw_both_lu1` mean)** | 16.10° | **0.68°** | **4.61 cm** | **1.2%** |
| `dw_both_lu1` (live schedule) | 15.19° | 1.22° | 4.96 cm | 7.1% |

`flu1` beats `fl` on standoff, crab and saturation — all unanimous across seeds, saturation by 32× —
and loses only tilt. All three complete 2.00 cycles with zero collisions, so none of this is bought
by doing less work.

**What the schedule itself is worth, finally isolated**: live vs its own frozen mean is +0.91° tilt
(unanimous) paid for with +0.54° crab and 6× the saturation (1.2% → 7.1%, both unanimous), standoff
undecided. So weights-varying does something real here, but it is narrow and it is a trade, not the
paper's 50%-class win.

Practical consequence: **for this task ship a fixed vector.** Diff-WMPC keeps earning its keep as a
**13-D optimizer** — every best-known vector in this table came out of it — and not as a scheduler.

(The `+0.91° tilt` credited to the schedule here was **later shown to be an aggressiveness effect,
not a scheduling one** — see the ceiling section below. `fl`/`flu1`/`haw` are superseded as
deployment candidates by `dr2st`, trained under DR.)

### The ceiling on weights-varying, measured directly (2026-08-08) — the settling result

Four training attempts had produced no measurable benefit from state-dependent weights, which
leaves two incompatible readings: the learner cannot find the gain, or there is no gain to find.
`--pin_dr` / `--pin_spec` collapse every DR range to a single deterministic vehicle, which makes
"the optimal weight vector for THIS vehicle" well defined. Optimising a static vector per vehicle
(`--static`, i.e. Diff-MPC) and comparing it against the shared DR-trained vector gives the
**ceiling**: the most any conditioned policy could win, since a policy can at best identify the
vehicle perfectly and reproduce its optimum.

Five vehicles, `settled`, seed 0 × 8 envs each. Trim is `(cob_x, cog_x)` in opposite signs, which
maximises the trim moment:

| vehicle | \|trim\| | volume | own | shared | **ceiling** | tilt |
|:--|--:|--:|--:|--:|--:|--:|
| B nominal | 0.000 | 1.00 | 4.92 cm | 4.92 | **0.00** | 0.25° |
| T3 | 0.025 | 0.85 | 5.81 | 5.82 | **0.00** | 20.15° |
| T2 | 0.050 | 1.00 | 7.04 | 6.85 | **−0.19** | 34.53° |
| T1 | 0.050 | 1.15 | 5.76 | 14.16 | **8.40** | 31.82° |
| A | 0.050 | 0.85 | 7.42 | 28.90 | **21.48** | 36.10° |

**Tilt ceiling is zero on all five** (−0.001 to −0.010°), across vectors whose `pitch` weight
differs by 26×. Within one vehicle no weighting moves tilt at all, while between vehicles tilt
ranges 0.25–36.10°. The unactuated pitch row is now settled beyond argument: **tilt is a property
of the vehicle, not of the controller.**

**Standoff and crab open only at |trim| max AND volume extreme.** Either alone gives zero: T2 (full
trim, nominal volume) and T3 (half trim, extreme volume) both measure 0. Crab tracks standoff
exactly (ceilings 0.00 / −0.05 / 0.00 / 1.38 / 3.37°), which is the same causal chain the ablation
established — heading error makes the echo sounder over-read, and that IS the standoff error.

**Two corrections to earlier entries in this file, both from over-reading small samples:**

1. *"Buoyancy ±34 N is what breaks the NMPC"* — **wrong**. A 3×3 screen (volume × trim, shared
   vector) has the trim-zero column flat at 4.84–4.99 cm across the whole volume range. Volume
   alone does nothing; it only amplifies trim. The failure is the **CoB/CoG trim offset**.
2. *"The failure region is the trim axis, i.e. half the DR box"* — **wrong**, claimed from the
   screen before the ceilings were measured. The shared vector's error UPPER-BOUNDS the ceiling
   but does not predict it: T2's shared error is 6.85 cm with a ceiling of 0, because that is the
   vehicle's intrinsic difficulty, not a mis-weighting. Only the corner qualifies.

#### Why weights-varying cannot help here — the mechanism

**Cost weights encode which objective to SACRIFICE, and this task rarely forces a sacrifice.** The
deployable vector saturates 0.4% of steps, i.e. it has ~99.6% of its authority spare. Where nothing
must be given up, many weightings reach the same solution — measured directly as a **flat valley**:
T2 and T3's optima put `roll` at 3.0–3.8 against the shared vector's 62.7, a 20× difference, and
closed-loop performance is identical. The ceiling opens only at A and T1, which are exactly the
cells where trim plus extreme volume makes authority tight, and there the optimum is to **abandon
roll** (27.1 → ~3) and spend the heave differential on heading instead.

Three supporting reasons:

* **An MPC is already situation-adaptive.** It re-solves the whole OCP every step against the
  current state and the previewed reference. Weights only add a *preference*, and the preference
  here is situation-independent: always 1.5 m off the wall, always normal to it, always level.
* **The solver already has the paper's input.** `reference_preview` hands the full ramp over all N
  stages to the OCP, so a look-ahead feature shows the policy information the optimiser has already
  consumed. And the reference is four legs at piecewise-constant speed, against a racetrack's
  continuously varying curvature.
* **The loss is myopic** (secondary): it is evaluated at 0.25–1.5 s nodes while a trim offset acts
  as a slow standing offset, so even where headroom exists the gradient barely sees it. Measured:
  the trained policy's response to the plant channels is 1.1×, i.e. it learned to ignore them.

The contrast with the paper is a **problem/method fit**, not an implementation failure: a racecar
at the friction limit must constantly choose what to give up, so choosing well is worth a lot.

#### Deployable result

`dw_dr2_static` — a single vector optimised under DR with the two train/eval mismatches fixed
(observer reset per segment, spawn heading error widened to match evaluation):

```
--werr radial=100.4 --werr z=56.6 --werr s=162.3 --werr v_rad=1.8 --werr v_tan=20.6
--werr v_z=119.2 --werr head_x=29.5 --werr head_y=27.6 --werr roll=62.7
--werr pitch=27.3 --werr w_x=0.2 --werr w_y=0.8 --wu 0.0315
```

Stress DR, 8 envs × 3 seeds: wall 4.96 cm, crab 1.44°, tilt 18.07°, **saturated 0.4%**, 0 failures
in 48 spawns (two independent trainings). Against the RL policy's 31.22 cm that is **6.3×** on
standoff. No network at inference, no observer, no checkpoint.

**Aggressiveness, not conditioning, is the axis that trades tilt against everything else.** `fpl`
(the conditioned policy's own emitted mean, frozen — same q/r, no conditioning) reproduces the
conditioned arm on every metric including its divergence, in the same env, to within noise:

| | conditioning | q/r | tilt | crab | wall | sat | fail/24 |
|:--|:--:|--:|--:|--:|--:|--:|--:|
| `dr2st` | ✗ | 3187 | 18.07° | **1.44°** | **4.96 cm** | **0.4%** | **0** |
| `fpl` | ✗ | 60605 | 16.05° | 6.52° | 25.78 cm | 4.9% | 1 |
| `dr2pl` | ✓ | 60605 | 16.41° | 6.23° | 23.61 cm | 5.6% | 1 |

Raising q/r 19× buys ~2° of tilt and costs 5× the standoff plus a divergence. Conditioning adds
nothing on top. The divergence was diagnosed with `--log_weights` (per-step, per-env weights and
plant features): in the failing env the weights and the plant features both stay inside the range
the healthy envs use — no collapse, no out-of-distribution input — and the vehicle simply sits at
100% saturation from t=0 and never recovers. It is an authority failure at an extreme spawn.

#### Tooling added for this work

`--pin_dr {A,B,C}` / `--pin_spec VOL,COB_X,COG_X` (both scripts) pin the DR draw to one vehicle;
`--static` trains Diff-MPC (`algorithms.diff_wmpc.StaticWeights`, the paper's `theta_k = theta`
branch) and prints a ready-to-use flag string; `--log_weights` stores per-step per-env
`werr_t`/`wu_t`/`plant_t` in the npz; `--plant_feat` appends the observer-derived plant block to the
policy input; `--sym_sway`, `--fixed_wu`, `--spawn_yaw_err`; `--full_3axis` models the DVL and gyro
as the 3-axis instruments they are (the default still passes body `v_z` and the roll/pitch rates
through from ground truth — under-modelling that flatters the vertical axis and partly ground-truth
feeds the observer); `scripts/probe_weight_schedule.py` measures the phase schedule and the
anticipation term offline. Every flag defaults to the legacy behaviour so published numbers keep
their meaning.

**Do not re-open weights-varying for this task without new physics.** The ceiling has been measured
directly: zero for tilt everywhere, zero for standoff except at one corner of the DR box. The
remaining levers are actuation (a pitch arm) and narrowing the DR to the vehicle's measured trim.

### Scan COVERAGE: the axis no metric was watching, and what actually controls it (2026-08-09)

A trajectory plot of a healthy-looking stress-DR run showed the scan drawing **diagonals**: the
vehicle drifted along the wall through legs that are supposed to be vertical. Every existing
metric was fine — wall 4.7 cm, both leg speeds on target, 2.00 cycles — because none of them
looked at whether the planned arc was covered. For a wall INSPECTION that is the mission.

`eval_metrics` gained three metrics, and the reference-free pair is the one to compare on:

| metric | definition | depends on the controller's own plan? |
|:--|:--|:--|
| `s_track_err_cm` (+ rms/p95/max) | \|s_gt − s_ref\| | **yes** |
| **`heave_drift_cm`** | arc displacement inside a complete VERTICAL leg | no — the ideal is 0 for everyone |
| **`sway_step_err_cm`** | \|arc advance over a sway leg − `sway_step`\| | no |

`s_ref` comes from each controller's own state machine, which only advances a phase once the
error is small, so a lagging controller gets a lagging reference and is flattered. During a
vertical leg the plan says hold the arc position, full stop — no loophole.

**Rescoring every run reversed the ranking.** `dr2st`, recommended as the deployment candidate
one section above on the strength of wall 4.96 cm and 0.4% saturation, drifts **1.64 m** per
vertical leg — 8th of 9. `fl`, 4th on standoff, was 1st on coverage. RL is 350 cm, worst by 2×.

#### Coverage is set by TRIM, not by the controller

Correlating tilt against drift over 10 configurations: **r = 0.92**, `drift ≈ 16 cm × tilt°`. The
mechanism is direct — a hull tilted θ tilts its heave thrust vector, so `sin θ` of the vertical
thrust leaks sideways for the whole leg. And tilt is set by the CoB/CoG trim offset, on an axis
with no allocation row.

A trim sweep (same vector, evaluation only) confirms the closed form `θ_eq = asin(trim / z_cb)`
with `z_cb = 0.150 m` at every level:

| trim | predicted pitch | measured tilt | coverage | wall | crab |
|--:|--:|--:|--:|--:|--:|
| ±5.0 cm (`Eval` default) | 19.5° | 18.07° | 164.1 cm | 4.96 | 1.44° |
| ±2.5 cm | 9.6° | 9.60° | 99.3 | 4.58 | 0.70° |
| **±1.0 cm** | 3.8° | **3.97°** | **64.0** | 4.56 | 0.41° |
| ±0.5 cm | 1.9° | 2.03° | 58.2 | 4.60 | 0.34° |
| 0 (pinned) | 0.0° | 0.25° | 55.2 | 4.92 | 0.18° |

**±1.0 cm is the recommended build spec**: below it coverage improves by only 6 cm while assembly
gets much harder. Trim is measurable (float the vehicle, read the rest angle, `trim = z_cb·sin θ`)
and correctable with ballast, so this is a wrench problem, not a control problem. `--dr_trim_cm`
narrows the DR to it; training against ±5 cm teaches the policy to give up on an axis it cannot
move (`dw_dr2_static` emitted `pitch = 27`; retrained at ±1 cm it emits **5.0**).

#### The floor that survives a perfect vehicle is the reference itself

Coverage bottoms out at ~55 cm even at zero trim. Measured cause: **the arc reference moves
55–56 cm sideways during vertical legs, identically at every trim level.** A sway leg may clear up
to `reach_eps` (0.6 m) short, and `s_ramp` keeps slewing toward the stale target into the next
phase. The scan draws diagonals because it was asked to.

This was already diagnosed on 2026-07-26 (`wallscan_env_cfg`: *"every descend STARTED −34 cm off
s_ref and repaid it at 2.3 cm/s mid-descend, 96% same-sign, the whole residual drift"*) and fixed
by `reach_eps = 0.3`, then reverted on 07-27 for an **RL-bootstrap** reason — a fresh policy never
reaches a waypoint at 0.3 and sigma collapses. That reason does not apply to the NMPC.

Two ways to remove the bleed, measured at trim ±1 cm:

| | refMove | coverage | **cycles** | complete legs |
|:--|--:|--:|--:|--:|
| default `reach_eps` 0.6 | 55.3 cm | 64.0 | **1.83** | 22 |
| `reach_eps` 0.3 | 23.9 | 45.3 | **0.46** | 9 |
| `reach_eps` 0.1 | 1.4 | 35.6 | **0.08** | 1 |
| **`ScanCfg.snap_ramp_on_vertical`** | **0.0** | **32.4** | **1.83** | **22** |

Tightening the gate removes the bleed by making the vehicle chase the remainder, and the scan
stops. **Snapping abandons the remainder** — the next vertical line lands 0.9 m over instead of
1.0 m rather than being drawn as a diagonal — and costs nothing. Off by default (`--snap_ramp` on
both scripts); the RL policy trained with the bleed present.

Coverage decomposition, 164.1 cm total: **trim-driven 100 cm** (ballast), **reference bleed 32 cm**
(snap), **residual 32 cm** (open — `sway_step_err` sits at 58 cm across every configuration ever
run, i.e. sway legs stop ~0.58 m short of a 1.0 m step, which `reach_eps = 0.6` calls "arrived").

#### Diff-WMPC's fifth attempt, on a clean environment — it got worse, not better

Four contaminations were removed at once and both arms retrained identically: `--dr_trim_cm 1.0`
(inside the authority envelope), `--snap_ramp` (the reference no longer commands a diagonal),
`--l_s 10` (coverage actually in the objective — it had sat at 1.0 against `l_radial` 2.0), and
`--full_3axis` (the DVL/gyro modelled as 3-axis, so the observer's force channel, which the plant
features are built from, is no longer partly ground truth).

| | coverage | swayErr | wall | crab | tilt | cycles | fail/24 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| previous vector + snap | 32.4 cm | 58.5 | 4.55 | 0.38° | 3.99° | 1.83 | 0 |
| **FAIR static (Diff-MPC)** | 34.2 | 58.4 | **4.53** | **0.33°** | 4.03° | 1.83 | **0** |
| FAIR conditioned (Diff-WMPC) | **23.0** | 47.5 | **89.11** | **16.52°** | 3.43° | 1.75 | 1 |

The conditioned arm wins coverage and tilt and **loses control**: 89 cm of standoff, 16.5° of
crab, saturation 0.4% → 13%, one divergence. Two of three seeds sit at ~120 cm from the wall. Its
coverage number is better because it is no longer doing the mission.

**Why the clean environment made it worse, not better.** The ceiling section above measured where
conditioning has anything to win: nowhere except trim-max × volume-extreme. Narrowing trim to
±1 cm — correct for deployment — **deletes exactly that region**. What is left is a vehicle with
99.6% of its authority spare, where the static optimum drops `pitch` to 5.0 and `roll` to 12.7
because there is no longer a fight to arbitrate. Cost weights choose what to sacrifice; with
nothing to sacrifice, a scheduler has no job, and this one found an aggressive corner of the
weight space instead.

**Raising `l_s` does not reach the residual either** — tested twice, before the environment fixes
(×10, ×40: coverage 164.1 → 165.0 → 165.8) and after (`s` weight 162 → 343: coverage 32.4 → 34.2).
The remaining 32 cm is the sway leg stopping short, which is `reach_eps` versus `sway_step`
geometry, not a cost-weight question.

#### Current best controller

`checkpoints/dw_fair_static` — Diff-MPC, one vector, no network, trained on the corrected
environment:

```
--werr radial=95.6 --werr z=55.8 --werr s=342.8 --werr v_rad=1.8 --werr v_tan=47.8
--werr v_z=120.0 --werr head_x=28.6 --werr head_y=26.9 --werr roll=12.7 --werr pitch=5.0
--werr w_x=0.3 --werr w_y=0.3 --wu 0.0315
```

Run it with `--dr_trim_cm 1.0 --snap_ramp --full_3axis`. Wall 4.53 cm, crab 0.33°, tilt 4.03°,
coverage 34.2 cm, 0.4% saturated, 0 failures in 48 spawns. **The two build/plan fixes are worth far
more than any controller change measured this session**: coverage 164 → 32 cm and tilt 18.07 → 4.03°
came from ballast and a state-machine line, while five Diff-WMPC training attempts produced zero.

### RL vs NMPC under stress DR, the paired numbers (2026-08-06)

RL's stress-DR standoff had never been measured — only its tilt — so the head-to-head had no
number behind it. `play.py --log_traj --eval_steps 18500 --score_episode 1`, 8 envs × 3 seeds,
`settled` (rescored at `settle_s=20` through `rescore_trajectories.py`, since `play.py` writes no
settled block of its own). The checkpoint was trained on the **shipped** plant while every NMPC
number is on **fixed + z_slender**, so `play.py` gained `--tam`/`--hydro` (defaulting to the legacy
classes) and RL is reported on both — one arm alone would rig the comparison:

| | tilt | crab | wall err | collided /24 |
|:--|--:|--:|--:|--:|
| RL, native plant (`rlDR`) | 13.74° | 2.80° | 31.22 cm | 1 |
| RL, NMPC plant (`rlDRfx`, off-distribution) | 12.46° | 2.39° | 29.82 cm | 2 |
| NMPC learned vector (`fl`) | 13.84° | **1.77°** | **5.48 cm** | **0** |
| NMPC tilt-end vector (`haw`) | **10.89°** | 3.72° | 9.11 cm | **0** |

13.74° reproduces the published 14.3°, which validates the setup. Then:

- **Standoff is the decisive win: 5.48 vs 31.22 cm, 5.7×**, and it does not depend on which plant
  RL runs on (29.8–31.2 cm either way).
- **`haw` dominates RL on both target metrics at once** — tilt 10.89° vs 13.74°, standoff 9.11 vs
  31.22 cm, collisions 0 vs 1 — losing only crab (3.72° vs 2.80°). `fl` trades that tilt margin
  for another 3.6 cm of standoff. So "RL and NMPC tie under DR" is **wrong**; it was an artifact
  of comparing tilt alone, on a weight vector that had not been tuned for standoff.
- RL's crab is genuinely good (2.4–2.8°) — it is not the RL policy's weak axis. Standoff is.

Caveats kept visible: 3 of the 6 RL runs scored a truncated episode (159–161 s of 180 s) because
an env collided, and RL scored 18–24 complete heave legs against NMPC's 24. Rate metrics are
unaffected; `cycles_mean` (1.50–2.00) is not comparable to the NMPC's 2.00.

Tooling added for this: `--werr NAME=VALUE` on `run_wallscan_mpc.py` (repeatable, applied after
`--w_roll` so it can split that flag's roll/pitch pair — that coupling is why `--w_roll` could
never have answered this) and `--wu` (the effort weight; needed because `werr` is capped at 5000,
so the learned ~200× effort ratio is not reachable with `--werr` alone),
`scripts/compare_paired_runs.py` (paired per-seed verdicts plus a
config-drift check that fails an A/B differing in anything but weights; validated by reproducing
the observer result above), and the `roll_deg`/`pitch_deg` npz channels. Those channels are
deliberately **not** in `TrajectoryLog.FIELDS` — that tuple is a contract `play.py` must also
satisfy — so the runner merges them in after `as_arrays`.

Remaining known defects, in the order the measurements rank them: **pitch has no allocation row**
(above — still true of the TAM, but 2026-08-05 demotes it: the weight vector moves both pitch and
standoff more than the missing row does, so cost tuning outranks actuation as the next lever); `s` drifts 60–91 cm under DR because UKF-M updates only `(r, φ)` and never
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
