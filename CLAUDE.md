# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠ sim-to-real 작업 시작 전 필수 (one-shot 지침)

**이 세션에서 sim-to-real / E5 / 수조·현장 실험 / Jetson / mj_ws 관련 작업을
처음 시작한다면, 코드·문서를 만지기 전에
`docs/experiments/sim-to-real/NEXT_SESSION.md`를 반드시 먼저 읽어라.**
전체 여정, 완료(DONE) 표기, 미완성 상태의 정확한 스냅샷과 다음 할 일이 거기 있다.
읽고 후속 작업에 착수한 직후에는 그 문서 안의 자기-삭제 규약에 따라 **그 파일과
이 절을 함께 삭제·커밋**할 것 (불필요한 반복 열람 방지). 파일이 이미 없다면 이
절도 낡은 것이니 같이 지워라.

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

## Long-running work: always register a monitoring cron job

**Whenever a task is expected to run ≥30 minutes** (training, tuning, experiment sweeps,
image builds), register a self-monitoring cron job (CronCreate, ~30 min interval) at
launch time — do not rely on ad-hoc checks or a warn-only Monitor. The job's prompt must:

1. Compare progress against a recorded baseline (artifact/metrics file counts, trials.csv
   rows — **not** stdout, which is block-buffered under redirection and goes quiet),
2. On stall + CPU spin, diagnose (py-spy via a `--pid=container:<name>` +
   `--cap-add SYS_PTRACE` sidecar) and restart the stalled work automatically,
3. Scope the job to the **whole remaining pipeline**, not the current step — delete it
   only when everything is done (learned 2026-08-08: a loop scoped to "E2 done" was
   correctly deleted at E2 completion, leaving the 5-hour retune that followed unwatched).

Session cron jobs are in-memory only — after a session restart, re-register before
resuming the work.

**Tuning/optimization runs additionally require a mid-run spot check.** Do not wait for
the full budget to finish before validating: partway through (e.g. ~half the trials),
snapshot the current best candidate from `trials.csv` and evaluate it on the HELD-OUT
protocol (the evaluation seeds/conditions the tuning objective never sees), via a
separate config whose `exp:` writes to its own results dir (e.g. `e2_interim/`) so real
experiment artifacts are never polluted. If the candidate already shows the failure mode
(e.g. per-seed blowups), stop and fix the protocol instead of spending the remaining
budget. Rationale: attempts 1-2 of BO tuning burned their full budget on a single-seed
objective and were only caught afterwards by E1/E2 (see `docs/experiments/tuning_history/bo_tuning_history.md`).

**Judge spot checks and result analyses against the parent paper's MOBO-WMPC pattern**
(Diff-WMPC, RA-L 2026, Jahncke et al. — see `docs/experiments/tuning_history/bo_tuning_history.md` §0): a tuned
static baseline must (1) match or beat Fixed-W nominal in-distribution, (2) degrade
gracefully (not collapse) under perturbation, (3) lose only to the adaptive methods.
Fixed-W nominal is the boundary between "zero-shot structural limit" and "tuning
artifact" — losing to nominal means the tuning is broken, not that zero-shot is hopeless.

## Commands

### Tests (no Isaac Sim needed — run natively, fast)

`marinelab/tests/conftest.py` stubs `isaaclab` (with a real quaternion implementation) and
shims the `marinelab` package so its heavy `__init__` never runs. Only torch is required:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/mjkim/.conda/envs/acmpc_sim/bin/python \
  -m pytest marinelab/tests/ -q                          # 111 tests
# single file / single test
... -m pytest marinelab/tests/pkrc_wallscan/test_geometry.py -q
... -m pytest marinelab/tests/test_eval_metrics.py -q -k crab
```

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

### acados / NMPC (Diff-WMPC port, in progress)

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
- **`run_experiment.py` multi-cell in one process is broken at HEAD** (2026-08-19): the
  second `build_env` dies with `A prim already exists: /World/envs/env_0/Tank`, any method
  (repro: `isaaclab/logs/probe_teardown.yaml`). Campaign SOP has always been one
  invocation per cell (`--cond`/`--seed`) × ≤4 parallel containers. Also `isaaclab.sh -p`
  can return exit 0 on a Python traceback — judge success by metrics file counts only.
- `train_and_push.sh` (train → commit checkpoint → push) has paths hardcoded to
  `/root/home/rl_ws/...`, which is **not** this host's container layout
  (`/workspace/UnderwaterScanningRobot`). Fix the paths before using it here.
- Note `README.md`'s "84 tests" is stale — the suite is 111 after `test_eval_metrics.py`.

## Reference docs in-repo

- `marinelab/scripts/experiments/README.md` — competitor-comparison experiments (E1–E4):
  runner/tuning/aggregation/figure pipeline, results layout (`experimental_results/`)
- `docs/experiments/competitor_framework_plan.md` — the experiment framework's design decisions
  (controller layer, tuning protocol, figure plan F1–F10, SSI-MPC GPL isolation)
- `docker/README.md` — this host's runtime, GUI, reproduction commands
- `marinelab/docs/wallscan-training-code-guide.md` — code walkthrough in execution order (Korean)
- `marinelab/docs/wallscan-project-report.md` — results write-up
- `marinelab/docs/architecture.md`, `marinelab/docs/installation.md`, `marinelab/CONTRIBUTING.md`
- W&B training curves: https://wandb.ai/yju1121-postech/pkrc_wallscan
