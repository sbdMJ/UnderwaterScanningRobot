# Changelog

All notable changes to marinelab are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Changed

- README restructured to standard OSS layout (hero/badges/features; core API reference
  in a collapsible section); `docs/architecture.md` un-staled (P5 docs pass, 2026-07-12).
- `.gitignore` covers session/tool state dirs (`.claude/`, `.ruff_cache/`, `.omx/`,
  `.trash/`).

### Fixed

- Thruster first-order lag ran at half speed (P4 sim-fix, 2026-07-12): `bluerov_env.py`
  passed `physics_dt` to `ThrusterModel.apply_dynamics` from `_pre_physics_step`, which runs
  once per env step -- elapsed time between calls is `step_dt = physics_dt * decimation`
  (decimation=2 here), so effective T200 time constants were 2x the configured values. Now
  passes `step_dt`. Same root cause fixed in constrained-albc (4x there, decimation=4).
- `update_physx_state` caller in `bluerov_env.py` passed the full body tensor, silently
  relying on "root == body index 0" inside the `(num_envs, num_bodies, 6)` branch; it now
  passes `body_com_acc_w[:, self._body_id[0], :]` explicitly. Byte-identical while the hydro
  body is index 0.
- DORAEMON `--resume` device-mismatch crash (P3 hygiene, 2026-07-12):
  `DoraemonScheduler.load_state_dict` now restores `dist._a`/`dist._b` with
  `.double().cpu()`. A checkpoint loaded with a GPU map_location previously left them on
  CUDA while `_mins`/`_maxs`/`_ranges` stayed CPU (the class invariant), crashing
  `entropy()`/`get_flat_params()` on the first step after resume.

### Changed

- ALBC asset: `velocity_limit_sim` 6.28 -> 3.1 rad/s (P4 sim-fix, 2026-07-12; measured
  XW540-T260 no-load plateau, onboard 2026-07-06). Paired with the constrained-albc
  `arm_joint_vel` soft-threshold move 4.189 -> 2.8 rad/s in the same batch to keep the soft
  constraint inside the hard cap (else it can never fire -- dead constraint).
- Added `[tool.ruff]`/`[tool.pyright]` to pyproject.toml (line-length 120, py310, Google
  docstrings, marinelab first-party isort) so the format hook is no longer a no-op here;
  applied the safe autofix subset (isort ordering, one unused import, return-style).

### Removed

- Deprecated `marinelab.physics` shim (pure re-export of `marinelab.core`). All in-repo
  and downstream (constrained-albc) importers migrated to `marinelab.core`;
  CONTRIBUTING.md/README.md guidance updated.

### Added

- DORAEMON curriculum record & replay (`marinelab.algorithms.doraemon`):
  - `DoraemonScheduler` records its DR curriculum trajectory (one `(iter, a, b)`
    entry per update) and exposes `export_recording()` (param metadata +
    trajectory) and `export_trajectory()`. `step()` gains an optional `iteration`
    argument so the trajectory aligns to true RL iterations.
  - `CurriculumReplayer` duck-types the scheduler's
    `sample`/`step`/`record_episodes`/`state_dict` surface and replays a recorded
    curriculum with hold-last interpolation; the online optimizer is disabled.
    Used to give comparison baselines the cmdp run's exact DR difficulty timeline.
  - `DoraemonCfg.replay_curriculum_path` config field selects replay mode.
  - Tests: `tests/test_doraemon_replay.py`.

---

## [0.2.0] - 2026-05-25

Framework-ization: marinelab becomes a general-purpose UUV framework. The physics
moves into a reusable `marinelab.core` package with a setter-based domain-
randomization API and an injectable ocean-current component, while `bluerov`
remains as the reference example. `marinelab.physics` and `marinelab.utils.volume`
become deprecated re-export shims, so existing imports (including constrained-albc)
keep working unchanged.

### Added

- `marinelab.core` framework package:
  - `HydrodynamicsModel` with a setter-based DR API: `get_parameters`,
    `set_parameters` (absolute writes, auto-recompute buoyancy on volume/density
    change), `scale_parameters` (base * uniform(lo, hi)), and an immutable
    `base_parameters` snapshot captured at construction.
  - `OceanCurrent` — an injectable current component shared across hydrodynamics
    models via dependency injection.
  - `HydroParams` dataclass bundle and `default_rigid_inertia` helper unifying the
    inertia fallback.
  - `ThrusterModel` / `ThrusterCfg` and the volume utilities, relocated here.
- Isaac-Sim-free pytest suite (`tests/`, 27 tests): parameters, ocean_current,
  hydrodynamics (incl. known-value physics regression), thruster, attitude error,
  and a bluerov DR/obs smoke test. Runs via `tests/conftest.py` with real quat math.

### Changed

- Physics relocated from `marinelab.physics.{hydrodynamics,thruster}` and
  `marinelab.utils.volume` into `marinelab.core`. The old modules are now
  deprecated re-export shims (behavior identical; verified by the known-value
  regression tests).
- BlueROV reset domain randomization routes through the core `scale_parameters` /
  `set_parameters` API.
- Hot-path cleanup in `BlueROVEnv`: the world up-vector is cached once instead of
  allocated per observation, and the initial-pose DR is vectorized (6 scalar
  samples to 2 batched samples; identical ranges).
- README rewritten as a framework guide (core API reference + researcher workflow
  + DR semantics + migration note).

### Removed

- ~340 lines of dead code: the unused `EventCfg` and three orphaned event
  functions (`randomize_thruster_params`, `randomize_ocean_current`,
  `randomize_robot_pose`), `TaskBase.step`, `RewardManager.get_term_value` /
  `set_term_weight`, the linear `orientation_upright` reward (the live reward uses
  `orientation_upright_exp`), and the env-level legacy `goal_pos_range` /
  `initial_height` fields (the task config owns these).

### Fixed

- **A1** — BlueROV `enable_gyroscopic_forces` set to `False`: with
  `use_full_coriolis=True` the model already computes `C_RB` internally, so leaving
  PhysX gyroscopic forces on double-counted rigid-body gyroscopic effects.
  (hero_agent is unchanged: it uses `use_full_coriolis=False`, so PhysX correctly
  owns `C_RB` there.)
- **B1** — BlueROV reset called a nonexistent `HydrodynamicsModel.randomize_parameters`;
  DR now goes through the core scale/set API.
- **B2** — attitude `observation_space` corrected from 12 to 18 to match the
  environment's fixed observation layout (pos 3 + quat 4 + lin_vel 3 + ang_vel 3 +
  goal 3 + up 2).
- **B4** — attitude error is sign-corrected by the quaternion scalar part so the
  signal stays continuous across the quaternion double cover near 180 degrees.
- **B6** — the volume fallback warning now reports the resolved fallback magnitude.
- Broken reference: `_debug_vis_callback` read `self._hydro._current_velocity`, a
  buffer removed when the ocean current moved into the `OceanCurrent` component; it
  now reads `self._hydro.current.velocity_w`.

### Verification

- `pytest tests/` — 27 passed.
- `ast.parse` over the whole package — all parse OK.
- Import-integrity sweep — no leftover references to the moved modules outside the
  shims; constrained-albc imports only `HydrodynamicsModel` / `ThrusterModel` from
  `marinelab.physics`, both preserved by the shim.

### Notes

- **A2** (BlueROV vertical-thruster pitch/roll allocation): resolved as comment-only,
  matrix unchanged. Evidence (`references/hero_agent/.../bluerov/user.py` holds the
  Pitch/Roll RC channels at neutral) shows the standard BlueROV frame does not
  independently actuate pitch with two y-separated verticals; the pitch row is
  effectively a heave row. No numeric change, so no dynamics change.
- The **A1** fix changes BlueROV dynamics; existing BlueROV policies should be
  retrained.
- Migrating constrained-albc onto `marinelab.core` (replacing shim imports and
  adopting the setter-based DR API) is a separate, later coordinated phase. This
  release leaves constrained-albc untouched and importable via the shims.

## [0.1.0] - 2026-05-25

### Added

- Initial extraction from the isaaclab monorepo.
- BlueROV2 environments (Hover, Attitude) with Direct task interface — 5 registered Gym IDs.
- UUV assets: BlueROV2 and Hero Agent USD meshes, tracked with Git LFS.
- Shared marine physics: Fossen hydrodynamics model (`physics/hydrodynamics.py`) and
  thruster dynamics (`physics/thruster.py`), migrated from `isaaclab_tasks/models/`.
- Volume utility: buoyancy volume calculator (`utils/volume.py`), migrated from
  `isaaclab/utils/volume.py`.
- `pyproject.toml` with `isaaclab.extensions` entry-point for self-registration.
- Docker deploy files: `docker/Dockerfile` and `docker/docker-compose.yaml`.
- Documentation: `README.md`, `docs/installation.md`, `docs/architecture.md`,
  `CONTRIBUTING.md`.
- BSD-3-Clause `LICENSE` (inherited from upstream Isaac Lab).
