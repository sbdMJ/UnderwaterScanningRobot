# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Experiment history

**Searchable index:** [`docs/hero/experiments_index.json`](docs/hero/experiments_index.json) — 49 runs + 17 settled
decisions, queryable with `jq` (e.g. `jq '.experiments[] | select(.verdict=="BASELINE")'`). Each entry's
`source` field points to the full narrative in the archived changelogs below.

Archived changelogs (full narrative, preserved in `docs/hero/archive/`):

- [R9 → Student v1](docs/hero/archive/changelog_r9_to_student_v1.md) (2026-04-18 ~ 2026-04-22)
- [Full ALBC early development](docs/hero/archive/changelog_full_albc_early.md) (2026-03-31 ~ 2026-04-02)
- [Constrained ALBC development](docs/hero/archive/changelog_constrained_albc.md) (2026-03-27 ~ 2026-03-31, deprecated project)
- [Legacy development](docs/hero/archive/changelog_legacy.md) (2026-03-05 ~ 2026-03-26)

Per-round detail (Rounds 1-8, encoder ablation) lives in `docs/hero/experiments/`.

---

<!-- Active entries go below. Previous session log archived to
     docs/hero/changelog_r9_to_student_v1.md on 2026-04-22. -->

## [2026-05-25] isaaclab pristine-restore: strip scripts/ contamination, overlay owns train entry

### Context
A FullDOF smoke run after the 3-split failed at runner creation
(`ModuleNotFoundError: isaaclab_tasks.direct.constrained_full_albc`). Audit found
isaaclab `scripts/` still carried our code: train.py + play.py held a hardcoded
`_RUNNER_MAP` pointing at the defunct `constrained_full_albc` namespace, plus 9
hero/full_dof demo scripts, a play_student.py, and an accidental `import gymnasium`
deletion in `direct/__init__.py`. The split had left isaaclab non-pristine at the
scripts layer.

### Experiments / failures
- **Restored scripts/ from `upstream/main` first — wrong baseline.** Our isaaclab_rl
  is an OLD fork (fork point `cbf51abb`); upstream/main's train.py imports
  `handle_deprecated_rsl_rl_cfg`, a symbol our fork lacks. Result: even stock
  `Isaac-Cartpole-v0` died at import with `ImportError: cannot import name
  'handle_deprecated_rsl_rl_cfg'`. (User caught this by running it directly.)
- **Fix: restore from the fork point `cbf51abb`, not current upstream/main.** That tree
  is contemporaneous with our isaaclab_rl and carries zero of our code. Verified: stock
  Isaac-Cartpole trains via main train.py (0.48s); FullDOF trains via the overlay
  launcher (`Using overlay runner ConstraintEncoderRunner` → iteration 0→1, 2.94s);
  both exit 0.

### Decisions
- **"pristine" = our-code-free AND compatible with our forked isaaclab_rl** — NOT
  "byte-identical to today's upstream/main". Restore baseline is the merge-base
  `cbf51abb` (`git checkout cbf51abb -- scripts/...`). isaaclab_rl (`weight_decay`)
  and the forked rsl_rl package stay an INTENTIONAL fork (architecture.md). Resolves
  the conflict between `feedback-isaaclab-pristine` and the split design.
- **Overlay owns its train entry**, not a runpy delegation. Upstream train.py hardcodes
  only OnPolicyRunner/DistillationRunner; our two custom runners
  (ConstraintEncoderRunner, OnPolicyDoraemonRunner) can't share one alias branch. So
  `constrained-albc/scripts/train.py` replicates main() but owns its `_RUNNER_MAP` + a
  one-shot import hook for gym.register. Rejected runpy delegation (cannot dispatch two
  custom runners).
- **Replicate against our forked isaaclab_rl, not upstream/main** — same `handle_deprecated`
  lesson applied to the launcher: reconcile imports against
  `HEAD:source/isaaclab_rl/.../rsl_rl/__init__.py`.
- **Play entry deferred (YAGNI).** play.py restored; FullDOF play not re-created —
  `eval_dr.py static` is the canonical Full-DOF eval tool and self-registers.

### Open Questions
- The 108-file `source/` diff vs upstream/main is mostly upstream's own post-fork changes,
  not ours; a future rebase onto a newer upstream tag would reconcile them. Not urgent.
- Design/plan: `constrained-albc/docs/superpowers/specs/2026-05-25-isaaclab-pristine-restore-design.md`,
  `.../plans/2026-05-25-isaaclab-pristine-restore.md`.

## [2026-05-25] Env repo 3-split: isaaclab / marinelab / constrained-albc

### Context
The single isaaclab tree mixed three concerns: upstream framework, a public
underwater-environment layer (bluerov + UUV assets + shared marine physics),
and private research (constrained ALBC + TDC + student). Split into three
purpose-scoped repos with an external-extension overlay architecture so the
isaaclab fork stays a zero-touch clean fork and juniors get a clean public
deploy path. Design: `docs/superpowers/specs/2026-05-25-repo-3split-design.md`.
Plan: `docs/superpowers/plans/2026-05-25-repo-3split.md`.

### What moved
- **marinelab** (`/workspace/marinelab`, public overlay): `models/` →
  `marinelab/physics/` (Fossen hydro + thruster), uuv assets → `marinelab/assets/`
  (Git LFS, 18 meshes), bluerov env → `marinelab/tasks/bluerov/`,
  `isaaclab/utils/volume.py` → `marinelab/utils/`. Self-registers 5 Isaac-BlueROV-*.
- **constrained-albc** (`/workspace/constrained-albc`, private, history preserved
  via git filter-repo, 128 commits): constrained_full_albc + _tdc + student +
  `scripts/analysis/` → `constrained_albc/analysis/`. Self-registers 6 Isaac-FullDOF-*.
  Depends on marinelab. cli_args.py vendored into analysis/ (was a broken relative path).
- **isaaclab**: reverted 3 registration __init__ edits + volume export → clean fork.

### Verification
All 11 envs (5 BlueROV + 6 FullDOF) re-register from the overlays inside a
launched app, entry_points reading `marinelab.tasks.bluerov` /
`constrained_albc.envs.*`. isaaclab imports cleanly with 0 UUV/albc envs leaking.

### Decisions
- **External-extension overlay over thin-patch fork.** marinelab/constrained-albc
  self-register Gym envs via their own pyproject entry-points, so isaaclab core needs
  zero registration edits — chosen so upstream rebase stays conflict-free (HORA/RMA-style
  clean-fork convention). Rejected: keeping the registration __init__ edits as a documented
  patch (would re-conflict on every upstream pull).
- **Per-repo git history strategy:** constrained-albc preserved via `git filter-repo`
  (research blame value high); marinelab clean-start (junior-facing deploy).
- **Committed the 11 in-progress research files (5bb7bcd8fb) before re-extracting.** A first
  filter-repo extract took isaaclab's COMMITTED tree, but the 11 research edits were
  uncommitted working-tree → committed config.py passed `k_bias=` with no matching rewards.py →
  import TypeError. Committing made committed==working so the re-extract captured the latest code.
  Rationale: user confirmed working-tree is source of truth ("only the latest r13a/student code matters").
- **Deleted r13a_hist5/10/act3/layernorm + feat/constrained-norbc branches (not migrated).**
  Confirmed 2026-04-21 intermediate ablation snapshots: `diff main..r13a` is all-insertions (those
  albc files absent from main since Phase 3 moved them), r13_A already baseline-locked in
  experiments_index.json, final code lives in constrained-albc. isaaclab must stay a pure simulator
  fork, so albc-research branches don't belong there. Not backed up elsewhere, but final code is safe.

### Lessons
- **git-lfs install has repo-wide side effects.** Installing git-lfs (for marinelab meshes) activated
  isaaclab's pre-existing upstream LFS rules (`*.dae/*.obj/*.pt`). Old isaaclab main had 9 uuv meshes
  committed as plain blobs ("should have been pointers"), blocking a branch fast-forward. Fix:
  `GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= ... reset --hard <target>` (meshes deleted there anyway).
- **`git clone` of a multi-branch repo drags ALL branches.** constrained-albc (cloned from isaaclab for
  filter-repo) inherited 9 isaaclab branches incl. a stale pre-migration `main`. Cleaned to single main:
  deleted stale main + renamed work branch + dropped 7 unrelated branches.
- Pre-split staging: the 4 student launchers were grouped under `scripts/student/` (d0c54a74) then their
  internal paths re-fixed post-split (train_student.py → constrained-albc/scripts/, eval → analysis/,
  sibling calls via BASH_SOURCE dir).

### Notes
- **Retained fork couplings** (NOT removed): `isaaclab_rl` cfg fields
  (`state_dependent_std`, `weight_decay`) and the forked `rsl_rl` package —
  constrained-albc depends on both at runtime (documented in its docs/architecture.md).
- 1 pre-existing isaaclab test bug (`test_update_buoyancy_force_subset`) carried over,
  not migration-induced.
- All three repos are now on a single `main` branch, local-only (not pushed).

### Open Questions / next steps
- Push all 3 repos (create GitHub repos; constrained-albc visibility=private). origin still has
  isaaclab's old feat/encoder-tdc-integration + main remote branches — decide their fate on push.
- Fresh-machine deploy must provide the forked rsl_rl + isaaclab_rl cfg edits, else constrained-albc fails.
- marinelab/constrained-albc have no git remotes yet.

## [2026-04-22] Dead code purge after r13_A baseline lock

### Context
Baseline locked to r13_A on 2026-04-22 (Phase 0.7 decision, challenger Enc16
disqualified). The repo still carried three layers of experimental code that
were no longer reachable from any active config: (1) the disqualified
challenger task, (2) deprecated sibling task dirs `hero_agent/` and
`constrained_albc/` kept for historical imports, and (3) pre-Full-DOF analysis
scripts. Objective: shrink the reachable surface to just r13_A + the four
live ablation variants (v2 NoEncoder, v3 TRPO-NoIPO, v4 PPO-Enc, v5 PurePPO)
while Round 1 training was still running on both GPUs.

### Experiments
- **Import surface survey**: grep for live `from isaaclab_tasks.direct.hero_agent`
  and `from isaaclab_tasks.direct.constrained_albc` found 6 live importers beyond
  the registry: `scripts/analysis/common.py` (hero_agent DR + encoder cfg),
  `scripts/analysis/{eval_dr,collect_rollouts}.py` (constrained_albc runners/env),
  `scripts/demos/test_hero_thruster.py`, and the `constrained_full_albc_tdc`
  classical baseline (imported TDC + kinematics from `hero_agent/controllers/`).
  Additional hidden surface: dynamic runner dispatch maps in
  `scripts/reinforcement_learning/rsl_rl/{train,play}.py` referenced 5 dead
  runner classes (`BaseRunner`, `EncoderRunner`, `AdaptRunner`,
  `ConstraintEncoderRunner`, `SACMPCRunner`). These never fired on current
  configs (all use `FullDOFConstraintEncoderRunner` or `OnPolicyRunner`) so
  deletion was safe.
- **TDC baseline disambiguation**: user initially said "remove the TDC dir if
  it is RL-based TDC". Reading `constrained_full_albc_tdc/__init__.py:6` and
  `controllers/thruster_pd.py` showed it is TDC (arm) + stateless PD (thruster)
  classical control, not RL. User reclassified on inspection and asked to keep
  it. Evidence checked for TDC+PID alternative elsewhere in repo: none exists
  — only TDC+P(D) in this single dir. No run logs for `Isaac-FullDOF-TDC-v0`
  ever produced.
- **Challenger audit**: `FullDOFTRPOChallengerEnc16RunnerCfg` +
  `ALBCChallengerEnc16EnvCfg` + `Isaac-FullDOF-TRPO-ChallengerEnc16-v0` were
  still wired through `agents/__init__.py` and would have shown up as a
  gym-registered task for anyone pulling the repo. The 14M
  `challenger_hist5_act3_enc16.log` was still at repo-adjacent `/workspace/`
  after the run dir already preserved it.
- **Checkpoint-fallback path already existed**: `encoder_z_sweep.py` had both
  a hero_agent-backed `build_nominal_obs()`/`build_sweep_params()` path and a
  `build_sweep_params_from_checkpoint()` fallback guarded by a `try/except
  ImportError`. For any Full-DOF checkpoint (24D privileged) the hero_agent
  path returned a 19D array, triggered the dim-mismatch branch, and fell
  through to the checkpoint path anyway — i.e. the hero_agent branch was
  dead for every current use case. Collapsing to the checkpoint-only path
  removed the last reason to keep the hero_agent DR/encoder cfg imports in
  `common.py`.

### Decisions
- **Moved TDC controllers into `constrained_full_albc_tdc/controllers/`** (not
  left in a shrunken hero_agent dir) because the TDC baseline is the only
  consumer of `tdc.py` and `kinematics.py`. This makes the classical baseline
  self-contained and unblocks full deletion of `hero_agent/`. Rejected
  alternative: keep a pared-down `hero_agent/controllers/` package — would
  leave a cross-package import and the deprecated dir in the gym registry.
- **Deleted challenger task and code outright** rather than marking as
  deprecated. Rationale: the Phase 0.7 decision explicitly disqualified it
  (pitch regression +208%, yaw +125% at hard DR, 1-env catastrophic outlier
  with `att_lv=+0.976` coupling). Keeping the code invited re-running it.
  Still preserved: `logs/rsl_rl/.../challenger_hist5_act3_enc16/` run dir
  (git-ignored logs, contains the evaluated checkpoint).
- **Kept running ablation code untouched** (v2 NoEncoder, v3 TRPO-NoIPO, v4
  PPO-Enc, v5 PurePPO). Round 1 training was live at the time of the purge
  (v2 and v5 at iter 500/2500 at completion, ETA ~80 min). Editing runner
  configs mid-run risks Round 2 launch failure when the orchestrator
  re-imports the module. Deferred this tier to post-training.
- **Left dynamic runner maps with only `FullDOFConstraintEncoderRunner` plus
  the `OnPolicyRunner`/`DistillationRunner` elif branches**. No current cfg
  uses any other `class_name`.
- **Did not delete `hero_agent_hydro_demo.py`, `analyze_hero_mass.py`,
  `check_usd_*.py`**. They reference `HeroAgentBuoyHydrodynamicsCfg` / USD
  structure from `isaaclab_assets.robots.uuv`, not from the deprecated task
  dir. Asset package is still load-bearing for r13_A (robot cfg,
  hydrodynamics constants).

### Lessons
- **Deprecation without deletion grows tentacles.** "Deprecated" dirs
  accumulated live imports in three different systems (classical baseline,
  analysis scripts, dynamic runner dispatch). Grep-based audit caught them
  but the `try/except ImportError` in `common.py` hid one branch from
  grep-for-imports — only grep-for-function-names revealed it.
- **Dynamic runner dispatch maps are a blind spot.** A class_name -> module
  path dict does not trigger `ImportError` at import time; the missing
  module would only fail at runtime when a specific class_name is requested.
  These stale entries had been undiagnosed dead code since the pre-Full-DOF
  era.
- **Verify classification before acting on user directives.** User asked to
  "remove the TDC+PID baseline if it's RL-based". Reading the actual
  controller math (`thruster_pd.py` line 142: "P on attitude error, D on body
  angular rate") revealed it is P(D), not PID, not RL. Literal execution of
  the instruction would have deleted the wrong thing. Reading the code
  before executing the delete caught it.

### Open Questions
- `docs/hero/plans/2026-03-*.md` still reference now-deleted modules
  (`hero_agent.encoder.HistoryTCN`, `constrained_albc.algorithms.ConstraintTRPO`,
  etc.). Not actionable: these are historical specs superseded by live
  implementations in `constrained_full_albc/`. Defer to legacy-changelog
  archival when Tier D (ablation code) is purged.
- Ablation code (v2/v3/v4/v5) must be removed after Round 2 + cross-variant
  analysis completes. Budget: ~8 hours from training start, bounded by
  2500-iter runs at ~2.0-2.6s/iter on RTX 4070/4060.

