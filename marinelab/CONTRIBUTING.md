# Contributing to marinelab

This guide explains how to add new underwater-vehicle tasks to marinelab and how
to keep the codebase consistent before opening a pull request.

---

## Adding a new UUV task

### 1. Create a task directory

Add `marinelab/tasks/<your_vehicle>/` with these files:

| File | Purpose |
|:---|:---|
| `__init__.py` | `gym.register` calls |
| `<vehicle>_env.py` | `DirectRLEnv` subclass (Isaac Lab's direct-workflow RL environment base class) |
| `<vehicle>_env_cfg.py` | Environment config (`@configclass`, Isaac Lab's config decorator) |
| `agents/` | RL agent configs (rsl_rl, skrl, sb3, ...) |
| `mdp/` | Events, rewards, observations |

### 2. Register environments in `__init__.py`

Each variant (demo, train, eval) must be registered individually with
`gymnasium.register`. Use the `Isaac-<Vehicle>-<Task>-Direct-v0` naming
convention to stay consistent with the existing environments:

```python
import gymnasium as gym

gym.register(
    id="Isaac-MyVehicle-Hover-Direct-v0",
    entry_point="marinelab.tasks.myvehicle:MyVehicleEnv",
    kwargs={"cfg": MyVehicleHoverEnvCfg()},
)
```

### 3. Chain the new package from `tasks/__init__.py`

Add an import line to `marinelab/tasks/__init__.py` so the registrations load
when marinelab is imported:

```python
from . import myvehicle  # noqa: F401
```

### 4. Add assets (if any)

Place USD/DAE/OBJ meshes in `marinelab/assets/<your_vehicle>/meshes/`.
Track binary assets with Git LFS:

```bash
git lfs track "marinelab/assets/<your_vehicle>/meshes/*.usd"
git lfs track "marinelab/assets/<your_vehicle>/meshes/*.dae"
git add .gitattributes
```

### 5. Reuse shared physics

`marinelab.core` implements the Fossen model (`HydrodynamicsModel`) and thruster
dynamics (`ThrusterModel`) that all UUV environments share. Import them rather
than duplicating the physics logic in your task.

---

## Code style

marinelab follows the same conventions as Isaac Lab:

- Formatter and linter: **ruff** (line length 120, Google-style docstrings).
- Type checker: **pyright** (false-positive `reportCallIssue` on `@configclass`
  constructors is expected; suppress with `# type: ignore[call-arg]` if needed).

Run before committing:

```bash
ruff check marinelab/
ruff format marinelab/
```

---

## Verify before opening a PR

Run the random agent on one of the existing environments to confirm the
environment stack loads correctly end-to-end:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 \
    --num_envs 4 --headless
```

Then do the same for your new environment ID. A run that exits cleanly (no
import errors, no shape mismatches) is the minimum bar for a PR.

---

## Pull request checklist

- [ ] New task directory created under `marinelab/tasks/`.
- [ ] Gym IDs follow the `Isaac-<Vehicle>-<Task>-Direct-v0` convention.
- [ ] New package chained in `marinelab/tasks/__init__.py`.
- [ ] Binary assets tracked with Git LFS (`.gitattributes` updated).
- [ ] `ruff check` and `ruff format` pass.
- [ ] `random_agent.py` runs cleanly with the new task ID.
- [ ] `CHANGELOG.md` updated with a brief entry under `[Unreleased]`.
