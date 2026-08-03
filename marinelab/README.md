<div align="center">

# marinelab

**A general-purpose underwater-vehicle (UUV) environment framework for NVIDIA Isaac Lab.**

![version](https://img.shields.io/badge/version-0.2.0-blue)
![license](https://img.shields.io/badge/license-BSD--3--Clause-green)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![isaac-sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76b900)

[Documentation](docs/) | [Getting Started](#getting-started) | [Contributing](CONTRIBUTING.md)

</div>

## Overview

marinelab separates a reusable marine-physics core from the example environments built
on it. Researchers import and configure the core models — hydrodynamics, thrusters,
ocean current — through a stable public API, and build their own Isaac Lab tasks on top,
the same way marinelab itself builds on Isaac Lab. The BlueROV2 task suite is the
in-repo reference consumer; downstream research overlays are the intended users.

## Key Features

- **Fossen hydrodynamics core**: 6-DOF added mass, linear/quadratic damping, and buoyancy/restoring forces with per-environment parameter buffers.
- **First-order thruster model**: thruster lag dynamics plus an allocation matrix, advanced with the env step so configured time constants hold.
- **Injectable ocean current**: one current source shared across hydrodynamics models, with sampling, drift, and explicit set paths.
- **Setter-based domain randomization**: `set_parameters` / `scale_parameters` always randomize from an immutable base snapshot — no compounding.
- **Reference BlueROV2 tasks**: five hover/attitude environments registered into the Isaac Lab Gym registry, showing the full wiring pattern.

## Getting Started

### Prerequisites

- Docker with the NVIDIA runtime, or an existing editable Isaac Lab install
- Git LFS (`apt-get install git-lfs` if absent) — USD meshes are LFS-tracked
- Python 3.10+ (the Isaac Sim interpreter)

### Installation

> **No conda, no venv.** Isaac Sim ships its own Python interpreter and marinelab
> installs into that one — do not create an environment for this repo. If a conda
> or virtual env is active, `isaaclab.sh -p` switches to that interpreter instead,
> and the install lands where Isaac Sim never looks. Deactivate any environment
> first — see [docs/installation.md](docs/installation.md#do-not-create-a-conda-or-venv-environment).

```bash
git clone https://github.com/luckkim123/marinelab.git /workspace/marinelab
cd /workspace/marinelab && git lfs install && git lfs pull

cd /workspace/isaaclab
./isaaclab.sh -p -m pip install -e /workspace/marinelab
```

`isaaclab` and `isaaclab_assets` are **not** pip dependencies — they are provided by
the Isaac Lab editable install already present in the environment.
See [docs/installation.md](docs/installation.md) for the Docker deploy path.

### Verify

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-BlueROV-Hover-Direct-v0 --num_envs 4 --headless
```

> `import marinelab` outside a launched Isaac Sim app raises `No module named pxr`
> (the USD runtime is needed transitively). Always run through `./isaaclab.sh -p`
> or the Docker container.

## Quick Start

Build a current, inject it into a hydrodynamics model, step the physics, and
randomize through the public API:

```python
import torch
from marinelab.core import HydrodynamicsModel, OceanCurrent
from marinelab.assets import BlueROVHydrodynamicsCfg, OceanCurrentCfg

num_envs, device = 4096, "cuda"

current = OceanCurrent(num_envs, device, OceanCurrentCfg())
hydro = HydrodynamicsModel(num_envs, device, BlueROVHydrodynamicsCfg(),
                           dt=0.005, current=current)

# Per-step: body-frame hydrodynamic wrench (includes the injected current).
forces, torques = hydro.compute_forces(root_lin_vel_w, root_ang_vel_w, root_quat_w)

# Per-reset domain randomization — all through the public API.
env_ids = torch.arange(num_envs, device=device)
hydro.scale_parameters(env_ids, added_mass=(0.8, 1.2), linear_damping=(0.8, 1.2))
```

`marinelab.tasks.bluerov.bluerov_env.BlueROVEnv` is the minimal end-to-end example of
this pattern inside a `DirectRLEnv`.

## Core API

All core models are importable from `marinelab.core`
(`marinelab.physics` was removed — update old imports).

<details>
<summary><strong>API reference</strong></summary>

### `HydrodynamicsModel`

Fossen 6-DOF rigid-body hydrodynamics: added mass, linear and quadratic damping,
and buoyancy/restoring forces. Holds per-environment parameter buffers.

```python
HydrodynamicsModel(num_envs, device, cfg, current_cfg=None, dt=0.01,
                   articulation_prim_path=None, current=None)
```

- `compute_forces(root_lin_vel_w, root_ang_vel_w, root_quat_w)` — body-frame force
  and torque for the current state (includes the injected ocean current).
- `update_physx_state(body_com_acc_w, root_quat_w)` — feed PhysX acceleration for
  the added-mass force path (only when `apply_added_mass`).
- `get_parameters(env_ids=None)` — read the current parameters as a `HydroParams`.
- `set_parameters(env_ids, **fields)` — absolute writes; updates buoyancy
  automatically when volume or water density changes.
- `scale_parameters(env_ids, **ranges)` — convenience: `base * uniform(lo, hi)`.
- `update_buoyancy_force(env_ids=None)` — recompute the buoyancy magnitude from
  volume and density.
- `reset(env_ids=None)` — restore parameters to base and reset the current.
- Properties: `base_parameters`, `current`, `buoyancy_force`, `volume`,
  `added_mass_matrix`, `linear_damping`, `quadratic_damping`,
  `center_of_buoyancy`, `center_of_gravity`, `water_density`, `body_mass`,
  `rigid_body_inertia`, `apply_added_mass`.

### `ThrusterModel`

First-order thruster dynamics with an allocation matrix.

```python
ThrusterModel(cfg, num_envs, device, enable_randomization=False)
```

- `apply_dynamics(commands, dt)` — advance the per-thruster first-order state.
  Call once per env step with the elapsed `step_dt` (not the raw physics dt).
- `compute_wrench()` — `(forces, torques)` in the body frame.
- `randomize_parameters(env_ids, thrust_coeff_scale, time_constant_scale)`.
- `reset(env_ids)`.
- Properties: `state`, `body_forces`, `body_torques`.

### `OceanCurrent`

An injectable current source shared by one or more hydrodynamics models.

```python
OceanCurrent(num_envs, device, cfg)
```

- `set(env_ids, velocity=None, strength=None)` — set explicitly, or sample within
  `max_velocity` when `velocity` is omitted.
- `add_drift(delta)` — additive increment to the current velocity.
- `reset(env_ids=None)`.
- Properties: `velocity_w` (6-D world-frame current), `max_velocity`.

### `HydroParams`

A dataclass bundle of the per-environment hydrodynamic parameters returned by
`get_parameters` / `base_parameters`.

### Domain randomization

Two entry points, both writing per-environment parameter buffers:

- **`set_parameters(env_ids, **fields)`** — absolute writes. You compute the value
  (for example, base CoB plus a sampled offset) and assign it.
- **`scale_parameters(env_ids, **ranges)`** — convenience for the common case:
  each field becomes `base * uniform(lo, hi)`.

Both read from an immutable snapshot of the base parameters captured at construction
(`base_parameters`), so repeated randomization always scales from the original
configuration rather than compounding. Changing volume or water density recomputes
the buoyancy force automatically.

</details>

## Registered Environments

| Task ID | Description |
|:--------|:------------|
| `Isaac-BlueROV-Hover-Direct-v0` | Hover control — demo / random agent |
| `Isaac-BlueROV-Hover-Train-Direct-v0` | Hover control — training variant |
| `Isaac-BlueROV-Hover-Eval-Direct-v0` | Hover control — evaluation variant |
| `Isaac-BlueROV-Attitude-Direct-v0` | Attitude control — demo / random agent |
| `Isaac-BlueROV-Attitude-Train-Direct-v0` | Attitude control — training variant |

## Project Structure

```
marinelab/
├── marinelab/
│   ├── core/        # reusable physics — the framework API (stable)
│   ├── assets/      # UUV configs + USD meshes (Git LFS)
│   ├── tasks/       # example environments built on core (bluerov is the reference)
│   ├── algorithms/  # DORAEMON domain-randomization engine
│   └── utils/
└── docs/            # architecture + installation
```

`marinelab` depends only on `isaaclab`; there are no upward dependencies. `core/`
knows nothing about any specific vehicle or task — see
[docs/architecture.md](docs/architecture.md) for the layering rationale.

## Contributing

Contributions that add new UUV vehicles or tasks on top of `core/` are welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md). Changelog: [CHANGELOG.md](CHANGELOG.md).

## License

BSD-3-Clause — see [LICENSE](LICENSE).
