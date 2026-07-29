# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest fixtures: run marinelab.core tests without Isaac Sim.

The generic mock-everything approach breaks physics tests because
quat_apply_inverse must actually rotate vectors. This conftest installs a
correct quaternion implementation while stubbing the rest of isaaclab.
"""

import sys
import types
from pathlib import Path

import torch


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vec by the inverse of quat (w, x, y, z). Matches isaaclab.utils.math."""
    q = quat
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # inverse rotation = conjugate for unit quat: (w, -x, -y, -z)
    # v' = v + 2 * q_conj_vec x (q_conj_vec x v + w_conj * v), w_conj = w
    qvec = torch.stack([-x, -y, -z], dim=-1)
    uv = torch.cross(qvec, vec, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vec + 2.0 * (w.unsqueeze(-1) * uv + uuv)


def _quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _configclass(cls):
    """Minimal stand-in for isaaclab.utils.configclass.

    The real one auto-generates __init__ from annotated class attributes with
    defaults. Our cfg classes only use simple typed defaults, so dataclass
    semantics suffice for tests.
    """
    import dataclasses

    return dataclasses.dataclass(cls)


def _install_isaaclab_mocks():
    if "isaaclab" in sys.modules:
        return
    for name in ["isaaclab", "isaaclab.utils", "isaaclab.utils.math"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["isaaclab.utils"].configclass = _configclass
    math_mod = sys.modules["isaaclab.utils.math"]
    math_mod.quat_apply_inverse = _quat_apply_inverse
    math_mod.quat_from_euler_xyz = _quat_from_euler_xyz


def _install_marinelab_package_shim():
    """Pre-register a bare ``marinelab`` package so its __init__ never runs.

    ``marinelab/__init__.py`` does ``from . import tasks`` which imports the
    bluerov env -> ``isaaclab.sim`` and the full Isaac Sim stack. We only want
    ``marinelab.core.*`` (pure physics) under test. Registering ``marinelab`` as
    an empty module with a real ``__path__`` lets normal imports resolve
    ``marinelab.core.parameters`` from disk while skipping the heavy __init__.
    ``marinelab.assets`` (cfg classes the core imports) is stubbed minimally.

    ``marinelab.tasks`` gets the same treatment: its real ``__init__.py`` also
    imports the bluerov/pkrc envs. Stubbing it (with a real ``__path__``) lets
    Isaac-Sim-free subpackages like ``marinelab.tasks.pkrc_wallscan`` (pure
    torch) import normally while skipping the heavy env registration.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if "marinelab" not in sys.modules:
        pkg = types.ModuleType("marinelab")
        pkg.__path__ = [str(repo_root / "marinelab")]
        sys.modules["marinelab"] = pkg
    if "marinelab.tasks" not in sys.modules:
        tasks_pkg = types.ModuleType("marinelab.tasks")
        tasks_pkg.__path__ = [str(repo_root / "marinelab" / "tasks")]
        sys.modules["marinelab.tasks"] = tasks_pkg
    if "marinelab.assets" not in sys.modules:
        assets = types.ModuleType("marinelab.assets")
        assets.HydrodynamicsCfg = type("HydrodynamicsCfg", (), {})
        assets.OceanCurrentCfg = type(
            "OceanCurrentCfg",
            (),
            {"max_velocity": (0.0,) * 6, "noise_scale": (0.0,) * 6},
        )
        assets.ThrusterCfg = type("ThrusterCfg", (), {})
        sys.modules["marinelab.assets"] = assets


_install_isaaclab_mocks()
_install_marinelab_package_shim()
