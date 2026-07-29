# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Hydrodynamic parameter bundle and base-parameter cache.

HydroParams is a named bundle of per-env parameter tensors used by both the
read API (get_parameters, for privileged observations) and the write API
(set_parameters, for domain randomization). default_rigid_inertia centralizes
the added_mass[3:6]*0.5 fallback that was previously duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HydroParams:
    """Named bundle of per-env hydrodynamic parameter tensors.

    All fields are optional: a None field means "not provided / leave unchanged"
    on the write path, and is simply absent on the read path if not requested.
    """

    added_mass: torch.Tensor | None = None  # (N, 6)
    linear_damping: torch.Tensor | None = None  # (N, 6)
    quadratic_damping: torch.Tensor | None = None  # (N, 6)
    volume: torch.Tensor | None = None  # (N,)
    water_density: torch.Tensor | None = None  # (N,)
    center_of_buoyancy: torch.Tensor | None = None  # (N, 3)
    center_of_gravity: torch.Tensor | None = None  # (N, 3)
    rigid_body_inertia: torch.Tensor | None = None  # (N, 3)
    body_mass: torch.Tensor | None = None  # (N,)


def default_rigid_inertia(cfg) -> list[float]:
    """Resolve rigid-body inertia, falling back to added_mass[3:6] * 0.5.

    Single source of the heuristic that was duplicated across the model init,
    stability validation, and randomization code.

    Args:
        cfg: A config with `added_mass` (len-6) and optional `rigid_body_inertia`.

    Returns:
        Three-element inertia diagonal [Ixx, Iyy, Izz].
    """
    if getattr(cfg, "rigid_body_inertia", None) is not None:
        return list(cfg.rigid_body_inertia)
    return [x * 0.5 for x in cfg.added_mass[3:6]]
