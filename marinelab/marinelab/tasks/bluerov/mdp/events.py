# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for BlueROV domain randomization.

``randomize_hydrodynamics`` randomizes the hydrodynamics model parameters
(added mass, damping, volume, inertia, and CoB/CoG offsets) for the specified
environments. It is called directly from BlueROVEnv._reset_idx on every reset
when domain randomization is enabled, using the core setter-based API
(``HydrodynamicsModel.scale_parameters`` / ``set_parameters``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..bluerov_env import BlueROVEnv


def randomize_hydrodynamics(
    env: BlueROVEnv,
    env_ids: torch.Tensor | None,
    added_mass_scale: tuple[float, float] = (0.8, 1.2),
    linear_damping_scale: tuple[float, float] = (0.8, 1.2),
    quadratic_damping_scale: tuple[float, float] = (0.8, 1.2),
    volume_scale: tuple[float, float] = (0.9, 1.1),
    cob_offset_x: tuple[float, float] = (0.0, 0.0),
    cob_offset_y: tuple[float, float] = (0.0, 0.0),
    cob_offset_z: tuple[float, float] = (0.0, 0.0),
    cog_offset_x: tuple[float, float] = (0.0, 0.0),
    cog_offset_y: tuple[float, float] = (0.0, 0.0),
    cog_offset_z: tuple[float, float] = (0.0, 0.0),
    inertia_scale: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Randomize hydrodynamic parameters for specified environments.

    Args:
        env: The BlueROV environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        added_mass_scale: Scale range for added mass coefficients.
        linear_damping_scale: Scale range for linear damping coefficients.
        quadratic_damping_scale: Scale range for quadratic damping coefficients.
        volume_scale: Scale range for vehicle volume (affects buoyancy).
        cob_offset_x: Offset range for CoB X in meters.
        cob_offset_y: Offset range for CoB Y in meters.
        cob_offset_z: Offset range for CoB Z in meters.
        cog_offset_x: Offset range for CoG X in meters.
        cog_offset_y: Offset range for CoG Y in meters.
        cog_offset_z: Offset range for CoG Z in meters.
        inertia_scale: Scale range for rigid body inertia.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "_hydro"):
        return

    hydro = env._hydro
    num_envs = len(env_ids)
    device = env.device

    # Scale-based randomization via the core API (base * uniform(lo, hi)).
    hydro.scale_parameters(
        env_ids,
        added_mass=added_mass_scale,
        linear_damping=linear_damping_scale,
        quadratic_damping=quadratic_damping_scale,
        volume=volume_scale,
        rigid_body_inertia=inertia_scale,
    )

    # Offset-based randomization for CoB / CoG (base + uniform offset).
    base = hydro.base_parameters

    def _offset(center, ranges):
        out = center[env_ids].clone()
        for axis, (lo, hi) in enumerate(ranges):
            out[:, axis] = out[:, axis] + (torch.rand(num_envs, device=device) * (hi - lo) + lo)
        return out

    hydro.set_parameters(
        env_ids,
        center_of_buoyancy=_offset(base.center_of_buoyancy, (cob_offset_x, cob_offset_y, cob_offset_z)),
        center_of_gravity=_offset(base.center_of_gravity, (cog_offset_x, cog_offset_y, cog_offset_z)),
    )
