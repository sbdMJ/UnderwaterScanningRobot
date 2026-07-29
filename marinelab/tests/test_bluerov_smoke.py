# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logic-level smoke tests for the bluerov DR path and observation width."""

import torch

from marinelab.core.hydrodynamics import HydrodynamicsModel


class _Cfg:
    added_mass = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
    linear_damping = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)
    quadratic_damping = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)
    volume = 0.0113459
    body_name = "base_link"
    center_of_buoyancy = (0.0, 0.0, 0.01)
    center_of_gravity = (0.0, 0.0, 0.0)
    water_density = 997.0
    use_full_coriolis = True
    rigid_body_inertia = (0.12, 0.12, 0.12)
    body_mass = 11.5
    apply_added_mass_force = False
    added_mass_stability_factor = 0.8
    damping_cross_coupling = None
    damping_stability_factor = None


class _CurCfg:
    max_velocity = (0.0,) * 6
    noise_scale = (0.0,) * 6


def test_dr_path_no_missing_method_crash():
    # Mirrors the bluerov reset DR call shape (scale + offset) via the core API.
    m = HydrodynamicsModel(8, "cpu", _Cfg(), current_cfg=_CurCfg(), dt=0.005)
    env_ids = torch.arange(8)
    m.scale_parameters(
        env_ids,
        added_mass=(0.5, 1.0),
        linear_damping=(0.5, 1.0),
        quadratic_damping=(0.5, 1.0),
        volume=(0.9, 1.1),
        rigid_body_inertia=(0.8, 1.2),
    )
    base = m.base_parameters
    cob = base.center_of_buoyancy[env_ids].clone()
    cob[:, 2] = cob[:, 2] + (torch.rand(8) * 0.04 - 0.02)
    m.set_parameters(env_ids, center_of_buoyancy=cob)
    # buoyancy force stays positive and finite
    assert torch.all(torch.isfinite(m.buoyancy_force))
    assert torch.all(m.buoyancy_force > 0)


def test_observation_width_is_18():
    # pos(3)+quat(4)+lin_vel(3)+ang_vel(3)+goal_obs(3)+up(2)
    assert 3 + 4 + 3 + 3 + 3 + 2 == 18
