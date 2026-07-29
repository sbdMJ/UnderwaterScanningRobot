# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.ocean_current."""

import torch

from marinelab.core.ocean_current import OceanCurrent


class _Cfg:
    max_velocity = (0.5, 0.5, 0.25, 0.0, 0.0, 0.0)
    noise_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_init_zero_velocity():
    oc = OceanCurrent(num_envs=4, device="cpu", cfg=_Cfg())
    assert oc.velocity_w.shape == (4, 6)
    assert torch.all(oc.velocity_w == 0.0)
    assert oc.max_velocity.shape == (6,)


def test_set_explicit_velocity():
    oc = OceanCurrent(4, "cpu", _Cfg())
    vel = torch.ones(2, 6)
    oc.set(torch.tensor([0, 2]), velocity=vel)
    assert torch.all(oc.velocity_w[0] == 1.0)
    assert torch.all(oc.velocity_w[1] == 0.0)  # untouched
    assert torch.all(oc.velocity_w[2] == 1.0)


def test_set_sampled_within_bounds():
    torch.manual_seed(0)
    oc = OceanCurrent(64, "cpu", _Cfg())
    oc.set(torch.arange(64))
    maxv = oc.max_velocity
    assert torch.all(oc.velocity_w.abs() <= maxv + 1e-6)


def test_set_strength_scales_sample():
    torch.manual_seed(0)
    oc = OceanCurrent(64, "cpu", _Cfg())
    oc.set(torch.arange(64), strength=torch.zeros(64))
    assert torch.allclose(oc.velocity_w, torch.zeros(64, 6))


def test_add_drift_and_reset():
    oc = OceanCurrent(4, "cpu", _Cfg())
    oc.add_drift(torch.full((4, 6), 0.1))
    assert torch.allclose(oc.velocity_w, torch.full((4, 6), 0.1))
    oc.reset(torch.tensor([0, 1]))
    assert torch.all(oc.velocity_w[0] == 0.0)
    assert torch.allclose(oc.velocity_w[2], torch.full((6,), 0.1))
    oc.reset()  # all
    assert torch.all(oc.velocity_w == 0.0)
