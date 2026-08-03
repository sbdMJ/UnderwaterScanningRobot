# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.parameters."""

import torch

from marinelab.core.parameters import HydroParams, default_rigid_inertia


def test_default_rigid_inertia_uses_added_mass_fallback():
    # When rigid_body_inertia is None, fallback = added_mass[3:6] * 0.5
    class Cfg:
        added_mass = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
        rigid_body_inertia = None

    assert default_rigid_inertia(Cfg()) == [0.06, 0.06, 0.06]


def test_default_rigid_inertia_prefers_explicit():
    class Cfg:
        added_mass = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
        rigid_body_inertia = (0.2, 0.3, 0.4)

    assert default_rigid_inertia(Cfg()) == [0.2, 0.3, 0.4]


def test_hydroparams_holds_optional_tensor_fields():
    p = HydroParams(volume=torch.ones(4))
    assert p.volume.shape == (4,)
    assert p.added_mass is None  # unset fields default to None
