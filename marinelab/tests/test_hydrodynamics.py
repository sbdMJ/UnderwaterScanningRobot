# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.hydrodynamics (Isaac-Sim-free)."""

import torch

from marinelab.core.hydrodynamics import HydrodynamicsModel
from marinelab.core.ocean_current import OceanCurrent


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


class _CurrentCfg:
    max_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    noise_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _model(n=4):
    return HydrodynamicsModel(num_envs=n, device="cpu", cfg=_Cfg(), current_cfg=_CurrentCfg(), dt=0.005)


def test_base_parameters_immutable_snapshot():
    m = _model()
    base = m.base_parameters
    # base reflects cfg
    assert torch.allclose(base.linear_damping[0], torch.tensor(_Cfg.linear_damping))
    # mutating live params must not change base_parameters
    m.set_parameters(torch.tensor([0]), linear_damping=torch.zeros(1, 6))
    assert torch.allclose(m.base_parameters.linear_damping[0], torch.tensor(_Cfg.linear_damping))


def test_get_parameters_reads_current_values():
    m = _model()
    p = m.get_parameters()
    assert torch.allclose(p.volume, torch.full((4,), 0.0113459))
    assert p.linear_damping.shape == (4, 6)


def test_set_parameters_absolute_write():
    m = _model()
    m.set_parameters(torch.tensor([1, 3]), volume=torch.tensor([0.02, 0.02]))
    v = m.get_parameters().volume
    assert abs(v[1].item() - 0.02) < 1e-6 and abs(v[3].item() - 0.02) < 1e-6
    assert abs(v[0].item() - 0.0113459) < 1e-6  # untouched


def test_set_parameters_updates_buoyancy_force():
    m = _model()
    before = m.buoyancy_force[0].item()
    m.set_parameters(torch.tensor([0]), volume=torch.tensor([0.02]))
    after = m.buoyancy_force[0].item()
    assert after > before  # rho*g*V increased with volume


def test_scale_parameters_multiplies_base():
    torch.manual_seed(0)
    m = _model(64)
    m.scale_parameters(torch.arange(64), linear_damping=(0.5, 0.5))
    ld = m.get_parameters().linear_damping
    expected = torch.tensor(_Cfg.linear_damping) * 0.5
    assert torch.allclose(ld[0], expected, atol=1e-5)


def test_injected_shared_current():
    cur = OceanCurrent(4, "cpu", _CurrentCfg())
    m = HydrodynamicsModel(4, "cpu", _Cfg(), current=cur, dt=0.005)
    assert m.current is cur


def test_buoyancy_neutral_magnitude():
    m = _model()
    # F_b = rho * g * V ~ 110.97 N for V=0.0113459
    fb = m.buoyancy_force[0].item()
    assert abs(fb - 997.0 * 9.81 * 0.0113459) < 1e-3


def test_set_ocean_current_delegates_to_component():
    m = _model()
    m.set_ocean_current(torch.tensor([0]), velocity=torch.ones(1, 6))
    assert torch.all(m.current.velocity_w[0] == 1.0)


def test_coriolis_added_mass_sign_known_value():
    # C_A(v) v for diagonal added mass, hand-computed reference.
    m = _model(1)
    lin = torch.tensor([[1.0, 0.5, -0.5]])
    ang = torch.tensor([[0.1, -0.2, 0.3]])
    vel6 = torch.cat([lin, ang], dim=-1)
    force, torque = m._compute_coriolis_added_mass(vel6)
    # a1 = M_A[:3] * lin = [5.5, 6.35, -7.285]; force = -(a1 x ang)
    a1 = torch.tensor([5.5 * 1.0, 12.7 * 0.5, 14.57 * -0.5])
    expected_force = -torch.cross(a1, ang[0], dim=-1)
    assert torch.allclose(force[0], expected_force, atol=1e-4)


def test_damping_quadratic_form():
    m = _model(1)
    vel = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d = m._compute_damping(vel)
    # surge: D_l*v + D_q*|v|*v = 4.03*2 + 18.18*2*2 = 8.06 + 72.72
    assert abs(d[0, 0].item() - (4.03 * 2 + 18.18 * 4)) < 1e-3


def test_buoyancy_restoring_zero_when_upright_and_cob_centered():
    # CoB at (0,0,0.01), upright: moment = r_cb x F_b. F_b along +z body => x,y moment only.
    m = _model(1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity
    wrench = m._compute_buoyancy_quat(quat)
    # r_cb=(0,0,0.01) x F=(0,0,Fb) => cross is zero (parallel)
    assert torch.allclose(wrench[0, 3:], torch.zeros(3), atol=1e-5)


def test_scale_parameters_accepts_per_env_tensor():
    # Per-env scalar scale tensor: base * scale, uniform across the field's trailing dims.
    m = _model(n=4)
    base_am = m.base_parameters.added_mass.clone()
    base_vol = m.base_parameters.volume.clone()
    ids = torch.tensor([0, 2])
    scales = torch.tensor([1.5, 0.5])
    m.scale_parameters(ids, added_mass=scales, volume=scales)
    am_diag = torch.diagonal(m._added_mass_matrix, dim1=-2, dim2=-1)
    assert torch.allclose(am_diag[0], base_am[0] * 1.5)
    assert torch.allclose(am_diag[2], base_am[2] * 0.5)
    assert torch.allclose(am_diag[1], base_am[1])          # untouched env
    assert torch.allclose(m._volume[ids], base_vol[ids] * scales)


def test_scale_parameters_tensor_volume_refreshes_buoyancy():
    # volume via set_parameters must refresh buoyancy force (existing auto-refresh path).
    m = _model(n=2)
    f0 = m._buoyancy_force_base.clone()
    m.scale_parameters(torch.tensor([0]), volume=torch.tensor([2.0]))
    assert torch.allclose(m._buoyancy_force_base[0], f0[0] * 2.0)
    assert torch.allclose(m._buoyancy_force_base[1], f0[1])


def test_scale_parameters_tuple_path_unchanged():
    # Regression: (lo, hi) tuple still samples uniform per env within bounds.
    m = _model(n=64)
    base_ld = m.base_parameters.linear_damping.clone()
    m.scale_parameters(torch.arange(64), linear_damping=(0.5, 1.5))
    ratio = m._linear_damping_diag / base_ld
    assert ratio.min() >= 0.5 - 1e-6 and ratio.max() <= 1.5 + 1e-6
    assert ratio.std() > 0.01  # actually random, not constant
