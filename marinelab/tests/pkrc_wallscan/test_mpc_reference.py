# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Tests for the wallscan NMPC reference layer (mpc_reference.py).

These are sign-convention and closed-form tests. They exist because the whole Diff-WMPC
port hinges on getting the inside-out mapping right (the orbit task faces the cylinder,
wallscan faces away from it), and a flipped sign there would produce a controller that
drives INTO the wall while its cost function reports success.
"""

import math

import pytest
import torch

from marinelab.tasks.pkrc_wallscan import mpc_reference as mr
from marinelab.tasks.pkrc_wallscan import scan_state_machine as ssm

CFG = mr.WallScanMPCCfg()


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    z = torch.sin(half)
    w = torch.cos(half)
    return torch.stack([w, torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)


def _state(pos, quat, v_b=None, w_b=None):
    n = pos.shape[0]
    v_b = torch.zeros(n, 3) if v_b is None else v_b
    w_b = torch.zeros(n, 3) if w_b is None else w_b
    return torch.cat([pos, quat, v_b, w_b], dim=-1)


# ---------------------------------------------------------------------------
# Sonar geometry
# ---------------------------------------------------------------------------


def test_sonar_range_reduces_to_clearance_when_perpendicular():
    r = torch.tensor([0.0, 2.0, 4.5, 5.9])
    t = mr.sonar_range(r, torch.zeros_like(r), CFG.tank_radius)
    assert torch.allclose(t, CFG.tank_radius - r, atol=1e-6)


def test_sonar_range_is_one_sided_and_matches_hand_computed_bias():
    """Oblique beams ALWAYS over-read; the bias is what makes crab-walk dangerous."""
    r = torch.full((4,), 4.5)  # clearance 1.5 m
    phi = torch.deg2rad(torch.tensor([10.0, 20.0, 30.0, 45.0]))
    bias = mr.clearance_bias(r, phi, CFG.tank_radius)
    assert (bias > 0).all(), "an oblique beam must never read SHORT of the true clearance"
    # exact values of -r*cos(phi) + sqrt(R^2 - r^2 sin^2 phi) - (R - r) at r=4.5, R=6
    expected = torch.tensor([0.01726, 0.07062, 0.16504, 0.40477])
    assert torch.allclose(bias, expected, atol=2e-4)


def test_clearance_bias_is_second_order_in_phi():
    """Doubling phi roughly quadruples the bias -> a tracking reward barely feels it."""
    r = torch.tensor([4.5])
    b1 = mr.clearance_bias(r, torch.deg2rad(torch.tensor([5.0])), CFG.tank_radius)
    b2 = mr.clearance_bias(r, torch.deg2rad(torch.tensor([10.0])), CFG.tank_radius)
    assert 3.5 < float(b2 / b1) < 4.5


def test_heading_offset_from_range_inverts_sonar_range():
    r = torch.tensor([3.0, 4.5, 4.5, 5.0])
    phi = torch.deg2rad(torch.tensor([5.0, 12.0, 40.0, 25.0]))
    t = mr.sonar_range(r, phi, CFG.tank_radius)
    assert torch.allclose(mr.heading_offset_from_range(t, r, CFG.tank_radius), phi, atol=1e-5)


def test_heading_offset_from_range_is_zero_at_true_clearance():
    r = torch.tensor([4.5])
    t = torch.tensor([1.5])
    assert float(mr.heading_offset_from_range(t, r, CFG.tank_radius)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Arc length
# ---------------------------------------------------------------------------


def test_arc_length_zero_at_anchor_and_scales_with_tank_radius():
    theta0 = torch.tensor([0.7])
    pos_xy = torch.stack([4.5 * torch.cos(theta0), 4.5 * torch.sin(theta0)], dim=-1)
    s0 = torch.tensor([3.0])
    assert torch.allclose(mr.arc_length(pos_xy, theta0, s0, CFG.tank_radius), s0, atol=1e-6)

    # a 0.1 rad step measures 0.1 * tank_radius (the WALL swath), not 0.1 * vehicle radius
    theta1 = theta0 + 0.1
    pos1 = torch.stack([4.5 * torch.cos(theta1), 4.5 * torch.sin(theta1)], dim=-1)
    got = mr.arc_length(pos1, theta0, s0, CFG.tank_radius) - s0
    assert torch.allclose(got, torch.tensor([0.1 * CFG.tank_radius]), atol=1e-5)


def test_arc_length_survives_the_antipode_within_a_horizon():
    """Anchoring per-solve is what keeps s continuous where a wrapped projection jumps."""
    theta0 = torch.tensor([math.pi - 0.02])
    theta1 = torch.tensor([-math.pi + 0.02])  # 0.04 rad forward, across the wrap
    p1 = torch.stack([4.5 * torch.cos(theta1), 4.5 * torch.sin(theta1)], dim=-1)
    ds = mr.arc_length(p1, theta0, torch.zeros(1), CFG.tank_radius)
    assert torch.allclose(ds, torch.tensor([0.04 * CFG.tank_radius]), atol=1e-5)


# ---------------------------------------------------------------------------
# Error vector signs -- the part a flipped convention would silently break
# ---------------------------------------------------------------------------


def _errors(pos, yaw, **kw):
    n = pos.shape[0]
    x = _state(pos, _quat_from_yaw(torch.as_tensor(yaw).reshape(n)), kw.pop("v_b", None), kw.pop("w_b", None))
    theta = torch.atan2(pos[:, 1], pos[:, 0])
    defaults = dict(
        z_ref=pos[:, 2].clone(), s_ref=torch.zeros(n), v_tan_des=torch.zeros(n), v_z_des=torch.zeros(n),
        theta_anchor=theta, s_anchor=torch.zeros(n), cfg=CFG,
    )
    defaults.update(kw)
    return mr.wallscan_errors(x, **defaults)


def test_zero_error_when_on_station_and_facing_the_wall():
    """+x axis pointing radially OUTWARD is the zero-heading-error configuration."""
    theta = torch.tensor([0.0, 1.1, -2.4, 3.0])
    pos = torch.stack([CFG.r_des * torch.cos(theta), CFG.r_des * torch.sin(theta), torch.full_like(theta, 5.0)], dim=-1)
    e = _errors(pos, theta)  # yaw == theta  =>  body +x == outward radial
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-5)


def test_radial_error_positive_means_closer_to_the_wall():
    theta = torch.zeros(1)
    pos = torch.tensor([[CFG.r_des + 0.3, 0.0, 5.0]])
    e = _errors(pos, theta)
    assert float(e[0, 0]) == pytest.approx(0.3, abs=1e-5)
    clearance = CFG.tank_radius - float(torch.linalg.norm(pos[0, :2]))
    assert clearance < CFG.d_ref, "positive radial error must mean LESS wall clearance"


def test_heading_error_matches_the_beam_offset_that_biases_the_sonar():
    """head_* is the chord of phi, so it is exactly the quantity sonar_range is sensitive to."""
    phi = torch.deg2rad(torch.tensor([25.0]))
    theta = torch.zeros(1)
    pos = torch.tensor([[CFG.r_des, 0.0, 5.0]])
    e = _errors(pos, theta + phi)
    chord = torch.linalg.norm(e[0, 6:8])
    assert float(chord) == pytest.approx(float(2 * torch.sin(phi / 2)), abs=1e-5)


def test_facing_inward_is_a_large_heading_error_not_a_small_one():
    """yaw_offset=pi is the whole inside-out mapping; inward-facing must NOT read as zero."""
    theta = torch.zeros(1)
    pos = torch.tensor([[CFG.r_des, 0.0, 5.0]])
    e = _errors(pos, theta + math.pi)
    assert float(torch.linalg.norm(e[0, 6:8])) == pytest.approx(2.0, abs=1e-5)


def test_tangential_and_radial_velocity_decomposition():
    theta = torch.tensor([math.pi / 2])  # at (0, r): outward = +y, +theta tangent = -x
    pos = torch.stack([CFG.r_des * torch.cos(theta), CFG.r_des * torch.sin(theta), torch.full_like(theta, 5.0)], dim=-1)
    # body +x == outward == world +y, so body +y == world -x == the +theta tangent
    v_b = torch.tensor([[0.07, 0.20, 0.0]])
    e = _errors(pos, theta, v_b=v_b)
    assert float(e[0, 3]) == pytest.approx(0.07, abs=1e-5)   # v_rad
    assert float(e[0, 4]) == pytest.approx(0.20, abs=1e-5)   # v_tan
    assert float(e[0, 5]) == pytest.approx(0.0, abs=1e-5)    # v_z


def test_velocity_targets_are_subtracted():
    theta = torch.zeros(1)
    pos = torch.tensor([[CFG.r_des, 0.0, 5.0]])
    v_b = torch.tensor([[0.0, 0.0, -0.2]])  # descending at 0.2 m/s
    e = _errors(pos, theta, v_b=v_b, v_z_des=torch.tensor([-0.2]))
    assert float(e[0, 5]) == pytest.approx(0.0, abs=1e-6), "matching the ramp rate is zero error"


# ---------------------------------------------------------------------------
# Reference preview vs. the real state machine
# ---------------------------------------------------------------------------


def test_ramp_preview_matches_scan_state_machine_slew():
    """The closed form must reproduce what scan_state_machine.step actually emits."""
    n_ctrl = 40
    cfg = mr.WallScanMPCCfg(dt_mpc=CFG.step_dt)  # 1 stage == 1 control step for a direct comparison
    scfg = ssm.ScanCfg(z_top=8.5, z_bottom=1.0, reach_eps=0.6, reach_hold=10,
                       ref_step=cfg.ref_step, ref_step_s=cfg.ref_step_s)
    st = ssm.ScanState(n=1)
    st.z_ramp[:] = 8.5
    z_far = torch.full((1,), 8.5)  # park z far from the bottom target so no phase advance fires
    emitted = []
    for _ in range(n_ctrl):
        z_ref, _s_ref, _sc, adv = ssm.step(st, z_far, torch.zeros(1), scfg)
        assert not bool(adv.any())
        emitted.append(float(z_ref))

    pred, disp = mr.ramp_preview(torch.full((1,), 8.5), torch.full((1,), scfg.z_bottom),
                                 cfg.ramp_per_stage_z, n_ctrl)
    assert torch.allclose(pred[0, 1:], torch.tensor(emitted), atol=1e-6)
    # second return is per-stage DISPLACEMENT; /dt_mpc is the speed target
    assert float(disp[0, 0]) == pytest.approx(-cfg.ramp_per_stage_z, abs=1e-6)
    assert float(disp[0, 0] / cfg.dt_mpc) == pytest.approx(-0.2, abs=1e-4)


def test_ramp_preview_saturates_at_the_target():
    ramp0 = torch.tensor([1.02])
    pred, disp = mr.ramp_preview(ramp0, torch.tensor([1.0]), 0.01, 8)
    assert float(pred[0, -1]) == pytest.approx(1.0, abs=1e-6)
    assert float(disp[0, -1]) == pytest.approx(0.0, abs=1e-6), "arrived ramp commands zero speed"


def test_reference_preview_holds_depth_and_moves_s_during_sway():
    phase = torch.tensor([1])  # SWAY_A
    out = mr.reference_preview(
        phase, z_ramp0=torch.tensor([4.0]), s_ramp0=torch.tensor([0.0]), s_ref=torch.tensor([1.0]),
        z_hold=torch.tensor([4.0]), cfg=CFG, n_stages=10,
    )
    assert torch.allclose(out["z_ref"], torch.full((1, 11), 4.0), atol=1e-6)
    assert torch.allclose(out["v_z_des"], torch.zeros(1, 11), atol=1e-6)
    assert float(out["v_tan_des"][0, 0]) == pytest.approx(CFG.ramp_per_stage_s / CFG.dt_mpc, abs=1e-6)
    assert float(out["v_tan_des"][0, 0]) == pytest.approx(0.1, abs=1e-4), "sway ramp is 0.1 m/s"


def test_reference_preview_descend_targets_the_bottom_at_scan_speed():
    out = mr.reference_preview(
        torch.tensor([0]), z_ramp0=torch.tensor([8.5]), s_ramp0=torch.tensor([0.0]),
        s_ref=torch.tensor([0.0]), z_hold=torch.tensor([0.0]), cfg=CFG, n_stages=10,
    )
    assert float(out["v_z_des"][0, 0]) == pytest.approx(-0.2, abs=1e-4), "heave ramp is 0.2 m/s down"
    assert (out["z_ref"][0].diff() < 0).all()


def test_reference_preview_ascend_targets_the_top():
    out = mr.reference_preview(
        torch.tensor([2]), z_ramp0=torch.tensor([1.0]), s_ramp0=torch.tensor([0.0]),
        s_ref=torch.tensor([0.0]), z_hold=torch.tensor([0.0]), cfg=CFG, n_stages=10,
    )
    assert float(out["v_z_des"][0, 0]) == pytest.approx(0.2, abs=1e-4)


# ---------------------------------------------------------------------------
# Tilt equilibrium: documents WHY the sway-tilt number is what it is
# ---------------------------------------------------------------------------


def test_sway_tilt_equilibrium_reproduces_the_measured_2_2_degrees():
    """The shipped TAM's parasitic arm predicts the tilt eval_metrics measures."""
    theta = mr.sway_tilt_equilibrium(
        v_sway=0.123, tam_arm=0.09, buoy_force=228.57, cob_z=0.15,
        lin_damp_y=119.44, quad_damp_y=38.51,
    )
    assert math.degrees(theta) == pytest.approx(2.3, abs=0.15)


def test_heave_differential_can_cancel_the_parasitic_moment_if_it_lands_on_roll():
    """8.6 N of heave differential at 0.123 m/s -- inside a 40 N thruster, so not a floor."""
    f_y = 119.44 * 0.123 + 38.51 * 0.123 ** 2
    needed = 0.09 * f_y / 0.16
    assert needed == pytest.approx(8.6, abs=0.3)
    assert needed < 40.0
