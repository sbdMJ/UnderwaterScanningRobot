# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Tests for the wall-relative (r, phi, s) estimator.

These pin the geometry and the observability claims the Phase 2 analysis rests on, because
those claims — not the filter code — are what the design decisions were made from. The
end-to-end accuracy numbers live in ``scripts/replay_wall_frame_ekf.py``, which replays a real
logged trajectory; this file covers the pieces that must be right for that replay to mean
anything.
"""

import math

import numpy as np
import pytest

from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import (
    WallFrameEKF,
    WallFrameEKFCfg,
    gyro_drift_over_leg,
    mounted_sonar_jacobian,
    mounted_sonar_range,
    range_information,
    sonar_jacobian,
    sonar_range,
)

R = 6.0


# --- geometry ---------------------------------------------------------------


def test_sonar_range_matches_the_torch_implementation():
    """The scalar mirror must agree with mpc_reference's batched version, or the filter and
    the MPC would be predicting different sensors."""
    import torch

    from marinelab.tasks.pkrc_wallscan import mpc_reference as mref

    for r, phi in ((4.5, 0.0), (4.5, 0.3), (3.0, -0.5), (5.5, 0.05)):
        got = sonar_range(r, phi, R)
        ref = float(mref.sonar_range(torch.tensor([r]), torch.tensor([phi]), R))
        assert got == pytest.approx(ref, abs=1e-5)


def test_sonar_jacobian_matches_finite_differences():
    eps = 1e-6
    for r, phi in ((4.5, 0.1), (3.2, -0.4), (5.0, 0.02)):
        dr, dphi = sonar_jacobian(r, phi, R)
        fd_r = (sonar_range(r + eps, phi, R) - sonar_range(r - eps, phi, R)) / (2 * eps)
        fd_p = (sonar_range(r, phi + eps, R) - sonar_range(r, phi - eps, R)) / (2 * eps)
        assert dr == pytest.approx(fd_r, abs=1e-4)
        assert dphi == pytest.approx(fd_p, abs=1e-4)


def test_range_carries_no_heading_information_at_alignment():
    """dt/dphi = 0 at phi = 0 — the second-order insensitivity behind the crab-walk failure."""
    assert range_information(4.5, 0.0, R) == pytest.approx(0.0, abs=1e-12)
    assert range_information(4.5, math.radians(5), R) < range_information(4.5, math.radians(20), R)


def test_a_five_degree_offset_is_buried_under_the_sonar_bias():
    """Quantifies why one echo sounder cannot observe small heading errors.

    The signature of 5 deg is ~4 mm against a per-episode bias of up to 100 mm; no amount of
    averaging removes a bias, so this ratio is the hard limit the Phase 2 conclusion rests on.
    """
    signature = sonar_range(4.5, math.radians(5), R) - (R - 4.5)
    assert signature == pytest.approx(0.0043, abs=5e-4)
    assert signature / 0.10 < 0.05, "5 deg must be <5% of the bias for the claim to hold"


def test_gyro_drift_over_an_unexcited_leg():
    assert math.degrees(gyro_drift_over_leg(0.02, 39.0)) == pytest.approx(44.7, abs=0.5)
    assert math.degrees(gyro_drift_over_leg(4.8e-5, 39.0)) < 0.2


# --- velocity rotation -----------------------------------------------------


def test_velocity_components_at_alignment_are_the_body_axes():
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0)))
    v_rad, v_tan = ekf.velocity_components(0.07, 0.20)
    assert v_rad == pytest.approx(0.07)
    assert v_tan == pytest.approx(0.20)


def test_velocity_components_rotate_by_phi_not_by_yaw():
    """At phi = 90 deg the body x axis points along +theta, so surge becomes tangential."""
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, math.pi / 2, 0.0)))
    v_rad, v_tan = ekf.velocity_components(1.0, 0.0)
    assert v_rad == pytest.approx(0.0, abs=1e-9)
    assert v_tan == pytest.approx(1.0)


# --- filter behaviour ------------------------------------------------------


def _run(ekf, steps, *, v_bx=0.0, v_by=0.0, gyro=0.0, dt=0.02, truth=None, noise=0.0,
         rng=None, sensor_mount=None):
    """Feed the filter a constant-input segment, optionally with synthetic sonar.

    The synthetic range is generated with the MOUNTED model, because that is what a real
    transducer produces: it sits at body (0.10, 0) m, not at the body origin. ``sensor_mount``
    overrides the mount used to GENERATE the measurement, which is how the tests below simulate
    a filter whose model disagrees with the hardware.
    """
    mount = ekf.cfg.sonar_mount_pos if sensor_mount is None else sensor_mount
    for _ in range(steps):
        sonar = None
        if truth is not None:
            sonar = mounted_sonar_range(*truth, R, mount, ekf.cfg.sonar_yaw_offset)
            if noise:
                sonar += float(rng.normal(0, noise))
        ekf.step(v_bx=v_bx, v_by=v_by, gyro_z=gyro, dt=dt, sonar=sonar)
    return ekf


def test_stationary_filter_converges_r_from_sonar_alone():
    """With phi known-good, one range pins r — this is the part the sonar does well."""
    rng = np.random.default_rng(0)
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.0, 0.0, 0.0)))
    _run(ekf, 500, truth=(4.5, 0.0), noise=0.05, rng=rng)
    assert abs(ekf.r - 4.5) < 0.05, f"r={ekf.r}"


def test_clearance_is_reported_from_the_axis_distance():
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0)))
    assert ekf.clearance == pytest.approx(1.5)


def test_tangential_motion_advances_s_at_the_wall_radius():
    """s is the swath position along the WALL, so ds = R * dtheta, not r * dtheta."""
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0), q_phi=0.0, q_r=0.0, q_s=0.0))
    steps, dt, v_tan = 100, 0.02, 0.1
    _run(ekf, steps, v_by=v_tan, dt=dt)  # body +y is tangential at phi = 0
    expected = R * (v_tan / 4.5) * steps * dt
    assert ekf.s == pytest.approx(expected, rel=0.02)


def test_phi_tracks_gyro_when_there_is_no_rotation_of_the_radial():
    """Pure yaw with no translation: phi must follow the gyro one-for-one."""
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0)))
    _run(ekf, 100, gyro=0.1, dt=0.02)
    assert ekf.phi == pytest.approx(0.1 * 100 * 0.02, abs=1e-3)


def test_surge_is_only_radial_when_aligned():
    """At phi = 0, body +x IS the outward radial, so surge rotates nothing."""
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.2, 0.0, 0.0)))
    _run(ekf, 50, v_bx=0.1, dt=0.02)
    assert ekf.phi == pytest.approx(0.0, abs=1e-6)
    assert ekf.r > 4.2, "surge at phi=0 must move outward"


def test_surge_while_misaligned_also_rotates_the_radial():
    """The coupling that makes (r, phi) observable in the first place.

    Driving body +x at a nonzero beam offset produces a tangential component
    ``v_tan = v_bx*sin(phi)``, which sweeps the bearing and so changes ``phi = yaw - theta``
    even with the gyro at zero: ``dphi/dt = -v_tan/r``. This is exactly the mechanism the
    module docstring's observability argument relies on, so it is worth pinning numerically.
    """
    phi0, r0, v_bx, steps, dt = 0.15, 4.2, 0.1, 50, 0.02
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(r0, phi0, 0.0)))
    _run(ekf, steps, v_bx=v_bx, dt=dt)
    expected = phi0 - (v_bx * math.sin(phi0) / r0) * steps * dt
    assert ekf.phi == pytest.approx(expected, rel=0.05)
    assert ekf.phi < phi0, "the bearing sweeps forward, so the offset shrinks"


def test_gate_rejects_an_absurd_return_and_keeps_the_state():
    """A first return off a curved wall at a large offset is unreliable; one bad ping must not
    move the estimate."""
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0)))
    before = ekf.x.copy()
    accepted = ekf.update_sonar(5.9)  # implies a wildly different geometry
    assert accepted is False
    assert ekf.n_gated == 1
    assert np.allclose(ekf.x, before)


def test_ukfm_update_pulls_both_r_and_phi():
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.0, 0.3, 0.0)))
    for _ in range(50):
        ekf.update_ukfm(4.5, 0.0)
    assert abs(ekf.r - 4.5) < 0.05
    assert abs(ekf.phi) < 0.05
    assert ekf.n_ukfm == 50


def test_r_is_clamped_inside_the_tank():
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(5.9, 0.0, 0.0)))
    _run(ekf, 200, v_bx=1.0, dt=0.02)  # drive hard at the wall
    assert 0.0 < ekf.r < R, f"r escaped the tank: {ekf.r}"


def test_covariance_grows_without_measurements_and_shrinks_with_them():
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0)))
    p0 = ekf.P[1, 1]
    _run(ekf, 100, dt=0.02)                      # predict only
    assert ekf.P[1, 1] > p0
    grown = ekf.P[1, 1]
    for _ in range(100):
        ekf.update_ukfm(4.5, 0.0)
    assert ekf.P[1, 1] < grown


# ---------------------------------------------------------------------------
# Transducer mount in the measurement model (added 2026-07-31)
# ---------------------------------------------------------------------------

MOUNT = (0.10, 0.0)


def test_mounted_range_matches_the_cartesian_model_the_env_samples():
    """Cross-check against ``geometry.sonar_wall_distance``, which is the authority.

    The env draws the sonar truth from that Cartesian function; the filter has to predict the
    same number from its wall-relative state, or every update is biased. Checked across
    bearings too, since the wall-relative form must be invariant to where on the wall we are.
    """
    import torch

    from marinelab.tasks.pkrc_wallscan import geometry

    for r in (3.0, 4.5, 5.2):
        for phi in (0.0, 0.2, -0.35):
            for theta in (0.0, 1.3, -2.6):
                for mount in (MOUNT, (0.10, 0.05), (-0.08, 0.0)):
                    for yaw_off in (0.0, 0.04):
                        pos = torch.tensor([[r * math.cos(theta), r * math.sin(theta)]])
                        ref = float(geometry.sonar_wall_distance(
                            pos, torch.tensor([theta + phi]),
                            torch.tensor([list(mount)]), yaw_off, R)[0])
                        got = mounted_sonar_range(r, phi, R, mount, yaw_off)
                        assert got == pytest.approx(ref, abs=1e-5), (r, phi, theta, mount, yaw_off)


def test_zero_mount_reduces_to_the_body_origin_formula():
    for r, phi in ((4.5, 0.0), (4.2, 0.3), (5.0, -0.5)):
        assert mounted_sonar_range(r, phi, R, (0.0, 0.0), 0.0) == pytest.approx(
            sonar_range(r, phi, R), abs=1e-12
        )


def test_mounted_range_at_alignment_is_the_clearance_minus_the_forward_offset():
    """The simple case worth stating: facing the wall, the beam starts 10 cm closer to it."""
    assert mounted_sonar_range(4.5, 0.0, R, MOUNT, 0.0) == pytest.approx(1.40, abs=1e-9)
    assert sonar_range(4.5, 0.0, R) == pytest.approx(1.50, abs=1e-9)


def test_mounted_jacobian_matches_central_differences():
    """A wrong Jacobian is silent — it just makes every update the wrong size."""
    eps = 1e-6
    for r, phi in ((4.5, 0.12), (3.2, -0.4), (5.0, 0.02), (4.5, 0.0)):
        for mount in (MOUNT, (0.10, 0.05)):
            for yaw_off in (0.0, 0.04):
                dr, dphi = mounted_sonar_jacobian(r, phi, R, mount, yaw_off)
                fd_r = (mounted_sonar_range(r + eps, phi, R, mount, yaw_off)
                        - mounted_sonar_range(r - eps, phi, R, mount, yaw_off)) / (2 * eps)
                fd_p = (mounted_sonar_range(r, phi + eps, R, mount, yaw_off)
                        - mounted_sonar_range(r, phi - eps, R, mount, yaw_off)) / (2 * eps)
                assert dr == pytest.approx(fd_r, abs=1e-4), (r, phi, mount, yaw_off)
                assert dphi == pytest.approx(fd_p, abs=1e-4), (r, phi, mount, yaw_off)


def test_modelling_the_mount_removes_the_r_bias():
    """Sonar-only convergence to the TRUE r, which is the point of the whole change."""
    rng = np.random.default_rng(0)
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.0, 0.0, 0.0), sonar_mount_pos=MOUNT))
    _run(ekf, 800, truth=(4.5, 0.0), noise=0.0075, rng=rng)
    assert abs(ekf.r - 4.5) < 0.02, f"r={ekf.r}"


def test_omitting_the_mount_biases_r_by_about_the_offset():
    """Regression guard for the bug this replaced.

    A forward-mounted beam reads SHORT (1.40 m instead of 1.50 m at r = 4.5). A filter that
    places the beam at the body origin can only explain a short range by pushing ``r`` OUT, so
    it ends up believing the vehicle is nearer the wall than it is — and the controller then
    holds station too far away. Measured in the closed loop before the fix: +5.3 cm, the
    equilibrium between the sonar demanding +10 cm and UKF-M pulling back to truth. It got
    WORSE as the sonar improved — 10 cm r-RMSE with the Ping1D noise figure versus 5 cm with
    the placeholder — because a trusted sensor drags the state further toward a missing offset.
    """
    rng = np.random.default_rng(0)
    naive = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0), sonar_mount_pos=(0.0, 0.0)))
    _run(naive, 800, truth=(4.5, 0.0), noise=0.0075, rng=rng, sensor_mount=MOUNT)
    assert naive.r == pytest.approx(4.5 + MOUNT[0], abs=0.02), (
        f"r={naive.r}: the unmodelled 10 cm offset must land entirely in r, pushing it OUT"
    )


def test_clearance_uses_the_vehicle_not_the_transducer():
    """Collision margin is about the hull, so the reported clearance must not be the beam's."""
    ekf = WallFrameEKF(WallFrameEKFCfg(initial=(4.5, 0.0, 0.0), sonar_mount_pos=MOUNT))
    assert ekf.clearance == pytest.approx(1.5)
    assert mounted_sonar_range(4.5, 0.0, R, MOUNT, 0.0) < ekf.clearance


# ---------------------------------------------------------------------------
# Finite beam width (Ping1D is 25 deg, not a ray) — added 2026-08-03
# ---------------------------------------------------------------------------

BEAM = math.radians(12.5)


def test_inside_the_cone_the_reading_is_the_true_clearance():
    """The tank wall is CONCAVE from inside, so the cone still reaches the perpendicular.

    This overturns the single-ray reasoning used earlier in the project: crab does NOT bias the
    range at all until the misalignment exceeds the half-angle.
    """
    for deg in (0.0, 5.0, 10.0, 12.4):
        got = mounted_sonar_range(4.5, math.radians(deg), R, (0.0, 0.0), 0.0, BEAM)
        assert got == pytest.approx(6.0 - 4.5, abs=1e-9), deg


def test_outside_the_cone_only_the_excess_misalignment_shows():
    for deg, expect_cm in ((20.0, 0.97), (30.0, 5.37), (45.0, 19.60)):
        bias = mounted_sonar_range(4.5, math.radians(deg), R, (0.0, 0.0), 0.0, BEAM) - 1.5
        assert 100 * bias == pytest.approx(expect_cm, abs=0.05), deg


def test_the_wide_beam_always_reads_shorter_than_a_single_ray():
    """A cone can only find a nearer return than its axis, never a farther one."""
    for deg in (0.0, 8.0, 20.0, 40.0):
        phi = math.radians(deg)
        assert (mounted_sonar_range(4.5, phi, R, (0.10, 0.0), 0.0, BEAM)
                <= mounted_sonar_range(4.5, phi, R, (0.10, 0.0), 0.0, 0.0) + 1e-12)


def test_zero_half_angle_reproduces_the_single_ray_model():
    for r, phi in ((4.5, 0.0), (4.2, 0.3), (5.0, -0.5)):
        assert mounted_sonar_range(r, phi, R, (0.10, 0.0), 0.0, 0.0) == pytest.approx(
            mounted_sonar_range(r, phi, R, (0.10, 0.0), 0.0), abs=1e-12)


def test_range_carries_no_direct_heading_information_inside_the_cone():
    """With the beam at the body origin the sensitivity is EXACTLY zero inside the cone."""
    for deg in (0.0, 5.0, 11.0):
        _, dphi = mounted_sonar_jacobian(4.5, math.radians(deg), R, (0.0, 0.0), 0.0, BEAM)
        assert dphi == 0.0, deg
    _, dphi_out = mounted_sonar_jacobian(4.5, math.radians(25.0), R, (0.0, 0.0), 0.0, BEAM)
    assert abs(dphi_out) > 0.1, "outside the cone it must reappear"


def test_a_forward_mount_leaks_a_little_heading_sensitivity_through_the_lever_arm():
    """Not exactly zero once the transducer is offset: rotating the vehicle swings it sideways
    and changes its own axis distance. Small, but it is real and should not be zeroed out."""
    _, inside = mounted_sonar_jacobian(4.5, math.radians(5.0), R, (0.10, 0.0), 0.0, BEAM)
    _, outside = mounted_sonar_jacobian(4.5, math.radians(25.0), R, (0.10, 0.0), 0.0, BEAM)
    assert 0.0 < abs(inside) < 0.02, f"leak should be small: {inside}"
    assert abs(outside) > 20 * abs(inside), "and far smaller than a genuine off-cone reading"
    # it comes from the mount alone: zero mount -> exactly zero
    _, no_mount = mounted_sonar_jacobian(4.5, math.radians(5.0), R, (0.0, 0.0), 0.0, BEAM)
    assert no_mount == 0.0


def test_beam_jacobian_matches_central_differences_outside_the_cone():
    eps = 1e-6
    for deg in (20.0, 30.0, -28.0):
        phi = math.radians(deg)
        for mount in ((0.0, 0.0), (0.10, 0.0)):
            dr, dphi = mounted_sonar_jacobian(4.5, phi, R, mount, 0.0, BEAM)
            f = lambda p, rr=4.5: mounted_sonar_range(rr, p, R, mount, 0.0, BEAM)  # noqa: E731
            g = lambda rr: mounted_sonar_range(rr, phi, R, mount, 0.0, BEAM)       # noqa: E731
            assert dphi == pytest.approx((f(phi + eps) - f(phi - eps)) / (2 * eps), abs=1e-4)
            assert dr == pytest.approx((g(4.5 + eps) - g(4.5 - eps)) / (2 * eps), abs=1e-4)


def test_beam_model_matches_the_cartesian_geometry_helper():
    """Same cross-check as the mount: geometry.sonar_wall_distance is the authority."""
    import torch

    from marinelab.tasks.pkrc_wallscan import geometry

    for r in (3.5, 4.5, 5.2):
        for deg in (0.0, 8.0, 20.0, -35.0):
            for theta in (0.0, 1.7, -2.2):
                phi = math.radians(deg)
                pos = torch.tensor([[r * math.cos(theta), r * math.sin(theta)]])
                ref = float(geometry.sonar_wall_distance(
                    pos, torch.tensor([theta + phi]), torch.tensor([[0.10, 0.0]]), 0.0, R, BEAM)[0])
                got = mounted_sonar_range(r, phi, R, (0.10, 0.0), 0.0, BEAM)
                assert got == pytest.approx(ref, abs=1e-5), (r, deg, theta)


def test_dvl_hold_steps_from_the_published_ping_rate():
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, dvl_hold_steps

    assert dvl_hold_steps(SensorCfg(), 0.02) == 1, "0 Hz means legacy: fresh every step"
    assert dvl_hold_steps(SensorCfg(dvl_rate_hz=15.0), 0.02) == 3
    # 1/(4*0.02) = 12.5 and Python rounds half to even, so 12; either is defensible at the
    # published worst-case rate.
    assert dvl_hold_steps(SensorCfg(dvl_rate_hz=4.0), 0.02) == 12


# ---------------------------------------------------------------------------
# 3-axis instrument modelling (WallFrameEstimator.full_3axis)
#
# The default passes body v_z and the roll/pitch rates through from ground truth. That is
# under-modelling -- a DVL-A50 is 4-beam and a 3DM-GV7 is 3-axis -- and it flatters the vertical
# axis, which is the wallscan's primary motion, plus wrench_observer's force channel, which is
# computed FROM v_z. These tests pin both behaviours so neither can drift silently.
# ---------------------------------------------------------------------------
def test_full_3axis_stops_the_ground_truth_passthrough():
    import numpy as np
    import torch
    from marinelab.tasks.pkrc_wallscan.estimator_loop import WallFrameEstimator
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet

    def make(full):
        return WallFrameEstimator(
            scfg=SensorCfgDatasheet(), tank_radius=6.0, step_dt=0.02,
            sonar_mount_nom=torch.tensor([[0.10, 0.0]]), sonar_yaw_nom=0.0,
            gyro_bias=0.0, rng=np.random.default_rng(0), full_3axis=full)

    pos = torch.tensor([[4.4, 0.0, 5.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v_b = torch.tensor([[0.05, 0.02, -0.20]])       # a real descent: v_z is the big one
    w_b = torch.tensor([[0.03, -0.04, 0.01]])

    for est, full in ((make(False), False), (make(True), True)):
        est.reset(pos, quat, 0.0)
        out = est.step(0, pos, quat, v_b, w_b, 0.0, 0.0, 0.0, 0.0)
        vz_is_gt = abs(float(out.v_b[0, 2]) - float(v_b[0, 2])) < 1e-9
        wxy_is_gt = torch.allclose(out.w_b[0, :2], w_b[0, :2], atol=1e-9)
        assert vz_is_gt == (not full), f"full_3axis={full}: v_z passthrough wrong"
        assert wxy_is_gt == (not full), f"full_3axis={full}: w_xy passthrough wrong"


def test_full_3axis_default_is_off_so_published_numbers_hold():
    from marinelab.tasks.pkrc_wallscan.estimator_loop import WallFrameEstimator
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(WallFrameEstimator)}
    assert fields["full_3axis"].default is False
