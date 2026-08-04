# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""WallFrameStateEstimator: SensorSample -> VehicleState reconstruction invariants."""

import math

import numpy as np
import pytest

from marinelab.control.estimator import SensorSample, WallFrameStateEstimator
from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import WallFrameEKFCfg, sonar_range

R_TANK = 6.0
R0, PHI0, THETA0 = 4.5, 0.1, 0.7


def _make():
    est = WallFrameStateEstimator(
        WallFrameEKFCfg(sonar_mount_pos=(0.0, 0.0), sonar_yaw_offset=0.0))
    est.reset(r0=R0, phi0=PHI0, theta0=THETA0)
    return est


def _sample(depth=5.0):
    return SensorSample(v_bx=0.0, v_by=0.0, gyro_z=0.0,
                        sonar=sonar_range(R0, PHI0, R_TANK), depth=depth,
                        roll=0.0, pitch=0.0, v_bz=0.1, gyro_x=0.02, gyro_y=0.03)


def test_requires_reset():
    est = WallFrameStateEstimator()
    with pytest.raises(RuntimeError):
        est.step(_sample(), 0.02)


def test_anchor_and_stationary_consistency():
    est = _make()
    assert est.theta_hat == pytest.approx(THETA0)
    assert est.s_hat == pytest.approx(0.0)

    for _ in range(50):
        state = est.step(_sample(), 0.02)

    # stationary + model-consistent sonar: the filter must hold (r0, phi0)
    assert est.ekf.r == pytest.approx(R0, abs=0.05)
    assert est.ekf.phi == pytest.approx(PHI0, abs=0.05)
    assert est.s_hat == pytest.approx(0.0, abs=0.02)

    # reconstruction: |xy| = r, z = depth, yaw = theta_hat + phi, passthrough channels intact
    assert float(np.linalg.norm(state.pos_w[:2])) == pytest.approx(est.ekf.r, abs=1e-9)
    assert state.pos_w[2] == pytest.approx(5.0)
    yaw = 2.0 * math.atan2(state.quat_wb[3], state.quat_wb[0])
    assert yaw == pytest.approx(est.theta_hat + est.ekf.phi, abs=1e-9)
    assert float(np.linalg.norm(state.quat_wb)) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(state.lin_vel_b, [0.0, 0.0, 0.1])
    np.testing.assert_allclose(state.ang_vel_b, [0.02, 0.03, 0.0])


def test_x13_feeds_mpc_layout():
    est = _make()
    x = est.step(_sample(), 0.02).to_x13()
    assert x.shape == (13,)
