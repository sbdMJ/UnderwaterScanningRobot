# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""TopicSampleAssembler: the pure half of the hardware estimator bridge.

These pin the three things the e5_ekf campaign showed actually matter: measurement
freshness (one message = one correction), the marker-world -> tank frame conversion, and
the absolute bearing riding along as the third ukfm element (the s-drift fix's input).
"""
import math

import pytest

from marinelab.control.hw_bridge import TankCalib, TopicSampleAssembler


def _feed_prediction_inputs(a: TopicSampleAssembler, t: float = 0.0) -> None:
    a.feed_dvl(0.1, 0.02, 0.0, t)
    a.feed_imu(0.0, 0.0, 0.05, 0.01, -0.02, t)
    a.feed_depth(3.0, t)


def test_not_ready_until_every_prediction_input_arrived():
    a = TopicSampleAssembler()
    assert a.assemble(0.0) is None
    assert a.missing() == ("dvl", "imu", "depth")
    a.feed_dvl(0.1, 0.0, 0.0, 0.0)
    a.feed_imu(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert a.assemble(0.0) is None, "depth still missing"
    assert a.missing() == ("depth",), "the node's why-am-I-silent log names the holdout"
    a.feed_depth(3.0, 0.0)
    assert a.assemble(0.0) is not None
    assert a.missing() == ()


def test_measurements_are_consumed_once_and_prediction_inputs_hold():
    a = TopicSampleAssembler()
    _feed_prediction_inputs(a)
    a.feed_wall_range(1.45, 0.0)
    a.feed_ukfm(4.5, 0.0, 0.0, 0.0)

    first = a.assemble(0.02)
    assert first.sonar == pytest.approx(1.45) and first.ukfm is not None
    second = a.assemble(0.04)
    assert second.sonar is None and second.ukfm is None, "one message, one correction"
    assert second.v_bx == pytest.approx(0.1), "DVL holds between updates"

    a.feed_wall_range(1.50, 0.05)
    assert a.assemble(0.06).sonar == pytest.approx(1.50)


def test_depth_converts_pressure_depth_to_tank_z():
    a = TopicSampleAssembler(TankCalib(tank_height=10.0))
    _feed_prediction_inputs(a)
    a.feed_depth(3.0, 0.0)  # 3 m below surface
    assert a.assemble(0.0).depth == pytest.approx(7.0)  # 7 m above the floor


def test_ukfm_fix_maps_to_wall_frame_and_carries_the_absolute_bearing():
    a = TopicSampleAssembler(TankCalib())  # identity calibration
    _feed_prediction_inputs(a)
    # robot at (3, 4): r = 5, theta = atan2(4, 3); heading = theta (nose on the radial)
    theta = math.atan2(4.0, 3.0)
    a.feed_ukfm(3.0, 4.0, theta, 0.0)
    ukfm = a.assemble(0.0).ukfm
    assert len(ukfm) == 3, "the third element (theta) is the s-correction's input"
    assert ukfm[0] == pytest.approx(5.0)
    assert ukfm[1] == pytest.approx(0.0, abs=1e-12)
    assert ukfm[2] == pytest.approx(theta)


def test_marker_calibration_transform_is_applied_before_wall_frame():
    # Marker origin 1 m along tank +x, marker frame rotated +90 deg vs tank frame:
    # marker-frame (1, 0) with yaw 0 -> tank (1, 1), yaw pi/2.
    calib = TankCalib(marker_x=1.0, marker_y=0.0, marker_yaw=math.pi / 2)
    a = TopicSampleAssembler(calib)
    _feed_prediction_inputs(a)
    a.feed_ukfm(1.0, 0.0, 0.0, 0.0)
    r, phi, theta = a.assemble(0.0).ukfm
    assert r == pytest.approx(math.sqrt(2.0))
    assert theta == pytest.approx(math.pi / 4)
    assert phi == pytest.approx(math.pi / 2 - math.pi / 4)


def test_ages_report_channel_staleness_for_the_health_gate():
    a = TopicSampleAssembler()
    _feed_prediction_inputs(a, t=1.0)
    a.feed_wall_range(1.4, 1.2)
    ages = a.ages(now=2.0)
    assert ages["dvl"] == pytest.approx(1.0) and ages["wall"] == pytest.approx(0.8)
