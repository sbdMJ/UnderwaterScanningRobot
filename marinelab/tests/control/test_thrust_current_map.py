# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""ThrustCurrentMap: u (sim order, ±1=±40 N) -> VESC ampere commands."""
import numpy as np
import pytest

from marinelab.control.thrust_current_map import SIM_TO_TELEOP_SIGN, ThrustCurrentMap

NO_SIGN = (1.0,) * 6


def test_uncalibrated_full_scale_matches_the_teleop_manual_currents():
    m = ThrustCurrentMap(sign=NO_SIGN)
    amps = m.map([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    assert not m.calibrated
    assert list(amps) == [3.0, -3.0, 3.0, -3.0, 5.0, -5.0]


def test_default_sign_translates_sim_axes_into_teleop_command_space():
    """Bench 2026-08-09: sim +y (port) needs a NEGATIVE teleop sway command on T3/T4,
    and sim +z (up) is teleop heave DOWN on T6 — the (+,+,-,-,+,-) translation."""
    assert ThrustCurrentMap().sign == SIM_TO_TELEOP_SIGN == (1.0, 1.0, -1.0, -1.0, 1.0, -1.0)
    amps = ThrustCurrentMap().map([1.0] * 6)
    assert list(np.sign(amps)) == [1.0, 1.0, -1.0, -1.0, 1.0, -1.0]


def test_calibrated_path_converts_newtons_through_the_thrust_constant():
    # k = 10 N/A everywhere: u=0.5 -> 20 N -> 2 A
    m = ThrustCurrentMap(sign=NO_SIGN, newton_per_amp=(10.0,) * 6, max_thrust=40.0)
    amps = m.map([0.5] * 6)
    assert m.calibrated
    assert amps == pytest.approx([2.0] * 6)
    # limits still clamp: u=1 -> 4 A but surge limit is 3 A
    amps = m.map([1.0] * 6)
    assert amps[:4] == pytest.approx([3.0] * 4) and amps[4:] == pytest.approx([4.0] * 2)


def test_order_remap_moves_sim_channels_into_vesc_slots():
    m = ThrustCurrentMap(sign=NO_SIGN, order=(0, 1, 2, 3, 5, 4))
    amps = m.map([0.0, 0.0, 0.0, 0.0, 1.0, -1.0])
    assert amps[4] == pytest.approx(-5.0) and amps[5] == pytest.approx(5.0)


def test_u_is_clamped_before_conversion():
    m = ThrustCurrentMap()
    assert np.max(np.abs(m.map([7.0] * 6))) <= 5.0


def test_bad_order_is_rejected():
    with pytest.raises(ValueError):
        ThrustCurrentMap(order=(0, 0, 2, 3, 4, 5))
