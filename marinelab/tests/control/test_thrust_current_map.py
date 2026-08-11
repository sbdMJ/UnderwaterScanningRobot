# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""ThrustCurrentMap: u (sim order, ±1=±40 N) -> VESC ampere commands."""
import numpy as np
import pytest

from marinelab.control.thrust_current_map import (
    SIM_TO_TELEOP_SIGN,
    ThrustCurrentMap,
    fit_thrust_affine,
    fit_thrust_constant,
    split_pair_constants,
)

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


def test_fit_recovers_the_thrust_constant_from_clean_bollard_data():
    amps = [0.5, 1.0, 1.5, 2.0]
    k_true = 2.1
    k, resid = fit_thrust_constant(amps, [k_true * a for a in amps])
    assert k == pytest.approx(k_true)
    assert resid == pytest.approx(0.0, abs=1e-12)


def test_fit_works_on_signed_reverse_direction_samples():
    # reverse-direction rows carry negative amps; the fit uses magnitudes
    k, _ = fit_thrust_constant([-0.5, -1.0, -2.0], [1.0, 2.0, 4.0])
    assert k == pytest.approx(2.0)


def test_fit_flags_a_deadzone_point_via_the_worst_residual():
    # 0.5 A sits in the stiction region and reads low; the flag must exceed 10%
    amps = [0.5, 1.0, 1.5, 2.0]
    f = [0.5 * 2.0 * 0.5, 2.0, 3.0, 4.0]  # first point at half the true thrust
    _, resid = fit_thrust_constant(amps, f)
    assert resid > 0.10


def test_fit_rejects_degenerate_input():
    with pytest.raises(ValueError):
        fit_thrust_constant([], [])
    with pytest.raises(ValueError):
        fit_thrust_constant([0.0, 0.0], [1.0, 2.0])


def test_pair_split_satisfies_both_the_sum_and_the_null_condition():
    # pair sum 4.0 N/A; yaw-null found at (I_a*, I_b*) = (1.0, 1.25)
    k_a, k_b = split_pair_constants(4.0, 1.0, 1.25)
    assert k_a + k_b == pytest.approx(4.0)
    # zero-moment condition: k_a·I_a* = k_b·I_b*
    assert k_a * 1.0 == pytest.approx(k_b * 1.25)


def test_pair_split_is_equal_for_a_symmetric_null():
    assert split_pair_constants(4.2, 1.5, 1.5) == pytest.approx((2.1, 2.1))


def test_pair_split_rejects_nonpositive_inputs():
    for bad in ((0.0, 1.0, 1.0), (4.0, -1.0, 1.0), (4.0, 1.0, 0.0)):
        with pytest.raises(ValueError):
            split_pair_constants(*bad)


def test_affine_fit_recovers_slope_and_deadzone():
    # F = 3.2·(I − 0.7), plus a genuine deadzone point at 0.5 A reading zero
    amps = [0.5, 1.0, 1.5, 2.0]
    f = [0.0, 3.2 * 0.3, 3.2 * 0.8, 3.2 * 1.3]
    k, i0, resid = fit_thrust_affine(amps, f)
    assert k == pytest.approx(3.2)
    assert i0 == pytest.approx(0.7)
    assert resid == pytest.approx(0.0, abs=1e-12)


def test_affine_fit_clamps_negative_deadzone_to_zero():
    amps = [1.0, 2.0]
    k, i0, _ = fit_thrust_affine(amps, [2.1, 4.0])  # intercept slightly positive
    assert i0 == 0.0 and k > 0


def test_affine_fit_needs_two_live_points():
    with pytest.raises(ValueError):
        fit_thrust_affine([0.5, 1.0], [0.0, 1.0])


def test_calibrated_map_applies_the_deadzone_offset():
    # k = 1.6 N/A, I0 = 0.7 A, max_thrust = 1.6·(3.0 − 0.7): u=±1 lands exactly
    # on the 3 A limit, and a tiny u inside the deadband commands zero.
    m = ThrustCurrentMap(sign=NO_SIGN, newton_per_amp=(1.6,) * 6,
                         amps_offset=(0.7,) * 6, max_thrust=1.6 * 2.3,
                         amps_limit=(3.0,) * 6)
    amps = m.map([1.0, -1.0, 0.5, 0.02, 0.0, -0.5])
    assert amps[0] == pytest.approx(3.0)
    assert amps[1] == pytest.approx(-3.0)
    assert amps[2] == pytest.approx(0.7 + 0.5 * 2.3)
    assert amps[3] == 0.0 and amps[4] == 0.0  # deadband_u = 0.05
    assert amps[5] == pytest.approx(-(0.7 + 0.5 * 2.3))
