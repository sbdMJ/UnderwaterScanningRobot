# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Actuator-rate model bookkeeping — the pure half of the nx 13+nu augmentation.

The OCP itself needs acados (verified in-container by ``isaaclab/logs/_probe_rate_mpc.py``);
what is natively testable is the applied-force state that ``WallScanMPC.solve`` keeps
between ticks: it must integrate the normalized rate command at the modelled slew limit
over the CONTROL tick, respect the per-thruster force caps, and never step faster than the
teleop ramp can realize — that invariant is what makes publishing the one-tick-ahead force
exact rather than optimistic.
"""
import numpy as np

from marinelab.tasks.pkrc_wallscan.mpc_controller import advance_actuator

RATE = np.array([27.098, 27.098, 29.818, 29.818, 16.83, 16.83])  # hw2026 N/s
FLIM = np.full(6, 3.68)
DT = 0.02


def test_full_rate_steps_exactly_the_slew_times_tick():
    f = advance_actuator(np.zeros(6), np.ones(6), RATE, FLIM, DT)
    np.testing.assert_allclose(f, RATE * DT)
    # heave: 16.83 N/s * 0.02 s = 0.34 N per tick -> a +-3.68 N reversal needs ~22 ticks
    assert abs(f[4] - 0.3366) < 1e-9


def test_rate_command_is_clipped_to_unit_range():
    f = advance_actuator(np.zeros(6), np.full(6, 5.0), RATE, FLIM, DT)
    np.testing.assert_allclose(f, RATE * DT)  # not 5x


def test_force_saturates_at_the_per_thruster_caps():
    caps = np.array([0.0, 0.0, 0.0, 0.0, 2.25, 2.25])  # depth-hold session limits
    f = np.zeros(6)
    for _ in range(2000):
        f = advance_actuator(f, np.ones(6), RATE, caps, DT)
    np.testing.assert_allclose(f, caps)
    # neutered horizontal thrusters can never leave zero
    f = advance_actuator(np.zeros(6), -np.ones(6), RATE, caps, DT)
    np.testing.assert_allclose(f[:4], 0.0)


def test_reversal_takes_the_measured_ramp_time():
    """+-3 A heave reversal at 17 A/s took ~0.35 s in the 03_41/04_15 bags; the model's
    force-domain equivalent must reproduce that order, not teleport."""
    f = np.full(6, 3.68)
    ticks = 0
    while f[4] > -3.68 + 1e-9:
        f = advance_actuator(f, -np.ones(6), RATE, FLIM, DT)
        ticks += 1
    assert 0.3 < ticks * DT < 0.5  # 7.36 N swing / 16.83 N/s = 0.437 s


def test_scalar_and_partial_inputs_broadcast():
    f = advance_actuator([0.0], [0.5], [10.0], [1.0], 0.1)
    np.testing.assert_allclose(f, [0.5])
