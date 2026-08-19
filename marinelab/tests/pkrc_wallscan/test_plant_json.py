# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PlantParams JSON round-trip — the hardware side's only source of plant truth.

The Jetson cannot call ``PlantParams.from_env`` (no sim env), so it loads the JSON the sim
exported (``marinelab/config/pkrc_plant_fixed_tam.json``). These tests pin that the file
parses back into an identical dataclass and that it carries the fixed-TAM signature the
controller's tilt compensation depends on.
"""
import dataclasses
import os

import numpy as np

from marinelab.tasks.pkrc_wallscan.mpc_controller import PlantParams

_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "config",
                     "pkrc_plant_fixed_tam.json")


def test_round_trip_is_lossless(tmp_path):
    prm = PlantParams.from_json(_JSON)
    out = tmp_path / "plant.json"
    prm.to_json(str(out))
    again = PlantParams.from_json(str(out))
    assert dataclasses.asdict(again) == dataclasses.asdict(prm)


def test_committed_export_carries_the_fixed_tam_signature():
    prm = PlantParams.from_json(_JSON)
    assert prm.mass == np.float32(22.8)
    B = np.asarray(prm.allocation_matrix)
    assert B.shape == (6, 6)
    assert np.all(B[4] == 0.0), "My row must be zero (heave pair at x=0 — fixed TAM)"
    assert B[3, 2] == B[3, 3] == 0.09, "sway arm lives in Mx on the fixed TAM"


def test_heave_roll_arms_match_the_bench_survey():
    """2026-08-09 bench: T5 (u4) is the STARBOARD heave thruster and the pair sits
    29.5 cm apart, so the roll arm is -0.1475 on u4 and +0.1475 on u5 (was the sim
    guess of +/-0.16 with the sides swapped)."""
    B = np.asarray(PlantParams.from_json(_JSON).allocation_matrix)
    assert B[3, 4] == -0.1475, "u4 -> T5 (starboard, y = -0.1475)"
    assert B[3, 5] == +0.1475, "u5 -> T6 (port, y = +0.1475)"


_JSON_HW = os.path.join(os.path.dirname(_JSON), "pkrc_plant_hw2026.json")


def test_hw2026_export_carries_the_measured_field_values():
    """2026-08-15 field calibration (thruster_mapping.md §4d/§4g): 0.5 kg lead trim,
    measured drivetrain limit and per-axis drag scales (surge x0.175, sway x0.151,
    heave x0.20 on both damping terms). TAM/added-mass/rotational damping inherit the
    fixed-TAM export (unmeasured)."""
    hw = PlantParams.from_json(_JSON_HW)
    sim = PlantParams.from_json(_JSON)
    assert hw.mass == np.float32(22.8) + 0.5, "0.5 kg lead ballast on the dry mass"
    assert abs(hw.buoyancy_force - (hw.mass * 9.81 + 0.24)) < 1e-6, "+0.24 N measured residual"
    assert hw.max_thrust == 3.68, "min k_i(limit_i - I0_i), binding T1 @ 3 A"
    for i, s in enumerate((0.175, 0.151, 0.20)):
        assert abs(hw.linear_damping[i] - sim.linear_damping[i] * s) < 1e-3
        assert abs(hw.quadratic_damping[i] - sim.quadratic_damping[i] * s) < 1e-3
    assert hw.linear_damping[3:] == sim.linear_damping[3:], "rotational drag unmeasured"
    assert hw.allocation_matrix == sim.allocation_matrix, "TAM is the bench-verified one"


def test_hw2026_carries_the_actuator_rate_model():
    """2026-08-19 (bag 04_15_35 postmortem): the deployed chain slew-limits the VESC
    current at 17-30 A/s (teleop ramp); the OCP models the conservative 17 A/s so its
    plan is always realizable. force_rate_limit = newton_per_amp * 17."""
    hw = PlantParams.from_json(_JSON_HW)
    k = (1.594, 1.594, 1.754, 1.754, 0.99, 0.99)  # thrust_mapper.py calibration defaults
    assert hw.force_rate_limit is not None, "hw plant must enable the rate model"
    for fr, ki in zip(hw.force_rate_limit, k):
        assert abs(fr - ki * 17.0) < 1e-2
    assert hw.thrust_limits is None, "session force caps are a node parameter, not plant truth"


def test_sim_export_keeps_the_legacy_instant_force_model():
    """E1-E4 reproducibility: the sim plant JSON predates the actuator fields and must
    keep loading as the instant-force model (nx 13)."""
    sim = PlantParams.from_json(_JSON)
    assert sim.force_rate_limit is None
    assert sim.thrust_limits is None


def test_hw2026_round_trip_is_lossless(tmp_path):
    prm = PlantParams.from_json(_JSON_HW)
    out = tmp_path / "plant_hw.json"
    prm.to_json(str(out))
    assert dataclasses.asdict(PlantParams.from_json(str(out))) == dataclasses.asdict(prm)
