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
