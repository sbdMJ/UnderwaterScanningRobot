# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Controller thrust commands -> VESC current commands (the u -> ampere half of Phase C-③).

The controller's ``u`` is normalized per-thruster thrust in the SIM's allocation-matrix
column order (``u = ±1`` ↔ ``±PlantParams.max_thrust`` newtons). The vehicle's VESCs take
current commands (torque control), and propeller thrust and torque both scale ~rpm², so
thrust ≈ k·current to first order — which is what makes this a per-thruster scalar map.

Division of labour (docs/experiments/sim-to-real/thruster_mapping.md §5):

* here (repo-owned): sim-order -> VESC-order remap and the N -> A calibration curve;
* teleop (CAN owner): wiring polarity, deadzone compensation, current clamp, ramp, e-stop.

Calibration states:

* UNCALIBRATED (default): ``amps_at_full`` — |u| = 1 maps linearly to the teleop manual
  scale (surge/sway 3 A, heave 5 A). Same authority as a human driver, but u↔N consistency
  is NOT established: closed-loop gains are effectively scaled down. Low-bandwidth trials
  (depth hold) only.
* CALIBRATED: ``newton_per_amp`` per thruster (bollard pull / datasheet, Step 1-2) —
  ``I = u * max_thrust / k_i``, still clamped to ``amps_limit``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: u index (sim TAM column) -> VESC slot (T1..T6), per the Step-0 correspondence table.
#: Bench-verified 2026-08-09: identity holds.
SIM_TO_VESC_ORDER = (0, 1, 2, 3, 4, 5)

#: sim axis convention -> teleop command sign, derived from the 2026-08-09 bench run
#: anchored on teleop's proven absolute behaviours (UP drives forward, LEFT drives port —
#: the sway inversion is already absorbed at teleop's key layer — and depth-hold works,
#: fixing heave+ = descend). sim wants +x fwd / +y port / +z up per pair; teleop's command
#: space is +surge fwd / +sway starboard / +heave down, and the teleop auto path applies
#: its own wiring polarity on top, so this is axis translation only, not wiring.
SIM_TO_TELEOP_SIGN = (1.0, 1.0, -1.0, -1.0, 1.0, -1.0)


@dataclass
class ThrustCurrentMap:
    order: tuple[int, ...] = SIM_TO_VESC_ORDER
    #: per-thruster sign translating the sim axis convention into teleop command space
    #: (teleop's wiring polarity is applied downstream by teleop itself).
    sign: tuple[float, ...] = SIM_TO_TELEOP_SIGN
    #: teleop's manual full-scale currents (A) — the uncalibrated fallback scale.
    amps_at_full: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 5.0, 5.0)
    #: hard per-thruster clamp (A); teleop clamps again, this keeps the topic sane too.
    amps_limit: tuple[float, ...] = (3.0, 3.0, 3.0, 3.0, 5.0, 5.0)
    #: measured thrust constants (N/A) per thruster; None = uncalibrated fallback.
    newton_per_amp: tuple[float, ...] | None = None
    #: what |u| = 1 means in newtons (PlantParams.max_thrust).
    max_thrust: float = 40.0

    _order_idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if sorted(self.order) != [0, 1, 2, 3, 4, 5]:
            raise ValueError(f"order must be a permutation of 0..5, got {self.order}")
        self._order_idx = np.asarray(self.order, dtype=int)

    @property
    def calibrated(self) -> bool:
        return self.newton_per_amp is not None

    def map(self, u) -> np.ndarray:
        """(6,) u in sim order -> (6,) ampere commands in VESC slot order (T1..T6).

        Polarity is NOT applied here — that is wiring knowledge and stays in teleop.
        """
        u = np.clip(np.asarray(u, dtype=float).reshape(6), -1.0, 1.0)
        u = u * np.asarray(self.sign, dtype=float)
        if self.newton_per_amp is not None:
            k = np.asarray(self.newton_per_amp, dtype=float)
            amps_sim = u * self.max_thrust / k
        else:
            amps_sim = u * np.asarray(self.amps_at_full, dtype=float)
        out = np.zeros(6)
        out[self._order_idx] = amps_sim
        return np.clip(out, -np.asarray(self.amps_limit), np.asarray(self.amps_limit))
