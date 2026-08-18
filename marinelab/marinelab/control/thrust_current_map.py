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
* CALIBRATED + DEADZONE: the 2026-08-11 bollard pull measured zero thrust below
  ~0.7 A (friction torque must be overcome first), so the calibrated model is affine,
  ``F = k·(I − I₀)`` → ``I = sign(F)·(I₀ + |F|/k)``, with a small ``deadband_u`` under
  which the command is zeroed instead of jumping to ±I₀.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: u index (sim TAM column) -> VESC slot (T1..T6), per the Step-0 correspondence table.
#: Bench-verified 2026-08-09: identity holds.
SIM_TO_VESC_ORDER = (0, 1, 2, 3, 4, 5)

#: sim axis convention -> teleop command sign, from the 2026-08-09 bench run and pinned
#: by direct in-water observation (2026-08-11/12): the vehicle is positively buoyant and
#: (I_T5, I_T6) = (-1, +1) drives it DOWN, so sim +z (up) maps to (+, -) on the heave
#: pair. (A 2026-08-11 reanalysis of the 122531 bag briefly argued the opposite; that
#: session had tether handling — an unmodeled external force — and was retracted. See
#: thruster_mapping.md §4e.)
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
    #: measured deadzone currents I₀ (A, command space) per thruster; None = no offset.
    #: Only honoured on the calibrated path: I = sign(F)·(I₀ + |F|/k).
    amps_offset: tuple[float, ...] | None = None
    #: |u| at or below which the command is zeroed rather than jumping to ±I₀ —
    #: keeps closed-loop chatter around u=0 from banging the deadzone compensation.
    deadband_u: float = 0.05
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
            if self.amps_offset is not None:
                off = np.asarray(self.amps_offset, dtype=float)
                amps_sim = np.where(np.abs(u) > self.deadband_u,
                                    amps_sim + np.sign(u) * off, 0.0)
        else:
            amps_sim = u * np.asarray(self.amps_at_full, dtype=float)
        out = np.zeros(6)
        out[self._order_idx] = amps_sim
        return np.clip(out, -np.asarray(self.amps_limit), np.asarray(self.amps_limit))


def fit_thrust_affine(amps, newtons) -> tuple[float, float, float]:
    """Least-squares fit of the deadzone thrust model ``F = k·(I − I₀)``.

    The 2026-08-11 bollard pull read exactly zero at 0.5 A on every pair — friction
    torque must be overcome before the prop produces thrust, so the model is affine,
    not through-origin. Only thrust-producing points (F > 0) enter the fit; zero rows
    are the caller's consistency check (they must sit at or below the fitted I₀).
    Returns ``(k [N/A], i0 [A, >= 0], worst_abs_residual [N])``.
    """
    a = np.abs(np.asarray(amps, dtype=float).reshape(-1))
    f = np.abs(np.asarray(newtons, dtype=float).reshape(-1))
    live = f > 0.0
    if int(live.sum()) < 2:
        raise ValueError("need at least 2 thrust-producing points to fit (k, I0)")
    k, b = np.polyfit(a[live], f[live], 1)
    if k <= 0.0:
        raise ValueError(f"non-positive fitted slope {k:.3f} N/A — data inconsistent")
    i0 = max(0.0, float(-b / k))
    resid = float(np.max(np.abs(f[live] - k * (a[live] - i0))))
    return float(k), i0, resid


def split_pair_constants(k_sum: float, null_amps_a: float, null_amps_b: float) -> tuple[float, float]:
    """Decompose a pair-run calibration into per-thruster constants.

    Pair bollard pull at equal per-thruster current gives the SUM ``k_a + k_b`` (the
    moments of a symmetric pair cancel, which is what makes a narrow bridle workable —
    thruster_mapping.md §4b). The free-float zero-moment null — currents ``(I_a*, I_b*)``
    at which the pair produces no yaw (roll for the heave pair) — gives the RATIO via
    ``k_a·I_a* = k_b·I_b*``. Together they pin ``(k_a, k_b)``.
    """
    if k_sum <= 0.0 or null_amps_a <= 0.0 or null_amps_b <= 0.0:
        raise ValueError("k_sum and both null currents must be positive")
    r = null_amps_b / null_amps_a  # = k_a / k_b
    k_b = k_sum / (1.0 + r)
    return k_sum - k_b, k_b


def fit_thrust_constant(amps, newtons) -> tuple[float, float]:
    """Through-origin least squares fit of the per-thruster thrust constant k [N/A].

    Bollard-pull data reduction (thruster_mapping.md §4a): thrust and torque both scale
    ~rpm², so under VESC current (torque) control F = k·I through the origin. Returns
    ``(k, worst_frac_residual)`` — the worst |F − k·I|/|F| over the samples. A residual
    above ~0.1 usually means the lowest-current point sits in the stiction/deadzone
    region and should be dropped and refit.
    """
    a = np.abs(np.asarray(amps, dtype=float).reshape(-1))
    f = np.abs(np.asarray(newtons, dtype=float).reshape(-1))
    if a.size != f.size or a.size == 0:
        raise ValueError("amps and newtons must be equal-length, non-empty")
    if np.dot(a, a) == 0.0:
        raise ValueError("all current samples are zero — nothing to fit")
    k = float(np.dot(f, a) / np.dot(a, a))
    resid = np.abs(f - k * a) / np.maximum(f, 1e-9)
    return k, float(np.max(resid))
