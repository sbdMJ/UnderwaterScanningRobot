# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Controller interface every E1 method implements, plus shared cost instrumentation.

The interface is deliberately minimal: ``reset`` at episode boundaries, ``step`` once per
control tick. ``step`` receives the estimated :class:`~marinelab.control.types.VehicleState`
and the previewed :class:`~marinelab.control.types.ScanReference`; policy-based methods that
consume the task's 31-D observation vector take it through the optional ``obs`` argument.

``ControllerStats`` is the E4(c) measurement point: every adapter records its per-step
compute time and failure/saturation counts here, so the inference-cost table is produced the
same way for every method, in sim and on the Jetson.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .types import ControlOutput, ScanReference, VehicleState

SATURATION_THRESHOLD = 0.98  # |u_cmd| above this counts as saturated (matches run_wallscan_mpc)


@dataclass
class ControllerStats:
    """Accumulated per-step cost/health counters, reported by ``summary()``."""

    n_steps: int = 0
    n_fail: int = 0
    n_saturated: int = 0
    solve_ms: list = field(default_factory=list)

    def record(self, out: ControlOutput) -> None:
        self.n_steps += 1
        if out.status != 0:
            self.n_fail += 1
        if np.abs(out.u_cmd).max() > SATURATION_THRESHOLD:
            self.n_saturated += 1
        self.solve_ms.append(out.solve_ms)

    def summary(self) -> dict:
        ms = np.asarray(self.solve_ms, float) if self.solve_ms else np.zeros(1)
        n = max(1, self.n_steps)
        return {
            "n_steps": self.n_steps,
            "solve_ms_mean": float(ms.mean()),
            "solve_ms_p95": float(np.percentile(ms, 95)),
            "solve_ms_max": float(ms.max()),
            "fail_frac": self.n_fail / n,
            "saturated_frac": self.n_saturated / n,
        }


class WallScanController(ABC):
    """One scan-control method (Fixed-W NMPC, BO NMPC, PPO, SSI-MPC, Diff-WMPC, ...)."""

    #: short method key used in result filenames and tables ("fixed", "bo", "ppo", ...)
    name: str = "base"

    def __init__(self) -> None:
        self.stats = ControllerStats()

    @abstractmethod
    def reset(self, state: VehicleState) -> None:
        """Episode-boundary reset (warm starts, histories, online adaptation state)."""

    @abstractmethod
    def step(self, state: VehicleState, ref: ScanReference,
             obs: np.ndarray | None = None) -> ControlOutput:
        """Compute one control command. Implementations must ``self.stats.record()`` it."""
