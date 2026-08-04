# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Fixed-weight NMPC adapter — E1 method ① and, fed tuned weights, method ② (BO-static).

Wraps the existing :class:`marinelab.tasks.pkrc_wallscan.mpc_controller.WallScanMPC`
unchanged. The solver pair is expensive to build (acados codegen), so it can be injected
via ``mpc=`` — which is also what makes this adapter unit-testable without acados.

Weight sources, in priority order: explicit ``werr``/``wu`` arguments, a ``params_json``
file (the §6 tuning pipeline's ``best_params.json``), else ``DEFAULT_WERR``/``DEFAULT_WU``.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .base import WallScanController
from .types import ControlOutput, ScanReference, VehicleState


def load_weight_params(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``{"werr": [...], "wu": [...]}`` from a tuning ``best_params.json``."""
    with open(path) as fh:
        data = json.load(fh)
    return np.asarray(data["werr"], float), np.asarray(data["wu"], float)


class FixedWeightNMPC(WallScanController):
    name = "fixed"

    def __init__(self, *, mpc=None, plant=None, mpc_cfg=None, horizon: int = 30,
                 rti_iters: int = 8, code_export_root: str | None = None,
                 werr=None, wu=None, params_json: str | None = None):
        super().__init__()
        if mpc is None:
            if plant is None or mpc_cfg is None:
                raise ValueError("either pass a built WallScanMPC via mpc=, or plant= and mpc_cfg=")
            from marinelab.tasks.pkrc_wallscan.mpc_controller import WallScanMPC

            mpc = WallScanMPC(plant, mpc_cfg, N=horizon, rti_iters=rti_iters,
                              with_sensitivity=False, code_export_root=code_export_root)
        self._mpc = mpc

        from marinelab.tasks.pkrc_wallscan.mpc_controller import DEFAULT_WERR, DEFAULT_WU

        if params_json is not None:
            j_werr, j_wu = load_weight_params(params_json)
            werr = j_werr if werr is None else werr
            wu = j_wu if wu is None else wu
            self.name = "bo"  # weights came from the tuning pipeline
        werr = np.asarray(DEFAULT_WERR if werr is None else werr, float).reshape(-1)
        wu_arr = np.full(self._mpc.nu, DEFAULT_WU) if wu is None else np.asarray(wu, float).reshape(-1)
        if wu_arr.size == 1:
            wu_arr = np.full(self._mpc.nu, float(wu_arr[0]))
        self._weights = np.concatenate([werr, wu_arr])
        if self._weights.size != self._mpc.n_pglobal:
            raise ValueError(f"weights length {self._weights.size} != solver's {self._mpc.n_pglobal}")
        self._first = True
        # Optional per-step world-frame disturbance fed to the OCP's d_world parameter.
        # None = zeros. Subclasses with online model adaptation (SSI-MPC) set this.
        self._d_world: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def set_weights(self, werr, wu) -> None:
        """Swap the fixed weight set without rebuilding the solver (weights are per-solve).

        This is what makes the §6 tuning loop cheap: one acados build, N trials.
        """
        werr = np.asarray(werr, float).reshape(-1)
        wu_arr = np.asarray(wu, float).reshape(-1)
        if wu_arr.size == 1:
            wu_arr = np.full(self._mpc.nu, float(wu_arr[0]))
        w = np.concatenate([werr, wu_arr])
        if w.size != self._mpc.n_pglobal:
            raise ValueError(f"weights length {w.size} != solver's {self._mpc.n_pglobal}")
        self._weights = w

    def reset(self, state: VehicleState) -> None:
        self._first = True

    def _current_weights(self, state: VehicleState, ref: ScanReference) -> np.ndarray:
        """Weight source for this step; Diff-WMPC overrides this with its policy."""
        return self._weights

    def step(self, state: VehicleState, ref: ScanReference,
             obs: np.ndarray | None = None) -> ControlOutput:
        weights = self._current_weights(state, ref)
        P = self._mpc.param_matrix(ref.as_dict(), theta_anchor=ref.theta_anchor,
                                   s_anchor=ref.s_anchor, d_world=self._d_world)
        t0 = time.perf_counter()
        sol = self._mpc.solve(state.to_x13(), P, weights, want_sensitivity=False,
                              init_state_traj=self._first)
        solve_ms = 1e3 * (time.perf_counter() - t0)
        self._first = False
        out = ControlOutput(u_cmd=np.asarray(sol["u0_cmd"], float), solve_ms=solve_ms,
                            status=int(sol["status"]), u_newton=np.asarray(sol["u0"], float),
                            aux={"werr": weights[:-self._mpc.nu], "wu": weights[-self._mpc.nu:]})
        self.stats.record(out)
        return out
