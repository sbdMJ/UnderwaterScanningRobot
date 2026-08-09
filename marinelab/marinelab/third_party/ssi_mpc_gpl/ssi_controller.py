# SPDX-License-Identifier: GPL-3.0-only — see LICENSE in this directory.
"""SSI-MPC controller for the wallscan task: nominal NMPC + RFF online SysID.

Upstream (UM-iRaL/SSI-MPC @ e0d4afb) solves a quadrotor OCP whose stage parameters carry
the learned residual. Our wallscan OCP already has a per-stage world-frame disturbance
parameter ``d_world`` (mpc_controller.ND = 3), so upstream's "heuristic" injection mode
maps onto it with NO solver rebuild: the learner's residual acceleration on the body
linear-velocity dims becomes ``d_world = m * R_wb @ r_b``.

The nominal one-step predictor reuses the exact CasADi plant model the MPC solves with
(``mpc_controller._continuous_dynamics``) — the SysID sees the same nominal model as the
controller, as in upstream.
"""
from __future__ import annotations

import numpy as np

from marinelab.control.fixed_nmpc import FixedWeightNMPC
from marinelab.control.types import ControlOutput, ScanReference, VehicleState

from .rff_learner import RFFOnlineLearner

STATE_DIM, U_DIM = 13, 6
TARGET_MASK_DEFAULT = [7, 8, 9]  # body linear-velocity rows of x13 (v_dot residual)
# Features: attitude + velocities + control; absolute position excluded (residuals must
# not key on absolute xy in a rotationally symmetric tank). Tunable via the §6 pipeline.
INPUT_MASK_DEFAULT = list(range(3, STATE_DIM + U_DIM))


def _quat_to_rot_np(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class SSIMPCController(FixedWeightNMPC):
    name = "ssi"

    def __init__(self, *, step_dt: float, ssi_lr: float = 0.1, ssi_kernel_std: float = 1.0,
                 ssi_n_rf: int = 100, ssi_seed: int = 0, ssi_kernel: str = "gaussian",
                 target_mask=None, input_mask=None, predict_fn=None,
                 mass: float | None = None, max_thrust: float | None = None, **nmpc_kwargs):
        plant = nmpc_kwargs.get("plant")
        super().__init__(**nmpc_kwargs)
        self.name = "ssi"
        self._dt = float(step_dt)
        self._mass = float(mass if mass is not None else plant.mass)
        self._max_thrust = float(max_thrust if max_thrust is not None else plant.max_thrust)
        self._plant = plant
        self._predict_fn = predict_fn  # (x13, u_norm, dt) -> x13; casadi-built lazily if None
        self._masks = (list(target_mask or TARGET_MASK_DEFAULT),
                       list(input_mask or INPUT_MASK_DEFAULT))
        self._make_learner(lr=ssi_lr, kernel_std=ssi_kernel_std, n_rf=ssi_n_rf,
                           seed=ssi_seed, kernel=ssi_kernel)

    def _make_learner(self, *, lr, kernel_std, n_rf, seed, kernel) -> None:
        self._learner_cfg = dict(lr=lr, kernel_std=kernel_std, n_rf=n_rf, seed=seed,
                                 kernel=kernel)
        self._learner = RFFOnlineLearner(
            state_dim=STATE_DIM, u_dim=U_DIM, target_mask=self._masks[0],
            input_mask=self._masks[1], n_rf=int(n_rf), lr=float(lr),
            kernel_std=float(kernel_std), kernel=kernel, seed=int(seed))

    def reconfigure(self, *, lr=None, kernel_std=None, seed=None) -> None:
        """Fresh learner with new hyperparameters (§6 tuning trials); solver untouched."""
        cfg = dict(self._learner_cfg)
        if lr is not None:
            cfg["lr"] = float(lr)
        if kernel_std is not None:
            cfg["kernel_std"] = float(kernel_std)
        if seed is not None:
            cfg["seed"] = int(seed)
        self._make_learner(**cfg)
        self._d_world = None

    @property
    def learner(self) -> RFFOnlineLearner:
        return self._learner

    def _predict(self, x: np.ndarray, u_norm: np.ndarray, dt: float) -> np.ndarray:
        if self._predict_fn is None:
            self._predict_fn = self._build_casadi_predictor()
        return self._predict_fn(x, u_norm, dt)

    def _build_casadi_predictor(self):
        """RK4 one-step predictor from the SAME CasADi model the MPC solves with."""
        import casadi as ca

        from marinelab.tasks.pkrc_wallscan.mpc_controller import _continuous_dynamics

        prm = self._plant
        if prm is None:
            raise RuntimeError("no plant params: pass predict_fn= when injecting mpc=")
        B = np.asarray(prm.allocation_matrix, float)
        x = ca.SX.sym("x", STATE_DIM)
        u = ca.SX.sym("u", U_DIM)  # newtons
        dt = ca.SX.sym("dt")
        k1 = _continuous_dynamics(x, u, prm, B, ca.DM.zeros(3))
        k2 = _continuous_dynamics(x + 0.5 * dt * k1, u, prm, B, ca.DM.zeros(3))
        k3 = _continuous_dynamics(x + 0.5 * dt * k2, u, prm, B, ca.DM.zeros(3))
        k4 = _continuous_dynamics(x + dt * k3, u, prm, B, ca.DM.zeros(3))
        fn = ca.Function("ssi_pred", [x, u, dt], [x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])
        max_thrust = self._max_thrust

        def predict(x_np, u_norm, dt_s):
            return np.asarray(fn(x_np, np.asarray(u_norm, float) * max_thrust, dt_s)).reshape(-1)

        return predict

    def reset(self, state: VehicleState) -> None:
        super().reset(state)
        self._learner.reset_episode()
        self._d_world = None

    def step(self, state: VehicleState, ref: ScanReference,
             obs: np.ndarray | None = None) -> ControlOutput:
        x = state.to_x13()
        self._learner.update(x, self._dt, lambda xl, ul, dt: self._predict(xl, ul, dt))
        r_b = self._learner.residual_now(x)  # (3,) residual accel, body frame
        R = _quat_to_rot_np(x[3:7])
        self._d_world = self._mass * (R @ r_b)
        out = super().step(state, ref, obs)
        self._learner.record_control(x, out.u_cmd)
        out.aux["ssi_residual_b"] = r_b
        out.aux["ssi_alpha_norm"] = float(np.linalg.norm(self._learner.alpha))
        out.aux["ssi_pred_err"] = self._learner.last_pred_err
        return out
