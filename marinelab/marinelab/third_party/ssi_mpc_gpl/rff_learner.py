# SPDX-License-Identifier: GPL-3.0-only — see LICENSE in this directory.
"""SSI-MPC's online system identification: random Fourier features + no-regret OGD.

Port of UM-iRaL/SSI-MPC @ e0d4afb (see LICENSE for provenance). The adaptation law is
preserved verbatim from ``quad_3d_optimizer.run_optimization``::

    rf(Z)     = 1/sqrt(n_rf) * cos(omega @ Bz @ Z + b),   Z = [x; u]
    err_pred  = Bh.T @ (x_pred(alpha_last) - x_meas) / dt
    alpha    <-  alpha_last - 2 * lr * (err_pred @ rf.T)

with ``omega ~ N(0, kernel_std)`` of shape (n_rf, len(input_mask)) and
``b ~ U[0, 2*pi]`` of shape (n_rf, 1) exactly as ``mpc_node.set_random_features``
(Rahimi & Recht, NIPS 2007; Gaussian kernel option).

``x_pred`` uses the caller-supplied NOMINAL one-step predictor with the residual term
applied via explicit Euler — the upstream code's own documented alternative to the
alpha-augmented RK4 (their in-line comment at run_optimization: RK4 is non-convex in
alpha; the Euler form makes the update exactly the OGD step above). Under Euler the two
formulations are algebraically identical.

The learned residual is an acceleration on the ``target_mask`` state dimensions,
queried as ``alpha @ rf([x; 0])`` — upstream's "heuristic" injection mode, which maps
onto the wallscan OCP's existing per-stage disturbance parameter without a solver
rebuild (see ``ssi_controller.py``).
"""
from __future__ import annotations

import numpy as np


class RFFOnlineLearner:
    """No-regret online residual-dynamics learner (upstream adaptation law, verbatim)."""

    def __init__(self, *, state_dim: int, u_dim: int, target_mask: list[int],
                 input_mask: list[int], n_rf: int = 100, lr: float = 0.1,
                 kernel_std: float = 1.0, kernel: str = "gaussian", seed: int = 0):
        self.state_dim, self.u_dim = int(state_dim), int(u_dim)
        self.target_mask = list(target_mask)
        self.input_mask = list(input_mask)
        self.n_rf, self.lr = int(n_rf), float(lr)
        rng = np.random.default_rng(seed)
        # mpc_node.set_random_features: omega scale = kernel_std, b ~ U[0, 2pi]
        if kernel == "gaussian":
            self.omega = rng.normal(0.0, kernel_std, size=(self.n_rf, len(self.input_mask)))
        elif kernel == "cauchy":
            self.omega = rng.standard_cauchy(size=(self.n_rf, len(self.input_mask)))
        elif kernel == "laplace":
            self.omega = rng.laplace(0.0, 1.0, size=(self.n_rf, len(self.input_mask)))
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        self.b = rng.uniform(0.0, 2 * np.pi, size=(self.n_rf, 1))
        # quad_3d_optimizer.__init__: Bh maps residuals to state space, Bz selects features
        self.Bh = np.eye(self.state_dim)[self.target_mask].T
        self.Bz = np.eye(self.state_dim + self.u_dim)[self.input_mask]
        self.alpha = np.zeros((len(self.target_mask), self.n_rf))
        self.last_pred_err = float("nan")  # |err_pred| of the latest OGD step (Fig 3(c) diag)
        self._x_last: np.ndarray | None = None
        self._u_last: np.ndarray | None = None

    def rf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """(n_rf, 1) feature vector of Z = [x; u]."""
        Z = np.concatenate([np.asarray(x, float).reshape(-1),
                            np.asarray(u, float).reshape(-1)])[:, None]
        return (1.0 / np.sqrt(self.n_rf)) * np.cos(self.omega @ self.Bz @ Z + self.b)

    def record_control(self, x: np.ndarray, u: np.ndarray) -> None:
        """Remember the (state, applied control) pair the next update will be scored on."""
        self._x_last = np.asarray(x, float).reshape(self.state_dim).copy()
        self._u_last = np.asarray(u, float).reshape(self.u_dim).copy()

    def update(self, x_meas: np.ndarray, dt: float, predict_nominal) -> np.ndarray:
        """One OGD step against the newly measured state; returns the updated alpha.

        ``predict_nominal(x_last, u_last, dt) -> x_pred0`` is the alpha-free nominal
        one-step prediction; the residual term is applied via explicit Euler on the
        target dims (see module docstring).
        """
        if self._x_last is None or self._u_last is None:
            return self.alpha  # first tick: nothing to score yet
        x_meas = np.asarray(x_meas, float).reshape(self.state_dim)
        rf = self.rf(self._x_last, self._u_last)  # (n_rf, 1)
        x_pred0 = np.asarray(predict_nominal(self._x_last, self._u_last, dt),
                             float).reshape(self.state_dim)
        # x_pred(alpha) = x_pred0 + dt * Bh @ (alpha @ rf)  =>  upstream error_pred:
        err_pred = (self.Bh.T @ (x_pred0 - x_meas) / dt)[:, None] + self.alpha @ rf
        self.last_pred_err = float(np.linalg.norm(err_pred))
        self.alpha = self.alpha - 2.0 * self.lr * (err_pred @ rf.T)
        return self.alpha

    def residual_now(self, x: np.ndarray) -> np.ndarray:
        """(n_target,) current residual acceleration — upstream heuristic mode:
        ``alpha @ rf([x; 0])`` with zero control input."""
        rf = self.rf(x, np.zeros(self.u_dim))
        return (self.alpha @ rf).reshape(-1)

    def reset_episode(self) -> None:
        """Episode boundary: drop the stale transition; KEEP alpha (adaptation persists,
        as in upstream's continuous operation)."""
        self._x_last = self._u_last = None
