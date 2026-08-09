# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""E1 scoring scalar: accumulated task loss per episode (approved decision, plan §10-2).

The per-step loss is the numpy mirror of ``algorithms.diff_wmpc.wallscan_loss`` — the same
functional the proposed method trains on, and the analogue of the parent paper's
"accumulated loss per lap". Using it to score *every* method (and as the §6 tuning
objective) is maximally fair to the baselines: they are tuned on exactly what the proposed
method optimizes. A collision makes the trial's objective ``inf`` (failed).

Errors are always evaluated on GROUND TRUTH state, whatever state source the controller
ran on — scoring measures what the vehicle actually did.
"""
from __future__ import annotations

import numpy as np

from marinelab.algorithms.diff_wmpc import WallScanLossCfg


def loss_weights(cfg: WallScanLossCfg) -> np.ndarray:
    """(12,) per-entry weights in ``mpc_reference.ERROR_NAMES`` order."""
    return np.array([
        cfg.l_radial, cfg.l_z, cfg.l_s, cfg.l_v_rad, cfg.l_v_tan, cfg.l_v_z,
        cfg.l_heading, cfg.l_heading, cfg.l_rollpitch, cfg.l_rollpitch,
        cfg.l_omega, cfg.l_omega,
    ])


def step_losses(errors: np.ndarray, u_norm: np.ndarray, cfg: WallScanLossCfg) -> np.ndarray:
    """(n_envs,) per-step task loss. ``u_norm`` is the NORMALIZED command in [-1, 1]."""
    e = np.atleast_2d(np.asarray(errors, float))
    u = np.atleast_2d(np.asarray(u_norm, float))
    return e**2 @ loss_weights(cfg) + cfg.l_u * (u**2).mean(axis=1)


class ScoreAccumulator:
    """Accumulates per-step losses into per-episode records across parallel envs."""

    def __init__(self, n_envs: int, cfg: WallScanLossCfg | None = None):
        self.cfg = cfg or WallScanLossCfg()
        self.n_envs = int(n_envs)
        self._sum = np.zeros(n_envs)
        self._steps = np.zeros(n_envs, dtype=int)
        self.episodes: list[dict] = []

    def add(self, errors: np.ndarray, u_norm: np.ndarray,
            done: np.ndarray | None = None, collided: np.ndarray | None = None) -> None:
        self._sum += step_losses(errors, u_norm, self.cfg)
        self._steps += 1
        if done is not None:
            done = np.asarray(done, bool).reshape(self.n_envs)
            collided = (np.zeros(self.n_envs, bool) if collided is None
                        else np.asarray(collided, bool).reshape(self.n_envs))
            for env in np.flatnonzero(done):
                self.episodes.append({"env": int(env), "loss": float(self._sum[env]),
                                      "steps": int(self._steps[env]),
                                      "collided": bool(collided[env]), "partial": False})
                self._sum[env] = 0.0
                self._steps[env] = 0

    def finalize(self) -> None:
        """Close still-running episodes (flagged partial; scored only if nothing else exists)."""
        for env in range(self.n_envs):
            if self._steps[env] > 0:
                self.episodes.append({"env": env, "loss": float(self._sum[env]),
                                      "steps": int(self._steps[env]),
                                      "collided": False, "partial": True})
                self._sum[env] = 0.0
                self._steps[env] = 0

    def summary(self, score_episode: int = 0) -> dict:
        """Objective = mean scored-episode loss across envs; any scored collision -> inf."""
        per_env: dict[int, list[dict]] = {}
        for ep in self.episodes:
            per_env.setdefault(ep["env"], []).append(ep)
        scored, collided = [], False
        for env, eps in sorted(per_env.items()):
            full = [ep for ep in eps if not ep["partial"]] or eps
            ep = full[min(score_episode, len(full) - 1)]
            collided |= ep["collided"]
            scored.append(float("inf") if ep["collided"] else ep["loss"])
        objective = float("inf") if (collided or not scored) else float(np.mean(scored))
        return {
            "objective": objective,
            "collided": collided,
            "scored_losses": scored,
            "episodes": self.episodes,
        }
