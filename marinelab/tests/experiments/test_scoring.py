# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Scoring scalar: numpy loss mirrors wallscan_loss; episode accounting; collision -> inf."""

import numpy as np
import pytest
import torch

from marinelab.algorithms.diff_wmpc import WallScanLossCfg, wallscan_loss
from marinelab.experiments.scoring import ScoreAccumulator, step_losses


def test_step_loss_matches_torch_reference():
    """The numpy scorer must equal the training loss it mirrors (normalized command)."""
    rng = np.random.default_rng(0)
    cfg = WallScanLossCfg()
    e = rng.normal(size=12)
    u_norm = rng.uniform(-1, 1, size=6)
    expected = wallscan_loss(torch.as_tensor(e), torch.as_tensor(u_norm) * cfg.max_thrust, cfg)
    got = step_losses(e, u_norm, cfg)
    assert got.shape == (1,)
    assert got[0] == pytest.approx(float(expected), rel=1e-9)


def test_episode_split_and_objective():
    acc = ScoreAccumulator(2)
    e = np.zeros((2, 12))
    e[:, 0] = 1.0  # radial error only: per-step loss = l_radial
    u = np.zeros((2, 6))
    acc.add(e, u)
    acc.add(e, u, done=np.array([True, False]))  # env0 episode ends at 2 steps
    acc.add(e, u)
    acc.finalize()

    l_r = WallScanLossCfg().l_radial
    s = acc.summary(score_episode=0)
    # env0: full episode of 2 steps; env1: partial episode of 3 steps (only one available)
    assert s["objective"] == pytest.approx((2 * l_r + 3 * l_r) / 2)
    assert not s["collided"]
    env0 = [ep for ep in s["episodes"] if ep["env"] == 0][0]
    assert env0["steps"] == 2 and not env0["partial"]


def test_collision_makes_objective_inf():
    acc = ScoreAccumulator(1)
    acc.add(np.zeros((1, 12)), np.zeros((1, 6)),
            done=np.array([True]), collided=np.array([True]))
    s = acc.summary()
    assert s["objective"] == float("inf") and s["collided"]


def test_score_episode_selection():
    acc = ScoreAccumulator(1)
    e1 = np.zeros((1, 12)); e1[0, 1] = 1.0  # episode 0: z error
    e2 = np.zeros((1, 12)); e2[0, 2] = 2.0  # episode 1: s error
    acc.add(e1, np.zeros((1, 6)), done=np.array([True]))
    acc.add(e2, np.zeros((1, 6)), done=np.array([True]))
    cfg = WallScanLossCfg()
    assert acc.summary(0)["objective"] == pytest.approx(cfg.l_z)
    assert acc.summary(1)["objective"] == pytest.approx(cfg.l_s * 4.0)
    # index past the end falls back to the last full episode
    assert acc.summary(5)["objective"] == pytest.approx(cfg.l_s * 4.0)
