# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""WallScan PPO runner cfgs — reuse BlueROV hover hyperparameters, separate log dir."""
from __future__ import annotations

from isaaclab.utils import configclass

from marinelab.tasks.bluerov.agents.rsl_rl_ppo_cfg import (
    BlueROVHoverEvalPPORunnerCfg,
    BlueROVHoverPPORunnerCfg,
    BlueROVHoverTrainPPORunnerCfg,
)


def _kill_entropy_bonus(cfg) -> None:
    """Tame the entropy bonus for clamp(-1,1) actions: 0.005-0.01 blew sigma up
    (1.0 -> 95 -> 251, bang-bang control), but 0.0 collapsed it to 0.01 by ~19k
    iters (stage3_fix3) — exploration died and waypoint plateaued at ~375.
    0.001 keeps sigma alive without the free-reward inflation; watch
    `Mean action noise std`: healthy band is ~0.1-0.5, rising past 1.0 = red flag.

    gamma 0.99 -> 0.995 (07-24 f_stage3 park diag): the 0.99 effective horizon
    (~100 steps = 2 s) hid the ~29 s ascend transit's reward recovery behind the
    discount, so phase advance looked net-negative and the policy parked in SWAY_A.

    entropy history (full empirical map): 0.005-0.008 exploded sigma 1->251;
    0.001 looked stable on an already-collapsed lineage (sigma 0.01) but from a
    FRESH sigma=1.0 start it still exploded 1->39 by 9k iters (s_stage3); 0.0
    collapsed a healthy sigma to 0.01 by ~19k iters. Verdict: with clamp(-1,1)
    actions no positive entropy bonus is safe long-term -> 0.0, and rely on
    fresh-start sigma=1.0 for exploration. Watch `Mean action noise std`.
    """
    cfg.algorithm.entropy_coef = 0.0
    cfg.algorithm.gamma = 0.995


@configclass
class WallScanPPORunnerCfg(BlueROVHoverPPORunnerCfg):
    experiment_name = "pkrc_wallscan"

    def __post_init__(self):
        _kill_entropy_bonus(self)


@configclass
class WallScanTrainPPORunnerCfg(BlueROVHoverTrainPPORunnerCfg):
    experiment_name = "pkrc_wallscan"

    def __post_init__(self):
        _kill_entropy_bonus(self)


@configclass
class WallScanEvalPPORunnerCfg(BlueROVHoverEvalPPORunnerCfg):
    experiment_name = "pkrc_wallscan"

    def __post_init__(self):
        _kill_entropy_bonus(self)
