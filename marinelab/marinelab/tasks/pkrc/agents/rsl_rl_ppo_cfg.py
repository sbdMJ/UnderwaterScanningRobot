# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PKRC PPO runner cfgs — reuse BlueROV hover hyperparameters, separate log dir."""
from __future__ import annotations

from isaaclab.utils import configclass

from marinelab.tasks.bluerov.agents.rsl_rl_ppo_cfg import (
    BlueROVHoverEvalPPORunnerCfg,
    BlueROVHoverPPORunnerCfg,
    BlueROVHoverTrainPPORunnerCfg,
)


@configclass
class PKRCHoverPPORunnerCfg(BlueROVHoverPPORunnerCfg):
    experiment_name = "pkrc_hover"


@configclass
class PKRCHoverTrainPPORunnerCfg(BlueROVHoverTrainPPORunnerCfg):
    experiment_name = "pkrc_hover"


@configclass
class PKRCHoverEvalPPORunnerCfg(BlueROVHoverEvalPPORunnerCfg):
    experiment_name = "pkrc_hover"
