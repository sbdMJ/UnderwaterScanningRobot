# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent configurations for BlueROV environments.

This module provides pre-configured hyperparameters for training BlueROV
control policies for hover and attitude tasks.

Available configurations:
    Hover Task:
        - BlueROVHoverPPORunnerCfg: Base configuration
        - BlueROVHoverTrainPPORunnerCfg: Training with domain randomization
        - BlueROVHoverEvalPPORunnerCfg: Evaluation with aggressive randomization

    Attitude Task:
        - BlueROVAttitudePPORunnerCfg: Base configuration
        - BlueROVAttitudeTrainPPORunnerCfg: Training with domain randomization
"""

from .rsl_rl_ppo_cfg import (
    BlueROVAttitudePPORunnerCfg,
    BlueROVAttitudeTrainPPORunnerCfg,
    BlueROVHoverEvalPPORunnerCfg,
    BlueROVHoverPPORunnerCfg,
    BlueROVHoverTrainPPORunnerCfg,
)

__all__ = [
    "BlueROVHoverPPORunnerCfg",
    "BlueROVHoverTrainPPORunnerCfg",
    "BlueROVHoverEvalPPORunnerCfg",
    "BlueROVAttitudePPORunnerCfg",
    "BlueROVAttitudeTrainPPORunnerCfg",
]
