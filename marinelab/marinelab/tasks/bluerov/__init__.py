# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV underwater vehicle environments.

This module provides reinforcement learning environments for the BlueROV UUV:
    - BlueROVEnv: Thruster-based control with 6-DOF hydrodynamics

Available Tasks:
    - Hover: Position and attitude maintenance
    - Attitude: Orientation control only
"""

import gymnasium as gym

from .attitude_env_cfg import BlueROVAttitudeEnvCfg, BlueROVAttitudeTrainEnvCfg
from .bluerov_env import BlueROVEnv
from .bluerov_env_cfg import BlueROVEnvCfg, DomainRandomizationCfg
from .hover_env_cfg import BlueROVHoverEnvCfg, BlueROVHoverEvalEnvCfg, BlueROVHoverTrainEnvCfg

##
# Register Gym environments
##

# BlueROV Hover environments
gym.register(
    id="Isaac-BlueROV-Hover-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.bluerov:BlueROVHoverEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.bluerov.agents:BlueROVHoverPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BlueROV-Hover-Train-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.bluerov:BlueROVHoverTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.bluerov.agents:BlueROVHoverTrainPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BlueROV-Hover-Eval-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.bluerov:BlueROVHoverEvalEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.bluerov.agents:BlueROVHoverEvalPPORunnerCfg",
    },
)

# BlueROV Attitude environments
gym.register(
    id="Isaac-BlueROV-Attitude-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.bluerov:BlueROVAttitudeEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.bluerov.agents:BlueROVAttitudePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BlueROV-Attitude-Train-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.bluerov:BlueROVAttitudeTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.bluerov.agents:BlueROVAttitudeTrainPPORunnerCfg",
    },
)

__all__ = [
    # Environment
    "BlueROVEnv",
    # Configurations
    "BlueROVEnvCfg",
    "DomainRandomizationCfg",
    "BlueROVHoverEnvCfg",
    "BlueROVHoverTrainEnvCfg",
    "BlueROVHoverEvalEnvCfg",
    "BlueROVAttitudeEnvCfg",
    "BlueROVAttitudeTrainEnvCfg",
]
