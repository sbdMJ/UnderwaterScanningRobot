# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PKRC underwater vehicle environments (reuse BlueROVEnv, PKRC dynamics)."""

import gymnasium as gym

from .pkrc_hover_env_cfg import (
    PKRCHoverEnvCfg,
    PKRCHoverEvalEnvCfg,
    PKRCHoverTrainEnvCfg,
)

gym.register(
    id="Isaac-PKRC-Hover-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc:PKRCHoverEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc.agents:PKRCHoverPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PKRC-Hover-Train-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc:PKRCHoverTrainEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc.agents:PKRCHoverTrainPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PKRC-Hover-Eval-Direct-v0",
    entry_point="marinelab.tasks.bluerov:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc:PKRCHoverEvalEnvCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc.agents:PKRCHoverEvalPPORunnerCfg",
    },
)
