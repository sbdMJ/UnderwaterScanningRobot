# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PKRC WallScan underwater wall-scanning environment."""

import gymnasium as gym

# Lazy attribute access: gym entry-points are strings resolved at gym.make() time, so the env/cfg
# (which import isaaclab) must NOT be imported at package-import time — otherwise the pure-torch
# submodules (geometry/sensors/scan_state_machine) can't be imported/tested without the sim app.
_LAZY = {
    "WallScanEnv": ".wallscan_env",
    "WallScanStage1Cfg": ".wallscan_env_cfg",
    "WallScanStage2Cfg": ".wallscan_env_cfg",
    "WallScanStage3Cfg": ".wallscan_env_cfg",
    "WallScanTrainCfg": ".wallscan_env_cfg",
    "WallScanEvalCfg": ".wallscan_env_cfg",
}


def __getattr__(name):  # PEP 562 module-level lazy import
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

gym.register(
    id="Isaac-PKRC-WallScan-Stage1-Direct-v0",
    entry_point="marinelab.tasks.pkrc_wallscan:WallScanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc_wallscan:WallScanStage1Cfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc_wallscan.agents:WallScanPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PKRC-WallScan-Stage2-Direct-v0",
    entry_point="marinelab.tasks.pkrc_wallscan:WallScanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc_wallscan:WallScanStage2Cfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc_wallscan.agents:WallScanPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PKRC-WallScan-Stage3-Direct-v0",
    entry_point="marinelab.tasks.pkrc_wallscan:WallScanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc_wallscan:WallScanStage3Cfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc_wallscan.agents:WallScanPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PKRC-WallScan-Train-Direct-v0",
    entry_point="marinelab.tasks.pkrc_wallscan:WallScanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc_wallscan:WallScanTrainCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc_wallscan.agents:WallScanTrainPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PKRC-WallScan-Eval-Direct-v0",
    entry_point="marinelab.tasks.pkrc_wallscan:WallScanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marinelab.tasks.pkrc_wallscan:WallScanEvalCfg",
        "rsl_rl_cfg_entry_point": "marinelab.tasks.pkrc_wallscan.agents:WallScanEvalPPORunnerCfg",
    },
)
