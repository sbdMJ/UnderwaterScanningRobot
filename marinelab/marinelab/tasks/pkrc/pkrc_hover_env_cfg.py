# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PKRC hover env configs — reuse the BlueROV hover env, swap in PKRC robot/hydro/thrusters.

The env class (BlueROVEnv) is robot-agnostic: it reads cfg.robot / cfg.hydrodynamics /
cfg.thrusters. So PKRC only overrides those three; task logic, rewards, DR are inherited.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from marinelab.assets.pkrc import PKRC_CFG, PKRCHydrodynamicsCfg, PKRCThrusterCfg

from ..bluerov.hover_env_cfg import (
    BlueROVHoverEnvCfg,
    BlueROVHoverEvalEnvCfg,
    BlueROVHoverTrainEnvCfg,
)

_PKRC_ROBOT = PKRC_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class PKRCHoverEnvCfg(BlueROVHoverEnvCfg):
    robot = _PKRC_ROBOT
    hydrodynamics = PKRCHydrodynamicsCfg()
    thrusters = PKRCThrusterCfg()
    body_link_name = "Robot"          # single body is named 'Robot' (BlueROV uses 'base_link')


@configclass
class PKRCHoverTrainEnvCfg(BlueROVHoverTrainEnvCfg):
    robot = _PKRC_ROBOT
    hydrodynamics = PKRCHydrodynamicsCfg()
    thrusters = PKRCThrusterCfg()
    body_link_name = "Robot"          # single body is named 'Robot' (BlueROV uses 'base_link')


@configclass
class PKRCHoverEvalEnvCfg(BlueROVHoverEvalEnvCfg):
    robot = _PKRC_ROBOT
    hydrodynamics = PKRCHydrodynamicsCfg()
    thrusters = PKRCThrusterCfg()
    body_link_name = "Robot"          # single body is named 'Robot' (BlueROV uses 'base_link')
