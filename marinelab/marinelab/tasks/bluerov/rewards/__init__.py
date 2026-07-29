# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward system for BlueROV environments.

This module provides a modular reward system following Isaac Lab conventions.
Reward terms are defined as standalone functions or callable classes, composed
through RewardTermCfg configurations.
"""

from .reward_manager import RewardManager, RewardTermCfg
from .reward_terms import (
    action_magnitude_penalty,
    action_rate_penalty,
    alive_bonus,
    angular_velocity_penalty,
    linear_velocity_penalty,
    orientation_upright_exp,
    position_tracking_exp,
)

__all__ = [
    # Manager
    "RewardManager",
    "RewardTermCfg",
    # UUV reward terms
    "position_tracking_exp",
    "orientation_upright_exp",
    "linear_velocity_penalty",
    "angular_velocity_penalty",
    "action_rate_penalty",
    "action_magnitude_penalty",
    "alive_bonus",
]
