# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward term functions for BlueROV environments.

This module provides standalone reward functions following Isaac Lab conventions.
Each function computes a reward component for the given robot state.

Function signature:
    func(robot: Articulation, **params) -> torch.Tensor

where the returned tensor has shape (num_envs,).

Note: Weights and dt scaling are applied by the RewardManager, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


# =============================================================================
# Standard UUV Reward Terms
# =============================================================================


def position_tracking_exp(
    robot: Articulation,
    goal_pos_w: torch.Tensor,
    exp_scale: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Exponential position tracking reward.

    Computes: exp(-distance / exp_scale)

    This reward encourages the robot to minimize distance to the goal position.
    The exponential form provides dense reward signal even at large distances.

    Args:
        robot: Robot articulation for state access.
        goal_pos_w: Goal position in world frame. Shape: (num_envs, 3).
        exp_scale: Scale factor for exponential decay (larger = slower decay).

    Returns:
        Reward values. Shape: (num_envs,).
    """
    distance = torch.linalg.norm(goal_pos_w - robot.data.root_pos_w, dim=1)
    return torch.exp(-distance / exp_scale)


def orientation_upright_exp(
    robot: Articulation,
    exp_scale: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Exponential upright orientation reward.

    Computes: exp(-tilt_angle / exp_scale)

    This reward penalizes tilting more aggressively than linear reward.
    At exp_scale=0.5 (default):
        - 0 deg tilt  -> reward = 1.0
        - 15 deg tilt -> reward = 0.59
        - 30 deg tilt -> reward = 0.35
        - 45 deg tilt -> reward = 0.21

    Args:
        robot: Robot articulation for state access.
        exp_scale: Scale factor for exponential decay (radians).

    Returns:
        Reward values. Shape: (num_envs,).
    """
    num_envs = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device

    # World up vector
    up_w = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(num_envs, -1)

    # Transform to body frame and check alignment
    up_b = quat_apply_inverse(robot.data.root_quat_w, up_w)

    # Compute tilt angle from z-component: cos(tilt) = up_b[:, 2]
    # Clamp to avoid numerical issues with acos
    cos_tilt = up_b[:, 2].clamp(-1.0, 1.0)
    tilt_angle = torch.acos(cos_tilt)  # radians

    return torch.exp(-tilt_angle / exp_scale)


def linear_velocity_penalty(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Linear velocity penalty.

    Penalizes high linear velocities in the body frame.
    Returns: sum(v^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation for state access.

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    return torch.sum(torch.square(robot.data.root_lin_vel_b), dim=1)


def angular_velocity_penalty(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Angular velocity penalty.

    Penalizes high angular velocities in the body frame.
    Returns: sum(omega^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation for state access.

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    return torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)


def action_rate_penalty(
    robot: Articulation,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Action rate penalty.

    Penalizes rapid changes in action to encourage smooth control.
    Returns: sum((a_t - a_{t-1})^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation (unused but required for signature).
        actions: Current actions. Shape: (num_envs, num_actions).
        prev_actions: Previous actions. Shape: (num_envs, num_actions).

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    action_rate = actions - prev_actions
    return torch.sum(torch.square(action_rate), dim=1)


def action_magnitude_penalty(
    robot: Articulation,
    actions: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Action magnitude penalty.

    Penalizes large action values to encourage energy efficiency.
    Returns: sum(a^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation (unused but required for signature).
        actions: Current actions. Shape: (num_envs, num_actions).

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    return torch.sum(torch.square(actions), dim=1)


def alive_bonus(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Alive bonus.

    Constant reward for surviving each timestep.
    Returns: 1.0 for all environments.

    Args:
        robot: Robot articulation for accessing num_envs and device.

    Returns:
        Constant reward values. Shape: (num_envs,).
    """
    num_envs = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device
    return torch.ones(num_envs, device=device)


