# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attitude control task for UUV environments.

This module implements the attitude task where the underwater vehicle must
maintain a target orientation without position control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .task_base import TaskBase, TaskBaseCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class AttitudeTaskCfg(TaskBaseCfg):
    """Configuration for the attitude control task.

    The attitude task requires the UUV to achieve and maintain a target orientation.
    Position is not controlled.
    """

    # Target attitude randomization range (radians) [roll, pitch, yaw]
    target_attitude_range: tuple[float, float, float] = (0.5, 0.5, 3.14159)

    # Base (nominal) target attitude [roll, pitch, yaw] - added to random offset
    base_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Whether to randomize target attitude on reset
    randomize_target: bool = True


class AttitudeTask(TaskBase):
    """Attitude control task implementation for UUV environments.

    The vehicle must achieve and maintain a target orientation.
    This task is useful for:
        - Testing attitude control without position coupling
        - Pre-training attitude stabilization before full position control
        - Applications where only orientation matters (camera pointing, etc.)

    Observation Space (task contribution):
        - Target attitude error in body frame (3) - roll, pitch, yaw errors
    """

    cfg: AttitudeTaskCfg

    def __init__(
        self,
        cfg: AttitudeTaskCfg,
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the attitude task.

        Args:
            cfg: Attitude task configuration.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
        super().__init__(cfg, num_envs, device)

    def _setup_buffers(self) -> None:
        """Setup attitude task buffers."""
        # Target quaternion in world frame
        self._target_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._target_quat_w[:, 0] = 1.0  # Identity quaternion (w, x, y, z)

        # Target Euler angles for reference
        self._target_euler = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

    @property
    def goal_observation_dim(self) -> int:
        """Dimension of goal-related observations (attitude error)."""
        return 3

    def get_goal_observations(
        self,
        robot: Articulation,
    ) -> torch.Tensor:
        """Compute attitude error in body frame.

        Returns the difference between current and target orientation
        as roll, pitch, yaw errors.

        Args:
            robot: Robot articulation for pose access.

        Returns:
            Attitude error in body frame. Shape: (num_envs, 3).
        """
        # Compute relative quaternion: q_error = q_target^-1 * q_current
        # This gives the rotation needed to go from current to target
        current_quat = robot.data.root_quat_w  # (w, x, y, z)

        # Quaternion inverse: q^-1 = (w, -x, -y, -z) for unit quaternion
        target_quat_inv = self._target_quat_w.clone()
        target_quat_inv[:, 1:4] *= -1

        # Error quaternion: how much to rotate from current to target
        error_quat = quat_mul(target_quat_inv, current_quat)

        # Convert to axis-angle error. Sign-correct by the scalar part so the
        # error stays continuous across the quaternion double cover (q and -q
        # are the same rotation); without this the signal flips/shrinks near
        # 180 deg where the gradient matters most.
        w = error_quat[:, 0:1]
        vec = error_quat[:, 1:4]
        attitude_error = 2.0 * torch.sign(w) * vec

        return attitude_error

    def reset(
        self,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Reset target attitudes for specified environments.

        Args:
            env_ids: Environment indices to reset.
            env_origins: Origin positions for each environment (not used for attitude).
        """
        num_reset = len(env_ids)

        if self.cfg.randomize_target:
            # Sample random target attitude within range
            attitude_range = torch.tensor(self.cfg.target_attitude_range, device=self.device)
            base_attitude = torch.tensor(self.cfg.base_attitude, device=self.device)

            random_offset = (torch.rand(num_reset, 3, device=self.device) * 2 - 1) * attitude_range
            target_euler = base_attitude + random_offset
        else:
            # Use fixed base attitude
            target_euler = torch.tensor(self.cfg.base_attitude, device=self.device).unsqueeze(0).repeat(num_reset, 1)

        # Store Euler angles
        self._target_euler[env_ids] = target_euler

        # Convert to quaternion
        roll = target_euler[:, 0]
        pitch = target_euler[:, 1]
        yaw = target_euler[:, 2]
        self._target_quat_w[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

    def get_goal_state(self) -> dict[str, torch.Tensor]:
        """Get current goal state for visualization.

        Returns:
            Dictionary with target attitude tensors.
        """
        return {
            "target_quat_w": self._target_quat_w,
            "target_euler": self._target_euler,
        }

    @property
    def target_quat_w(self) -> torch.Tensor:
        """Target quaternion in world frame. Shape: (num_envs, 4)."""
        return self._target_quat_w

    @property
    def target_euler(self) -> torch.Tensor:
        """Target Euler angles (roll, pitch, yaw). Shape: (num_envs, 3)."""
        return self._target_euler
