# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hover task for UUV environments.

This module implements the hover task where the underwater vehicle must
maintain a stable position at a randomly sampled goal location.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .task_base import TaskBase, TaskBaseCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class HoverTaskCfg(TaskBaseCfg):
    """Configuration for the hover task.

    The hover task requires the UUV to reach and maintain a target position.
    """

    # Goal position randomization range (relative to env origin) [x, y, z]
    goal_pos_range: tuple[float, float, float] = (2.0, 2.0, 1.0)

    # Initial height for goal z-component base
    initial_height: float = 2.0


class HoverTask(TaskBase):
    """Hover task implementation for UUV environments.

    The vehicle must reach and maintain a randomly sampled target position.
    Goal observations include the target position in the body frame.

    Observation Space (task contribution):
        - Goal position in body frame (3)
    """

    cfg: HoverTaskCfg

    def __init__(
        self,
        cfg: HoverTaskCfg,
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the hover task.

        Args:
            cfg: Hover task configuration.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
        super().__init__(cfg, num_envs, device)

    def _setup_buffers(self) -> None:
        """Setup hover task buffers."""
        # Goal position in world frame
        self._goal_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

    @property
    def goal_observation_dim(self) -> int:
        """Dimension of goal-related observations (goal position in body frame)."""
        return 3

    def get_goal_observations(
        self,
        robot: Articulation,
    ) -> torch.Tensor:
        """Compute goal position in body frame.

        Args:
            robot: Robot articulation for pose access.

        Returns:
            Goal position in body frame. Shape: (num_envs, 3).
        """
        goal_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            self._goal_pos_w,
        )
        return goal_pos_b

    def reset(
        self,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Reset goal positions for specified environments.

        Samples new random goal positions within the configured range.

        Args:
            env_ids: Environment indices to reset.
            env_origins: Origin positions for each environment.
        """
        num_reset = len(env_ids)
        goal_range = torch.tensor(self.cfg.goal_pos_range, device=self.device)

        # Sample random XY within range, centered on env origin
        self._goal_pos_w[env_ids, :2] = (torch.rand(num_reset, 2, device=self.device) * 2 - 1) * goal_range[
            :2
        ] + env_origins[env_ids, :2]

        # Sample Z within range, offset from initial height
        self._goal_pos_w[env_ids, 2] = (
            torch.rand(num_reset, device=self.device) * goal_range[2] + self.cfg.initial_height
        )

    def get_goal_state(self) -> dict[str, torch.Tensor]:
        """Get current goal state for visualization.

        Returns:
            Dictionary with 'goal_pos_w' tensor.
        """
        return {"goal_pos_w": self._goal_pos_w}

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """Goal position in world frame. Shape: (num_envs, 3)."""
        return self._goal_pos_w
