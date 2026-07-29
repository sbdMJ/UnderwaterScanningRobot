# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for UUV task definitions.

This module provides an abstract base class for defining tasks in UUV environments.
Tasks encapsulate goal generation, observation composition, and task-specific
termination conditions.

The design follows Isaac Lab's Manager patterns adapted for DirectRLEnv usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class TaskBaseCfg:
    """Base configuration for UUV tasks.

    This configuration class defines the common interface for all task configurations.
    Specific tasks should inherit from this and add task-specific parameters.
    """

    # Goal position randomization range (relative to spawn) [x, y, z]
    goal_pos_range: tuple[float, float, float] = (2.0, 2.0, 1.0)

    # Initial height for goal z-component base
    initial_height: float = 2.0


class TaskBase(ABC):
    """Abstract base class for UUV task definitions.

    Tasks define the goal structure, observation composition, and task-specific
    termination conditions. This abstraction allows easy addition of new tasks
    (e.g., hover, track, dock) without modifying the core environment.

    Attributes:
        cfg: Task configuration.
        num_envs: Number of parallel environments.
        device: Computation device.
    """

    def __init__(
        self,
        cfg: TaskBaseCfg,
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the task.

        Args:
            cfg: Task configuration.
            num_envs: Number of parallel environments.
            device: Computation device (cpu or cuda).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Initialize goal buffers (subclasses should populate in _setup_buffers)
        self._setup_buffers()

    @abstractmethod
    def _setup_buffers(self) -> None:
        """Setup task-specific buffers.

        Subclasses should override this to initialize goal states and any
        task-specific data structures.
        """
        pass

    @abstractmethod
    def get_goal_observations(
        self,
        robot: Articulation,
    ) -> torch.Tensor:
        """Compute goal-related observations.

        Args:
            robot: Robot articulation for state access.

        Returns:
            Goal observations tensor. Shape: (num_envs, goal_obs_dim).
        """
        pass

    @property
    @abstractmethod
    def goal_observation_dim(self) -> int:
        """Dimension of goal-related observations."""
        pass

    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Reset task state for specified environments.

        Args:
            env_ids: Environment indices to reset.
            env_origins: Origin positions for each environment.
        """
        pass

    def get_terminations(self, robot: Articulation) -> torch.Tensor:
        """Compute task-specific termination conditions.

        Default implementation returns no terminations (all False).
        Override in subclasses for task-specific terminations.

        Args:
            robot: Robot articulation for state access.

        Returns:
            Termination flags. Shape: (num_envs,).
        """
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def get_goal_state(self) -> dict[str, torch.Tensor]:
        """Get current goal state for visualization and logging.

        Returns:
            Dictionary containing goal state tensors.
        """
        return {}
