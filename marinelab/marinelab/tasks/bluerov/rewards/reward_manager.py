# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for UUV DirectRLEnv environments.

This module provides a lightweight reward manager adapted from Isaac Lab's
ManagerBasedRLEnv RewardManager for use in DirectRLEnv environments.

The design follows Isaac Lab conventions:
- Composition-based reward terms via RewardTermCfg
- Weighted summation of individual terms
- Automatic dt scaling for time-step independent rewards
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class RewardTermCfg:
    """Configuration for a reward term.

    Following Isaac Lab's ManagerTermBaseCfg pattern, each reward term
    consists of a callable function, a weight, and optional parameters.
    """

    func: Callable[..., torch.Tensor] = MISSING
    """The reward function to call.

    The function signature should be:
        func(robot: Articulation, **params) -> torch.Tensor

    where the returned tensor has shape (num_envs,).
    """

    weight: float = MISSING
    """The weight multiplier for this reward term.

    The final contribution is: weight * dt * func(robot, **params).
    Use negative weights for penalties.
    """

    params: dict[str, Any] = field(default_factory=dict)
    """Parameters passed as keyword arguments to the function."""

    enabled: bool = True
    """Whether this term is active. Disabled terms are skipped in computation."""


class RewardManager:
    """Lightweight reward manager for DirectRLEnv UUV environments.

    This manager computes the total reward as a weighted sum of individual
    reward terms. Each term is configured via RewardTermCfg and can be
    enabled/disabled at runtime.

    The manager follows Isaac Lab conventions:
    - Rewards are scaled by dt for time-step independence
    - Episode sums are tracked per term for logging
    - Terms can be functions or callable objects

    Attributes:
        num_envs: Number of parallel environments.
        device: Computation device.
        active_terms: List of active reward term names.
    """

    def __init__(
        self,
        cfg: dict[str, RewardTermCfg],
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the reward manager.

        Args:
            cfg: Dictionary mapping term names to RewardTermCfg.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
        self.num_envs = num_envs
        self.device = device

        # Parse term configurations
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []

        for name, term_cfg in cfg.items():
            if term_cfg.enabled:
                self._term_names.append(name)
                self._term_cfgs.append(term_cfg)

        # Initialize buffers
        self._reward_buf = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(num_envs, dtype=torch.float32, device=device) for name in self._term_names
        }

    def __str__(self) -> str:
        """String representation showing active terms and weights."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"

        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Weight"]
        table.align["Name"] = "l"
        table.align["Weight"] = "r"

        for idx, (name, cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([idx, name, f"{cfg.weight:.4f}"])

        msg += table.get_string()
        return msg

    @property
    def active_terms(self) -> list[str]:
        """List of active reward term names."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episode sums for each reward term."""
        return self._episode_sums

    def compute(
        self,
        robot: Articulation,
        dt: float,
        **context: Any,
    ) -> torch.Tensor:
        """Compute total reward as weighted sum of terms.

        Args:
            robot: Robot articulation for state access.
            dt: Time step duration for scaling.
            **context: Additional context passed to reward functions
                (e.g., actions, prev_actions, goal_pos_w).

        Returns:
            Total reward tensor. Shape: (num_envs,).
        """
        self._reward_buf.zero_()

        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_cfg.weight == 0.0:
                continue

            # Call the reward function with robot and merged params
            merged_params = {**term_cfg.params, **context}
            term_value = term_cfg.func(robot, **merged_params)

            # Apply weight and dt scaling
            scaled_value = term_value * term_cfg.weight * dt

            # Accumulate
            self._reward_buf += scaled_value
            self._episode_sums[name] += scaled_value

        return self._reward_buf

    def reset(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reset episode sums for specified environments.

        Args:
            env_ids: Environment indices to reset.

        Returns:
            Dictionary of episode sums before reset (for logging).
        """
        # Capture sums before reset for logging
        sums_before_reset = {name: self._episode_sums[name][env_ids].mean().item() for name in self._term_names}

        # Reset sums
        for name in self._term_names:
            self._episode_sums[name][env_ids] = 0.0

        return sums_before_reset
