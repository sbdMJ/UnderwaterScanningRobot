# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Ocean current as a standalone, injectable component.

A single OceanCurrent instance can be shared by multiple HydrodynamicsModel
instances (e.g. a vehicle body and a buoyancy body) so they see the same flow
field without manual buffer copying. Time-varying current (e.g. OU drift) is
injected by the caller via set() / add_drift(); the framework stores the field
and exposes it for relative-velocity computation and clamping.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class OceanCurrent:
    """World-frame 6-DOF ocean current field shared across hydrodynamics models.

    Attributes:
        num_envs: Number of parallel environments.
        device: Computation device.
    """

    def __init__(self, num_envs: int, device: str, cfg) -> None:
        """Initialize the current field to zero.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            cfg: Object exposing `max_velocity` (len-6) and `noise_scale` (len-6).
        """
        self.num_envs = num_envs
        self.device = device
        kw = {"dtype": torch.float32, "device": device}
        self._velocity_w = torch.zeros(num_envs, 6, **kw)
        self._max_velocity = torch.tensor(cfg.max_velocity, **kw)
        self._noise_scale = torch.tensor(cfg.noise_scale, **kw)

    @property
    def velocity_w(self) -> torch.Tensor:
        """Current velocity in world frame. Shape: (num_envs, 6)."""
        return self._velocity_w

    @property
    def max_velocity(self) -> torch.Tensor:
        """Configured maximum current velocity. Shape: (6,)."""
        return self._max_velocity

    def set(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
        strength: torch.Tensor | None = None,
    ) -> None:
        """Set current velocity for specified environments.

        Args:
            env_ids: Environment indices to update.
            velocity: Explicit per-env (n, 6) velocity. If None, sampled uniformly
                from [-max, +max] plus optional gaussian noise.
            strength: Optional per-env scale in [0, 1] applied to the sampled
                velocity (ignored when `velocity` is given).
        """
        env_ids = self._as_index(env_ids)
        if velocity is None:
            n = len(env_ids)
            rand = torch.rand(n, 6, device=self.device)
            velocity = rand * self._max_velocity * 2 - self._max_velocity
            if self._noise_scale.any():
                velocity = velocity + torch.randn(n, 6, device=self.device) * self._noise_scale
            if strength is not None:
                velocity = velocity * strength.unsqueeze(-1)
        self._velocity_w[env_ids] = velocity

    def add_drift(self, delta: torch.Tensor) -> None:
        """Add an increment to the current field (for time-varying current).

        Args:
            delta: (num_envs, 6) increment to add to velocity_w.
        """
        self._velocity_w = self._velocity_w + delta

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Zero the current for specified environments (all if None)."""
        if env_ids is None:
            self._velocity_w.zero_()
        else:
            self._velocity_w[self._as_index(env_ids)] = 0.0

    def _as_index(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            return torch.tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids
