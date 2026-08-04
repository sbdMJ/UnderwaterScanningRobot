# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""E2/E3 condition variants, applied WITHOUT touching the existing task code.

E2 (robustness sweep): the fluid-coefficient randomization half-range of the stress-DR
Eval cfg is re-scaled on the cfg INSTANCE before ``gym.make`` — no new cfg classes, no
new gym IDs. Only the three fluid-coefficient scales sweep (plan: ±25/±50/±75%);
attitude (±45°) and sensor-mount (±8 cm) stress stays at the Eval values.

E3 (ocean current): ``HydrodynamicsModel`` always carries an ``OceanCurrent`` component
(zero by default — the wallscan cfg simply never drives it), so a time profile is
injected by writing the shared component per control step via ``OceanCurrent.set``.
No env subclass, no cfg field: the driver holds the component and a waveform.

Both helpers are pure (duck-typed cfg/component objects) and natively unit-tested.
"""
from __future__ import annotations

import math

import torch


def apply_fluid_dr_scale(cfg, scale: float):
    """Set the fluid-coefficient randomization half-range to ±scale (E2 sweep axis).

    Mutates ``cfg.randomization`` in place and returns cfg. Everything else in the
    randomization (attitude, thruster, CoB/CoG, volume) keeps the task's values.
    """
    if not 0.0 < scale < 1.0:
        raise ValueError(f"fluid DR scale must be in (0, 1); got {scale}")
    rnd = cfg.randomization
    rnd.added_mass_scale = (1.0 - scale, 1.0 + scale)
    rnd.linear_damping_scale = (1.0 - scale, 1.0 + scale)
    rnd.quadratic_damping_scale = (1.0 - scale, 1.0 + scale)
    return cfg


class CurrentDriver:
    """Drives a shared OceanCurrent component with a time profile (E3 conditions).

    Profiles (plan E3):
      - ``step``: constant current that reverses direction at ``t_switch`` s (the abrupt
        change — SSI-MPC's ground-effect analogue). ``mode: onset`` starts at zero and
        switches ON instead.
      - ``sine``: sinusoidal magnitude along a fixed heading, period ``period`` s.
    """

    def __init__(self, current, num_envs: int, device, profile: dict):
        self._current = current
        self._env_ids = torch.arange(num_envs, device=device)
        self._device = device
        self.kind = profile["type"]
        if self.kind not in ("step", "sine"):
            raise ValueError(f"unknown current profile {self.kind!r}")
        speed = float(profile.get("speed", 0.15))
        heading = math.radians(float(profile.get("heading_deg", 0.0)))
        self._v_dir = torch.zeros(6, device=device)
        self._v_dir[0] = speed * math.cos(heading)
        self._v_dir[1] = speed * math.sin(heading)
        self.t_switch = float(profile.get("t_switch", 60.0))
        self.period = float(profile.get("period", 30.0))
        self.mode = profile.get("mode", "reverse")

    def velocity_at(self, t: float) -> torch.Tensor:
        """(6,) world-frame current velocity at time t."""
        if self.kind == "step":
            if self.mode == "onset":
                gain = 0.0 if t < self.t_switch else 1.0
            else:  # reverse: full current that flips sign at t_switch
                gain = 1.0 if t < self.t_switch else -1.0
        else:  # sine
            gain = math.sin(2.0 * math.pi * t / self.period)
        return self._v_dir * gain

    def apply(self, t: float) -> None:
        v = self.velocity_at(t).unsqueeze(0).expand(len(self._env_ids), -1)
        self._current.set(self._env_ids, velocity=v)

    @classmethod
    def from_options(cls, opt: dict, env) -> "CurrentDriver | None":
        """Build from a cell's options (key ``current``); None when the cell has none."""
        profile = opt.get("current")
        if not profile:
            return None
        return cls(env._hydro._current, env.num_envs, env.device, profile)
