# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thruster model for underwater vehicles.

This module provides a modular thruster system implementation that converts
thruster commands to body forces and torques using an allocation matrix.
"""

from __future__ import annotations

import torch

# Import configuration class from isaaclab_assets
from marinelab.assets import ThrusterCfg

__all__ = ["ThrusterModel", "ThrusterCfg", "allocation_moment_residual"]


def allocation_moment_residual(allocation_matrix) -> torch.Tensor:
    """Per-thruster violation of ``tau . F_hat == 0`` in a thruster allocation matrix.

    A thruster's moment about the body origin is ``tau = r x F`` for SOME mounting position
    ``r``, and a cross product is always perpendicular to its operands. So for every column
    of a physically realizable TAM::

        (Mx, My, Mz) . normalize(Fx, Fy, Fz) == 0

    independently of where the thruster sits or how it is canted. That makes this a
    geometry-free invariant: a nonzero residual means the column cannot be produced by any
    thruster placement, i.e. the matrix is mis-transcribed.

    This exists because ``PKRCThrusterCfg`` shipped with exactly that defect — the sway
    thrusters' 0.09 moment arm sat in the ``My`` row while their force is pure ``+y``, and a
    pure ``+y`` force has ``tau_y == 0`` for any ``r``. Consequences were measured, not
    guessed (2026-07-30, ``isaaclab/logs/_probe_tam_tilt.py``): the mis-assignment turned an
    8.6 N-cancellable roll into an uncancellable 2.3 deg pitch. See
    :class:`~marinelab.assets.pkrc.PKRCThrusterCfgFixedTAM`.

    Args:
        allocation_matrix: (6, num_thrusters) array-like, rows ``[Fx, Fy, Fz, Mx, My, Mz]``.

    Returns:
        (num_thrusters,) signed residuals. Zero-force columns return the moment norm, since
        a moment with no force behind it is equally unphysical.
    """
    B = torch.as_tensor(allocation_matrix, dtype=torch.float64)
    if B.ndim != 2 or B.shape[0] != 6:
        raise ValueError(f"allocation_matrix must have shape (6, n); got {tuple(B.shape)}")
    force, moment = B[:3].T, B[3:].T  # (n, 3) each
    norm = force.norm(dim=-1)
    dead = norm < 1e-12
    residual = (moment * (force / norm.clamp(min=1e-12).unsqueeze(-1))).sum(-1)
    return torch.where(dead, moment.norm(dim=-1), residual)


class ThrusterModel:
    """Thruster model for underwater vehicles.

    This model implements first-order thruster dynamics with configurable
    time constants and an allocation matrix for converting thruster commands
    to body forces and torques.

    Attributes:
        cfg: Thruster configuration.
        num_envs: Number of parallel environments.
        device: Computation device.
    """

    def __init__(
        self,
        cfg: ThrusterCfg,
        num_envs: int,
        device: str,
        enable_randomization: bool = False,
        enable_fault: bool = False,
        enable_actuation_noise: bool = False,
    ) -> None:
        """Initialize the thruster model.

        Args:
            cfg: Thruster configuration.
            num_envs: Number of parallel environments.
            device: Computation device.
            enable_randomization: Whether to enable per-environment parameter storage.
            enable_fault: Whether to enable per-env per-thruster fault (health) injection.
                When False, no health buffer is allocated and ``compute_wrench`` skips the
                health multiply entirely (byte-identical to the fault-free model).
            enable_actuation_noise: Whether to arm the per-step multiplicative command
                noise (the THIRD actuation channel, independent of randomization/fault).
                When False, ``_act_noise_std`` is None and ``apply_dynamics`` is
                byte-identical (no RNG consumed).
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Setup buffers
        self._setup_buffers()

        # Setup allocation matrix
        self._allocation_matrix = torch.tensor(
            cfg.allocation_matrix, dtype=torch.float32, device=device
        )

        # Precompute time-constant tensors once (avoid per-step allocation).
        self._tau_up_scalar = torch.tensor(cfg.time_constant_up, dtype=torch.float32, device=device)
        self._tau_down_scalar = torch.tensor(cfg.time_constant_down, dtype=torch.float32, device=device)

        # Setup per-environment parameters if randomization enabled
        if enable_randomization:
            self._thrust_coeff = torch.full((num_envs,), cfg.thrust_coefficient, device=device)
            self._time_constant_up = torch.full((num_envs,), cfg.time_constant_up, device=device)
            self._time_constant_down = torch.full((num_envs,), cfg.time_constant_down, device=device)
        else:
            self._thrust_coeff = None
            self._time_constant_up = None
            self._time_constant_down = None

        # Per-env per-thruster health in [0, 1] (0 = dead, 1 = nominal). Distinct from
        # DR: a fault models an actuator FAILING, not a physical-parameter spread. Only
        # allocated when fault injection is on; None -> compute_wrench skips the multiply.
        if enable_fault:
            self._thruster_health = torch.ones((num_envs, cfg.num_thrusters), device=device)
        else:
            self._thruster_health = None

        # Per-step multiplicative command noise std (white). Distinct axis from DR
        # (_thrust_coeff, reset) and fault (_thruster_health, reset): this perturbs the
        # COMMAND every step. None when disabled -> apply_dynamics skips the multiply
        # (byte-identical). getattr guards cfgs that predate the field.
        if enable_actuation_noise:
            self._act_noise_std = getattr(cfg, "actuation_noise_std", 0.0)
        else:
            self._act_noise_std = None

    def _setup_buffers(self) -> None:
        """Setup internal buffers."""
        self._state = torch.zeros(self.num_envs, self.cfg.num_thrusters, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_torques = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def state(self) -> torch.Tensor:
        """Current thruster state (normalized). Shape: (num_envs, num_thrusters)."""
        return self._state

    @property
    def body_forces(self) -> torch.Tensor:
        """Computed body forces. Shape: (num_envs, 3)."""
        return self._body_forces

    @property
    def body_torques(self) -> torch.Tensor:
        """Computed body torques. Shape: (num_envs, 3)."""
        return self._body_torques

    def apply_dynamics(
        self,
        commands: torch.Tensor,
        dt: float,
    ) -> None:
        """Apply first-order thruster dynamics to commands.

        Args:
            commands: Raw thruster commands. Shape: (num_envs, num_thrusters).
            dt: Time step duration.
        """
        # Per-step multiplicative actuation noise (white). No-op when disabled:
        # _act_noise_std is None -> commands passes through unchanged (byte-identical).
        # Applied PRE-clamp: near saturation, positive noise is asymmetrically clipped
        # by the clamp below (real actuator can't push past its limit) -- intended.
        if self._act_noise_std is not None and self._act_noise_std > 0.0:
            noise = torch.randn_like(commands) * self._act_noise_std
            commands = commands * (1.0 + noise)
        target_state = commands.clamp(-1.0, 1.0)

        if self._thrust_coeff is not None:
            tau_up = self._time_constant_up.unsqueeze(-1)
            tau_down = self._time_constant_down.unsqueeze(-1)
        else:
            tau_up = self._tau_up_scalar
            tau_down = self._tau_down_scalar

        tau = torch.where(target_state > self._state, tau_up, tau_down)

        alpha = dt / tau
        self._state = self._state + alpha * (target_state - self._state)

    def _thrust_command(self) -> torch.Tensor:
        """Map normalized thruster state [-1, 1] to an effective command.

        When ``cfg.enable_thrust_curve`` is False this is the identity (the
        state itself), so ``compute_wrench`` stays byte-identical to the linear
        model. When True it applies a BlueROV T200 fidelity model in normalized
        command space:

        - Deadband: ``|state| < cfg.thrust_deadband`` -> 0 (the real T200 ESC
          produces zero thrust in a +/-25 us PWM deadzone around neutral).
        - Signed-square curve: outside the deadband the command follows
          ``sign(state) * state**2``, matching the quadratic propeller law
          (thrust ~ omega^2, omega ~ command) instead of a linear map.

        The output stays in [-1, 1] (``|state| <= 1`` => ``state**2 <= 1``), so
        the downstream ``* thrust_coefficient`` scaling is unchanged in range.
        """
        if not getattr(self.cfg, "enable_thrust_curve", False):
            return self._state
        deadband = getattr(self.cfg, "thrust_deadband", 0.075)
        active = self._state.abs() > deadband
        curved = torch.sign(self._state) * self._state.square()
        return torch.where(active, curved, torch.zeros_like(self._state))

    def compute_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute body forces and torques from current thruster state.

        Returns:
            Tuple of (forces, torques) tensors. Each shape: (num_envs, 3).
        """
        command = self._thrust_command()
        if self._thrust_coeff is not None:
            thrust_magnitude = command * self._thrust_coeff.unsqueeze(-1)
        else:
            thrust_magnitude = command * self.cfg.thrust_coefficient

        # Apply per-thruster fault health BEFORE clamping: a degraded thruster has a
        # reduced peak force, so health scales the unsaturated magnitude (health=0.5
        # halves output even in saturation). Skipped entirely when fault is disabled.
        if self._thruster_health is not None:
            thrust_magnitude = thrust_magnitude * self._thruster_health

        thrust_magnitude = thrust_magnitude.clamp(-self.cfg.max_thrust, self.cfg.max_thrust)

        body_wrench = torch.einsum("ij,nj->ni", self._allocation_matrix, thrust_magnitude)

        self._body_forces = body_wrench[:, :3]
        self._body_torques = body_wrench[:, 3:]

        return self._body_forces, self._body_torques

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset thruster state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._state[env_ids] = 0.0

    def set_thruster_health(self, env_ids: torch.Tensor, health: torch.Tensor) -> None:
        """Inject per-env per-thruster health (fault) for the given envs.

        Mirrors the ``randomize_parameters`` lifecycle: called at reset time for the
        envs being reset, writing only their rows. A fault is NOT cleared by an
        episode reset on its own -- the events layer resamples it at reset, exactly
        like domain randomization.

        Args:
            env_ids: Environment indices to set. Shape: (M,).
            health: Health values in [0, 1]. Shape: (M, num_thrusters); 0 = dead,
                1 = nominal, intermediate = partial degradation.

        No-op when fault injection is disabled (buffer is None), so callers need not
        know the toggle state.
        """
        if self._thruster_health is None:
            return
        self._thruster_health[env_ids] = health.to(self._thruster_health.dtype)

    def randomize_parameters(
        self,
        env_ids: torch.Tensor,
        thrust_coeff_scale: tuple[float, float] | torch.Tensor = (0.8, 1.2),
        time_constant_scale: tuple[float, float] | torch.Tensor = (0.8, 1.2),
    ) -> None:
        """Randomize thruster parameters for specified environments.

        Args:
            env_ids: Environment indices to randomize.
            thrust_coeff_scale: Scale range for thrust coefficient as tuple `(lo, hi)` or per-env scale tensor `[n]`.
            time_constant_scale: Scale range for time constants as tuple `(lo, hi)` or per-env scale tensor `[n]`.
        """
        if self._thrust_coeff is None:
            return

        num_envs = len(env_ids)

        def _scale(value: tuple[float, float] | torch.Tensor) -> torch.Tensor:
            # (lo, hi) tuple -> sample uniform (current behavior); tensor [n] -> use directly.
            if isinstance(value, torch.Tensor):
                return value.to(device=self.device, dtype=self._thrust_coeff.dtype)
            lo, hi = value
            return torch.rand(num_envs, device=self.device) * (hi - lo) + lo

        tc_scale = _scale(thrust_coeff_scale)
        self._thrust_coeff[env_ids] = self.cfg.thrust_coefficient * tc_scale

        time_scale = _scale(time_constant_scale)
        self._time_constant_up[env_ids] = self.cfg.time_constant_up * time_scale
        self._time_constant_down[env_ids] = self.cfg.time_constant_down * time_scale
