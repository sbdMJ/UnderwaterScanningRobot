# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fossen model hydrodynamics for underwater vehicles.

This module implements the 6-DOF hydrodynamic forces and torques based on
the Fossen model for marine craft dynamics.

Reference:
    Fossen, T. I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

# Import configuration classes from isaaclab_assets
from marinelab.assets import HydrodynamicsCfg, OceanCurrentCfg

from .ocean_current import OceanCurrent
from .parameters import HydroParams, default_rigid_inertia

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["HydrodynamicsModel"]


class HydrodynamicsModel:
    """Fossen model hydrodynamics calculator for underwater vehicles.

    This class computes hydrodynamic forces and torques acting on an underwater
    vehicle based on the Fossen model. The forces include:
        - Added mass effects (inertia from accelerating surrounding fluid)
        - Linear and quadratic damping (drag forces)
        - Coriolis and centripetal forces (coupling between linear/angular motion)
        - Buoyancy and restoring forces (orientation-dependent)

    The model assumes:
        - 6-DOF rigid body dynamics
        - Diagonal added mass and damping matrices (simplified)
        - Constant density fluid (incompressible)
        - No wave or surface effects

    Attributes:
        num_envs: Number of parallel environments.
        device: Computation device (cpu or cuda).
        cfg: Hydrodynamics configuration.
        current_cfg: Ocean current configuration.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        cfg: HydrodynamicsCfg,
        current_cfg: OceanCurrentCfg | None = None,
        dt: float = 0.01,
        articulation_prim_path: str | None = None,
        current: OceanCurrent | None = None,
    ) -> None:
        """Initialize the hydrodynamics model.

        This model computes hydrodynamic forces for underwater vehicles.
        Weight (gravity) is handled by PhysX, so this model only applies
        buoyancy as an external upward force.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            cfg: Hydrodynamics configuration.
            current_cfg: Ocean current configuration. Defaults to no current.
            dt: Simulation timestep for acceleration calculation.
            articulation_prim_path: USD path to articulation root for auto volume calculation.
                Only used if cfg.volume is None.
            current: Shared OceanCurrent component to inject. If None, the model
                builds its own from current_cfg (backward compatible).
        """
        # Store basic parameters
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.dt = dt

        # Common tensor creation kwargs
        self._tensor_kwargs = {"dtype": torch.float32, "device": device}

        # Ocean current: use injected shared component, or build one from cfg.
        if current is not None:
            self._current = current
        else:
            self._current = OceanCurrent(num_envs, device, current_cfg or OceanCurrentCfg())

        # Initialize components
        self._init_hydrodynamic_matrices(cfg)
        self._init_buoyancy_params(cfg, articulation_prim_path)
        self._init_state_buffers()
        self._base_parameters = self._snapshot_parameters()

    def _init_hydrodynamic_matrices(self, cfg: HydrodynamicsCfg) -> None:
        """Initialize added mass and damping matrices.

        Args:
            cfg: Hydrodynamics configuration.
        """
        # Added mass matrix (6x6 diagonal)
        added_mass_diag = torch.diag(torch.tensor(cfg.added_mass, **self._tensor_kwargs))
        self._added_mass_matrix = added_mass_diag.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Damping coefficients (linear and quadratic)
        self._linear_damping_diag = self._broadcast(torch.tensor(cfg.linear_damping, **self._tensor_kwargs))
        self._quadratic_damping_diag = self._broadcast(torch.tensor(cfg.quadratic_damping, **self._tensor_kwargs))

        # Rigid body inertia for Coriolis matrix
        inertia = torch.tensor(default_rigid_inertia(cfg), **self._tensor_kwargs)
        self._rigid_body_inertia = self._broadcast(inertia)
        # Non-DR'd copy for damping stability clamp.
        # DR randomizes _rigid_body_inertia (hydro model uncertainty), but PhysX's actual
        # inertia is NOT randomized by inertia_scale (only by body_mass_scale via set_masses).
        # Using the DR'd inertia in the clamp allows excessive damping when inertia_scale > 1,
        # causing velocity reversal. This buffer uses the base config inertia (always safe).
        self._clamp_inertia = self._broadcast(inertia)
        self._use_full_coriolis = cfg.use_full_coriolis
        if self._use_full_coriolis:
            warnings.warn(
                f"HydrodynamicsModel({cfg.body_name}): use_full_coriolis=True computes C_RB internally. "
                "Ensure enable_gyroscopic_forces=False in RigidBodyPropertiesCfg to avoid double-counting "
                "rigid body gyroscopic effects.",
                stacklevel=2,
            )

        self._validate_added_mass_stability(cfg)

        # Added mass force settings
        self._apply_added_mass = cfg.apply_added_mass_force
        self._am_stability_factor = cfg.added_mass_stability_factor

        # Off-diagonal damping cross-coupling
        self._damping_cross_coupling = cfg.damping_cross_coupling

        # Semi-implicit damping stability clamp
        self._damping_stability_factor = cfg.damping_stability_factor

    def _init_buoyancy_params(self, cfg: HydrodynamicsCfg, articulation_prim_path: str | None) -> None:
        """Initialize buoyancy related parameters.

        Args:
            cfg: Hydrodynamics configuration.
            articulation_prim_path: USD path to articulation root for auto volume calculation.
        """
        self._water_density = torch.full((self.num_envs,), cfg.water_density, **self._tensor_kwargs)
        self._gravity = 9.81

        # Volume and buoyancy force
        volume_value = self._resolve_volume(cfg, articulation_prim_path)
        self._volume = torch.full((self.num_envs,), volume_value, **self._tensor_kwargs)
        self._buoyancy_force_base = self._water_density * self._gravity * self._volume

        # Center of buoyancy/gravity in body frame
        self._r_cb = self._broadcast(torch.tensor(cfg.center_of_buoyancy, **self._tensor_kwargs))
        self._r_cg = self._broadcast(torch.tensor(cfg.center_of_gravity, **self._tensor_kwargs))

        # Nominal CoG and body mass for gravity restoring moment correction.
        # When CoG is randomized away from nominal, a correction torque is applied:
        #   M_correction = (r_cg - r_cg_nominal) x F_weight_body
        # This is zero when r_cg equals the nominal (URDF/PhysX) value.
        self._r_cg_nominal = torch.tensor(cfg.center_of_gravity, **self._tensor_kwargs)
        if cfg.body_mass is not None:
            self._body_mass: torch.Tensor | None = torch.full((self.num_envs,), cfg.body_mass, **self._tensor_kwargs)
        else:
            self._body_mass = None

    def _init_state_buffers(self) -> None:
        """Initialize cached buffers (current lives in the OceanCurrent component)."""
        # PhysX acceleration cache (body frame, updated via update_physx_state)
        self._physx_acc_b = torch.zeros(self.num_envs, 6, **self._tensor_kwargs)

        # Constant world-up direction, cached to avoid per-step allocation.
        self._up_dir_w = torch.zeros(self.num_envs, 3, **self._tensor_kwargs)
        self._up_dir_w[:, 2] = 1.0

    def _validate_added_mass_stability(self, cfg: HydrodynamicsCfg) -> None:
        """Validate added mass stability constraint: M_a[i] / I_rigid[i] < 1.0.

        Checks that added mass does not exceed rigid body inertia on any axis,
        which would cause forward Euler integration instability.

        Args:
            cfg: Hydrodynamics configuration.
        """
        if cfg.apply_added_mass_force and cfg.body_mass is None:
            warnings.warn(
                f"HydrodynamicsModel({cfg.body_name}): apply_added_mass_force=True but body_mass is None. "
                "Added mass stability validation skipped. Set body_mass for safety checks.",
                stacklevel=4,
            )
        if cfg.apply_added_mass_force and cfg.body_mass is not None:
            am = torch.tensor(cfg.added_mass, **self._tensor_kwargs)
            rot_inertia = default_rigid_inertia(cfg)
            gen_inertia = torch.tensor(
                [cfg.body_mass, cfg.body_mass, cfg.body_mass] + rot_inertia,
                **self._tensor_kwargs,
            )
            axis_names = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
            for i in range(6):
                if gen_inertia[i] > 0:
                    ratio = am[i].item() / gen_inertia[i].item()
                    if ratio >= 1.0:
                        raise ValueError(
                            f"HydrodynamicsModel({cfg.body_name}): Added mass stability violated on axis "
                            f"'{axis_names[i]}': M_a={am[i].item():.4f} / I_rigid={gen_inertia[i].item():.4f} "
                            f"= {ratio:.2f} >= 1.0. Reduce added_mass or increase body mass/inertia."
                        )
                    if ratio > 0.8:
                        warnings.warn(
                            f"HydrodynamicsModel({cfg.body_name}): Marginal added mass stability on axis "
                            f"'{axis_names[i]}': M_a/I_rigid = {ratio:.2f} (threshold=0.8). "
                            f"Consider reducing added_mass[{i}] or using a lower stability_factor.",
                            stacklevel=4,
                        )

    def _to_env_ids(self, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Convert env_ids to tensor format.

        Args:
            env_ids: Environment indices as tensor, list, or tuple.

        Returns:
            Environment indices as a long tensor on the correct device.
        """
        if not isinstance(env_ids, torch.Tensor):
            return torch.tensor(env_ids, dtype=torch.long, device=self.device)
        return env_ids

    def _broadcast(self, val: torch.Tensor) -> torch.Tensor:
        """Broadcast a single-env tensor to (num_envs, ...) with contiguous memory.

        Args:
            val: Tensor to broadcast (e.g. shape (3,) or (6,)).

        Returns:
            Contiguous tensor of shape (num_envs, *val.shape).
        """
        return val.unsqueeze(0).expand(self.num_envs, *val.shape).contiguous()

    def update_physx_state(
        self,
        body_com_acc_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> None:
        """Update cached PhysX acceleration after physics step.

        This method should be called after robot.update() to cache the acceleration
        computed by PhysX. Required for hybrid added mass force calculation.

        Args:
            body_com_acc_w: Body center of mass acceleration in world frame.
                Shape: (num_envs, num_bodies, 6) or (num_envs, 6).
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).
        """
        if not self._apply_added_mass:
            return

        # Handle both (num_envs, num_bodies, 6) and (num_envs, 6) shapes
        if body_com_acc_w.dim() == 3:
            acc_w = body_com_acc_w[:, 0, :]  # Root body acceleration
        else:
            acc_w = body_com_acc_w

        # Transform world frame acceleration to body frame
        lin_acc_b = quat_apply_inverse(root_quat_w, acc_w[:, :3])
        ang_acc_b = quat_apply_inverse(root_quat_w, acc_w[:, 3:])
        self._physx_acc_b = torch.cat([lin_acc_b, ang_acc_b], dim=-1)

    def _resolve_volume(self, cfg: HydrodynamicsCfg, articulation_prim_path: str | None) -> float:
        """Resolve volume with fallback: config > auto-calc from collision geometry > default 0.01 m^3.

        Args:
            cfg: Hydrodynamics configuration.
            articulation_prim_path: USD path to articulation root for auto volume calculation.

        Returns:
            Resolved volume in cubic metres.
        """
        if cfg.volume is not None:
            return cfg.volume

        fallback = 0.01  # default volume (m^3) when neither config nor auto-calc yields a value

        if articulation_prim_path is not None:
            from marinelab.core.volume import compute_collision_volume

            body_path = f"{articulation_prim_path}/{cfg.body_name}"
            volume = compute_collision_volume(body_path)
            if volume > 0:
                return volume
            warnings.warn(
                f"Auto-calculated volume is {volume} m^3 for {body_path}. "
                f"Falling back to default volume {fallback} m^3 (buoyancy = rho*g*{fallback}). "
                "Consider setting volume explicitly in config.",
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"Volume not specified and no articulation_prim_path provided. "
                f"Falling back to default volume {fallback} m^3 (buoyancy = rho*g*{fallback}).",
                stacklevel=3,
            )
        return fallback

    def compute_forces(
        self,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute hydrodynamic forces and torques in body frame.

        Args:
            root_lin_vel_w: Root linear velocity in world frame. Shape: (num_envs, 3).
            root_ang_vel_w: Root angular velocity in world frame. Shape: (num_envs, 3).
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).

        Returns:
            Tuple of (forces_b, torques_b) in body frame. Each has shape (num_envs, 3).
        """
        # Transform velocities to body frame
        lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        body_vel = torch.cat([lin_vel_b, ang_vel_b], dim=-1)

        # Transform ocean current to body frame
        current_w = self._current.velocity_w
        current_lin_b = quat_apply_inverse(root_quat_w, current_w[:, :3])
        current_ang_b = quat_apply_inverse(root_quat_w, current_w[:, 3:])
        current_b = torch.cat([current_lin_b, current_ang_b], dim=-1)

        # Relative velocity for hydrodynamic calculations
        relative_vel = body_vel - current_b

        # Compute hydrodynamic components
        damping = self._compute_damping(relative_vel)

        # Coriolis: C_RB uses absolute velocity, C_A uses relative velocity (per Fossen)
        if self._use_full_coriolis:
            coriolis = self._compute_coriolis_full(body_vel, relative_vel)
        else:
            coriolis = self._compute_coriolis(relative_vel)

        # Added mass force (M_A * v_dot)
        # When enabled, uses cached PhysX body acceleration from update_physx_state().
        added_mass_force = torch.zeros(self.num_envs, 6, device=self.device)
        if self._apply_added_mass:
            added_mass_force = self._compute_added_mass(self._physx_acc_b) * self._am_stability_factor

        # Total hydrodynamic wrench: tau = -C(v)*v - D(v)*v - M_A*v_dot + g(eta)
        hydro_wrench = -(coriolis + damping + added_mass_force)
        buoyancy = self._compute_buoyancy_quat(root_quat_w)

        forces_b = hydro_wrench[:, :3] + buoyancy[:, :3]
        torques_b = hydro_wrench[:, 3:] + buoyancy[:, 3:]

        return forces_b, torques_b

    def _compute_damping(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute damping forces (linear + quadratic) with semi-implicit stability clamp.

        When cross-coupling is disabled (default), uses diagonal damping:
            D_l * v + D_q * |v| * v

        When cross-coupling is enabled, velocity from coupled DOFs is added
        to the damping computation. For example, coupling (1, 5) means yaw
        velocity also contributes to sway damping, modeling the sway-yaw
        interaction common in slender underwater vehicles.

        Semi-implicit clamp (when damping_stability_factor is set):
            Since damping is applied as an external wrench (constant over a PhysX step),
            high damping coefficients can cause forward Euler instability where the
            damping force overshoots and reverses velocity. The clamp limits per-axis
            damping to: factor * generalized_mass * |v| / dt, ensuring the velocity
            change from damping never exceeds a fraction of the current velocity.
        """
        vel = body_vel
        if self._damping_cross_coupling is not None:
            vel = body_vel.clone()
            for i, j in self._damping_cross_coupling:
                vel[:, i] = vel[:, i] + body_vel[:, j]

        linear_term = self._linear_damping_diag * vel
        quadratic_term = self._quadratic_damping_diag * torch.abs(vel) * vel
        damping = linear_term + quadratic_term

        # Semi-implicit stability clamp: prevent damping from reversing velocity.
        # Uses the ROOT BODY's rigid_body_inertia (not system inertia) because external
        # damping torque is applied to the root body only, and at high frequencies
        # (100 Hz damping oscillation >> ~5 Hz joint PD bandwidth), articulation joints
        # are effectively free — the root body responds with its own inertia, not the
        # system's. Using system inertia here would allow excessive torque, causing
        # velocity reversal and instability.
        if self._damping_stability_factor is not None and self._body_mass is not None:
            gen_mass = torch.cat(
                [self._body_mass.unsqueeze(-1).expand(-1, 3), self._clamp_inertia],
                dim=-1,
            )
            max_damping = self._damping_stability_factor * gen_mass * torch.abs(vel) / self.dt
            damping = torch.clamp(damping, -max_damping, max_damping)

        return damping

    def _compute_added_mass(self, body_acc: torch.Tensor) -> torch.Tensor:
        """Compute added mass forces."""
        added_mass = torch.bmm(self._added_mass_matrix, body_acc.unsqueeze(-1)).squeeze(-1)
        return added_mass

    def _compute_coriolis_added_mass(self, velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Coriolis force and torque from added mass matrix.

        This is the common computation shared between _compute_coriolis and
        _compute_coriolis_full for the C_A (added mass Coriolis) component.

        Args:
            velocity: Body velocity (linear and angular). Shape: (num_envs, 6).

        Returns:
            Tuple of (force, torque) from added mass Coriolis. Each has shape (num_envs, 3).
        """
        lin_vel = velocity[:, :3]
        ang_vel = velocity[:, 3:]

        # M_A * v
        ma_v = torch.bmm(self._added_mass_matrix, velocity.unsqueeze(-1)).squeeze(-1)
        ma_lin = ma_v[:, :3]
        ma_ang = ma_v[:, 3:]

        # C_A force and torque
        force = -torch.cross(ma_lin, ang_vel, dim=-1)
        torque = -(torch.cross(ma_lin, lin_vel, dim=-1) + torch.cross(ma_ang, ang_vel, dim=-1))

        return force, torque

    def _compute_coriolis(self, body_vel: torch.Tensor) -> torch.Tensor:
        """Compute Coriolis and centripetal forces (C_A only, legacy method)."""
        force, torque = self._compute_coriolis_added_mass(body_vel)
        return torch.cat([force, torque], dim=-1)

    def _compute_coriolis_full(self, body_vel: torch.Tensor, relative_vel: torch.Tensor) -> torch.Tensor:
        """Compute full Coriolis: C(v) = C_RB(v) + C_A(v_r).

        Per Fossen's formulation:
            - C_RB uses absolute velocity (body_vel)
            - C_A uses relative velocity (body_vel - current)
        """
        # C_RB: Rigid body Coriolis (uses absolute velocity)
        ang_vel_abs = body_vel[:, 3:]
        h_rb = self._rigid_body_inertia * ang_vel_abs
        c_rb_force = torch.zeros_like(body_vel[:, :3])
        c_rb_torque = torch.cross(ang_vel_abs, h_rb, dim=-1)

        # C_A: Added mass Coriolis (uses relative velocity)
        c_a_force, c_a_torque = self._compute_coriolis_added_mass(relative_vel)

        total_force = c_rb_force + c_a_force
        total_torque = c_rb_torque + c_a_torque

        return torch.cat([total_force, total_torque], dim=-1)

    def _compute_buoyancy_quat(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Compute buoyancy force and restoring moment.

        Note:
            This method computes ONLY buoyancy force, not weight.
            Weight (gravity) is handled by PhysX with disable_gravity=False.
            This separation allows proper multi-body dynamics where each
            link's mass contributes to the gravitational force naturally.

        The restoring moment arises from the offset between Center of Buoyancy (CoB)
        and Center of Gravity (CoG). When the vehicle tilts, buoyancy acts at CoB
        while gravity acts at CoG, creating a restoring torque.

        Args:
            root_quat_w: Root orientation quaternion (w, x, y, z). Shape: (num_envs, 4).

        Returns:
            Buoyancy wrench in body frame [Fx, Fy, Fz, Mx, My, Mz]. Shape: (num_envs, 6).
        """
        # Transform cached world-up to body frame
        up_dir_b = quat_apply_inverse(root_quat_w, self._up_dir_w)

        # Buoyancy force: F_b = rho * V * g * up_direction (in body frame)
        buoyancy_force_b = self._buoyancy_force_base.unsqueeze(-1) * up_dir_b

        # Restoring moment from buoyancy acting at CoB
        # M = r_cb x F_buoyancy
        buoyancy_moment_b = torch.cross(self._r_cb, buoyancy_force_b, dim=-1)

        # CoG correction torque for domain randomization.
        # PhysX applies gravity at the nominal (URDF) CoG. When CoG is shifted
        # via randomization, apply: M_corr = delta_cg x F_weight_body
        # where F_weight_body = -m*g*up_dir_b (weight points downward in body frame).
        if self._body_mass is not None:
            delta_cg = self._r_cg - self._r_cg_nominal
            weight_force_b = -(self._body_mass.unsqueeze(-1) * self._gravity) * up_dir_b
            buoyancy_moment_b = buoyancy_moment_b + torch.cross(delta_cg, weight_force_b, dim=-1)

        wrench = torch.cat([buoyancy_force_b, buoyancy_moment_b], dim=-1)
        return wrench

    def set_ocean_current(
        self,
        env_ids: torch.Tensor | Sequence[int],
        velocity: torch.Tensor | None = None,
        strength: torch.Tensor | None = None,
    ) -> None:
        """Set ocean current (delegates to the OceanCurrent component)."""
        self._current.set(env_ids, velocity=velocity, strength=strength)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Reset hydrodynamics state for specified environments."""
        if env_ids is None:
            idx = slice(None)
        else:
            idx = self._to_env_ids(env_ids)
        self._current.reset(env_ids)
        self._physx_acc_b[idx] = 0.0
        self._water_density[idx] = self.cfg.water_density

    # --- Properties for parameter access (used by environment-specific randomization) ---

    @property
    def added_mass_matrix(self) -> torch.Tensor:
        """Added mass matrix (num_envs, 6, 6)."""
        return self._added_mass_matrix

    @property
    def linear_damping(self) -> torch.Tensor:
        """Linear damping coefficients (num_envs, 6)."""
        return self._linear_damping_diag

    @property
    def quadratic_damping(self) -> torch.Tensor:
        """Quadratic damping coefficients (num_envs, 6)."""
        return self._quadratic_damping_diag

    @property
    def volume(self) -> torch.Tensor:
        """Vehicle volume (num_envs,)."""
        return self._volume

    @property
    def buoyancy_force(self) -> torch.Tensor:
        """Buoyancy force magnitude (num_envs,)."""
        return self._buoyancy_force_base

    @property
    def center_of_buoyancy(self) -> torch.Tensor:
        """Center of buoyancy in body frame (num_envs, 3)."""
        return self._r_cb

    @property
    def center_of_gravity(self) -> torch.Tensor:
        """Center of gravity in body frame (num_envs, 3)."""
        return self._r_cg

    @property
    def water_density(self) -> torch.Tensor:
        """Water density per environment (num_envs,)."""
        return self._water_density

    @property
    def body_mass(self) -> torch.Tensor | None:
        """Rigid body mass per environment (num_envs,), or None if not configured."""
        return self._body_mass

    @property
    def rigid_body_inertia(self) -> torch.Tensor:
        """Rigid body inertia diagonal (num_envs, 3)."""
        return self._rigid_body_inertia

    @property
    def apply_added_mass(self) -> bool:
        """Whether added mass force (M_A * v_dot) is applied."""
        return self._apply_added_mass

    @property
    def current(self) -> OceanCurrent:
        """The ocean current component (shareable across models)."""
        return self._current

    @property
    def base_parameters(self) -> HydroParams:
        """Immutable snapshot of nominal parameters from cfg (computed once)."""
        return self._base_parameters

    def _snapshot_parameters(self) -> HydroParams:
        """Capture nominal parameters as an immutable (cloned) snapshot."""
        am_diag = torch.diagonal(self._added_mass_matrix, dim1=-2, dim2=-1)
        return HydroParams(
            added_mass=am_diag.clone(),
            linear_damping=self._linear_damping_diag.clone(),
            quadratic_damping=self._quadratic_damping_diag.clone(),
            volume=self._volume.clone(),
            water_density=self._water_density.clone(),
            center_of_buoyancy=self._r_cb.clone(),
            center_of_gravity=self._r_cg.clone(),
            rigid_body_inertia=self._rigid_body_inertia.clone(),
            body_mass=None if self._body_mass is None else self._body_mass.clone(),
        )

    def get_parameters(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> HydroParams:
        """Read current per-env hydrodynamic parameters.

        Args:
            env_ids: Indices to read (all if None).

        Returns:
            HydroParams with per-env tensors for the requested environments.
        """
        idx = slice(None) if env_ids is None else self._to_env_ids(env_ids)
        am_diag = torch.diagonal(self._added_mass_matrix[idx], dim1=-2, dim2=-1)
        return HydroParams(
            added_mass=am_diag,
            linear_damping=self._linear_damping_diag[idx],
            quadratic_damping=self._quadratic_damping_diag[idx],
            volume=self._volume[idx],
            water_density=self._water_density[idx],
            center_of_buoyancy=self._r_cb[idx],
            center_of_gravity=self._r_cg[idx],
            rigid_body_inertia=self._rigid_body_inertia[idx],
            body_mass=None if self._body_mass is None else self._body_mass[idx],
        )

    def set_parameters(self, env_ids: torch.Tensor | Sequence[int], **fields) -> None:
        """Write absolute per-env parameter values. None/absent fields untouched.

        Accepted keyword fields match HydroParams: added_mass, linear_damping,
        quadratic_damping, volume, water_density, center_of_buoyancy,
        center_of_gravity, rigid_body_inertia, body_mass. Automatically refreshes
        buoyancy force when volume or water_density changes.

        Args:
            env_ids: Environment indices to write.
            **fields: Per-env tensors keyed by HydroParams field name.
        """
        idx = self._to_env_ids(env_ids)
        if (am := fields.get("added_mass")) is not None:
            self._added_mass_matrix[idx] = torch.diag_embed(am)
        if (ld := fields.get("linear_damping")) is not None:
            self._linear_damping_diag[idx] = ld
        if (qd := fields.get("quadratic_damping")) is not None:
            self._quadratic_damping_diag[idx] = qd
        if (cb := fields.get("center_of_buoyancy")) is not None:
            self._r_cb[idx] = cb
        if (cg := fields.get("center_of_gravity")) is not None:
            self._r_cg[idx] = cg
        if (rbi := fields.get("rigid_body_inertia")) is not None:
            self._rigid_body_inertia[idx] = rbi
        if (bm := fields.get("body_mass")) is not None and self._body_mass is not None:
            self._body_mass[idx] = bm
        vol_changed = (vol := fields.get("volume")) is not None
        den_changed = (den := fields.get("water_density")) is not None
        if vol_changed:
            self._volume[idx] = vol
        if den_changed:
            self._water_density[idx] = den
        if vol_changed or den_changed:
            self.update_buoyancy_force(env_ids)

    def scale_parameters(self, env_ids: torch.Tensor | Sequence[int], **ranges) -> None:
        """Convenience DR: set each named field to base * uniform(lo, hi) per env.

        Args:
            env_ids: Environment indices to randomize.
            **ranges: field_name=(lo, hi) tuple (sample base * uniform(lo,hi)) or a per-env scale tensor of shape [n] (use base * scale directly).
        """
        idx = self._to_env_ids(env_ids)
        n = len(idx)
        base = self._base_parameters
        out: dict[str, torch.Tensor] = {}
        for name, value in ranges.items():
            base_val = getattr(base, name)
            if isinstance(value, torch.Tensor):
                # Per-env scalar scale [n]: broadcast over the field's trailing dims.
                scale = value.to(device=self.device, dtype=base_val.dtype)
                scale = scale.reshape((n,) + (1,) * (base_val.dim() - 1))
            else:
                lo, hi = value
                scale = torch.rand((n,) + base_val.shape[1:], device=self.device) * (hi - lo) + lo
            out[name] = base_val[idx] * scale
        self.set_parameters(env_ids, **out)

    def update_buoyancy_force(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Update buoyancy force after volume change.

        Call this after modifying volume to ensure buoyancy_force_base is consistent.

        Args:
            env_ids: Environment indices to update. If None, updates all.
        """
        if env_ids is None:
            self._buoyancy_force_base = self._water_density * self._gravity * self._volume
        else:
            env_ids = self._to_env_ids(env_ids)
            self._buoyancy_force_base[env_ids] = self._water_density[env_ids] * self._gravity * self._volume[env_ids]
