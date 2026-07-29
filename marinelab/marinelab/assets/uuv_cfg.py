# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration classes for underwater vehicles.

This module defines the configuration dataclasses for UUV hydrodynamics,
ocean currents, and thruster systems. These are used as base classes
for robot-specific configurations.

Reference:
    Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control. Wiley.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class HydrodynamicsCfg:
    """Configuration for Fossen model hydrodynamics.

    All coefficients are for a 6-DOF system: [surge, sway, heave, roll, pitch, yaw].
    Diagonal matrices are assumed for simplicity (off-diagonal terms can be added later).

    The model implements the complete Fossen formulation including:
        - Buoyancy force applied as external force (weight handled by PhysX gravity)
        - Full Coriolis matrix C(v) = C_RB(v) + C_A(v)
        - Quaternion-based buoyancy calculation (no gimbal lock)

    Physics Model:
        - PhysX handles gravity (weight) naturally for all bodies
        - HydrodynamicsModel applies buoyancy as external upward force
        - This ensures proper multi-body dynamics where each link's mass is considered
    """

    # Added mass coefficients (kg for linear, kg*m^2 for angular)
    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)

    # Linear damping coefficients (Ns/m for linear, Nms/rad for angular)
    linear_damping: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)

    # Quadratic damping coefficients (Ns^2/m^2 for linear, Nms^2/rad^2 for angular)
    quadratic_damping: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    # Vehicle volume for buoyancy calculation (m^3)
    # If None, will be auto-calculated from collision geometry when articulation_prim_path is provided
    volume: float | None = None

    # Body name for volume auto-calculation (e.g., "base", "base_link")
    # Only used when volume is None and articulation_prim_path is provided
    body_name: str = "base"

    # Center of buoyancy position in body frame (m, [x, y, z])
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.01)

    # Center of gravity position in body frame (m, [x, y, z])
    # Note: This is for restoring moment calculation only. Actual CoG is from PhysX.
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Water density (kg/m^3, default: freshwater, seawater: 1025.0)
    water_density: float = 997.0

    # Use full Coriolis matrix C(v) = C_RB(v) + C_A(v) per Fossen model.
    # When PhysX enable_gyroscopic_forces=True, set False to use C_A only and avoid
    # double-counting the rigid body Coriolis/gyroscopic terms already handled by PhysX.
    use_full_coriolis: bool = True

    # Rigid body inertia for full Coriolis (kg*m^2, [I_xx, I_yy, I_zz])
    # If None, estimates from added mass rotational terms
    rigid_body_inertia: tuple[float, float, float] | None = None

    # Body mass for CoG correction torque (kg)
    # When set, enables gravity restoring moment correction during CoG randomization.
    # Should match the URDF/USD rigid body mass so that the correction is zero at nominal.
    body_mass: float | None = None

    # Added mass force (M_A * v_dot) settings
    # When True, applies M_A * v_dot as external force for more accurate dynamics
    # Uses PhysX-computed acceleration from previous step
    # Default False for backward compatibility (only C_A velocity coupling is applied)
    apply_added_mass_force: bool = False

    # Stability factor for added mass force (0 < factor <= 1.0)
    # Lower values increase stability at the cost of physical accuracy
    added_mass_stability_factor: float = 0.8

    # Off-diagonal damping cross-coupling pairs.
    # Each tuple (i, j) means DOF j velocity contributes to DOF i damping.
    # Example: ((1, 5), (5, 1), (2, 4), (4, 2)) couples sway-yaw and heave-pitch.
    # None disables cross-coupling (standard diagonal damping only).
    damping_cross_coupling: tuple[tuple[int, int], ...] | None = None

    # Semi-implicit stability clamp for damping force.
    # Limits per-axis damping so that dt * D_eff / mass < factor, preventing
    # forward Euler instability where high damping reverses velocity.
    # Requires body_mass and rigid_body_inertia to be set.
    # None disables clamping (backward compatible default).
    damping_stability_factor: float | None = None


@configclass
class OceanCurrentCfg:
    """Configuration for ocean current disturbances."""

    # Maximum current velocity [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
    # All-zero values effectively disable ocean current disturbances.
    max_velocity: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Gaussian noise scale for current velocity
    noise_scale: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@configclass
class ThrusterCfg:
    """Configuration for thruster system.

    Defines the thruster allocation matrix and dynamics for converting
    normalized commands to body-frame forces and torques.
    """

    # Number of thrusters
    num_thrusters: int = 6

    # Maximum thrust per thruster (N)
    max_thrust: float = 50.0

    # Thrust coefficient: thrust = coefficient * command
    thrust_coefficient: float = 40.0

    # First-order dynamics time constants (seconds)
    time_constant_up: float = 0.1
    time_constant_down: float = 0.05

    # Nonlinear thrust-curve model (BlueROV T200 sim-to-real fidelity).
    # OFF by default: when False, compute_wrench keeps the linear map
    # (thrust = state * thrust_coefficient), byte-identical to prior runs.
    # When True, applies a normalized-command deadband + signed-square curve
    # so the sim thruster matches the real T200's zero-thrust deadzone and
    # quadratic (T ~ command^2) propeller law instead of a linear map.
    enable_thrust_curve: bool = False
    # Deadband half-width in normalized command space [-1, 1]: |state| below
    # this produces zero thrust. 0.075 = MarineGym T200 fit (also ~0.0625 from
    # the +/-25 us PWM deadband over the +/-400 us half-range). Only used when
    # enable_thrust_curve is True.
    thrust_deadband: float = 0.075

    # Per-step multiplicative actuation noise std (white Gaussian). Only armed when
    # ThrusterModel receives enable_actuation_noise=True; 0.0 -> disabled. This is the
    # THIRD actuation channel: distinct from DR (thrust_coefficient spread, reset-time)
    # and fault (thruster health, reset-time). It perturbs the COMMAND every step:
    # applied = commanded * (1 + eps), eps ~ N(0, actuation_noise_std). Initial value
    # arbitrary (thrust unmeasured) -- a sweep target.
    actuation_noise_std: float = 0.0

    # Thruster allocation matrix: wrench = allocation_matrix @ thrusts
    # Shape: (6, num_thrusters) where 6 = [Fx, Fy, Fz, Mx, My, Mz]
    # Default derived from BlueROV2 Heavy 6-thruster layout
    allocation_matrix: tuple[tuple[float, ...], ...] = (
        # Fx: forward surge force
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        # Fy: lateral sway force
        (-0.707, 0.707, 0.707, -0.707, 0.0, 0.0),
        # Fz: vertical heave force
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        # Mx: roll torque
        (0.0, 0.0, 0.0, 0.0, 0.1, -0.1),
        # My: pitch torque
        (0.0, 0.0, 0.0, 0.0, 0.12, 0.12),
        # Mz: yaw torque
        (0.19, -0.19, 0.19, -0.19, 0.0, 0.0),
    )
