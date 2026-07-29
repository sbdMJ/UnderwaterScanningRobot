# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV hover environment configurations.

This module provides environment configurations for BlueROV hover tasks
where the vehicle must maintain both position and attitude at a target.
"""

from __future__ import annotations

from isaaclab.utils import configclass

# Import robot configurations from isaaclab_assets
from marinelab.assets import (
    BLUEROV_CFG,
    BlueROVHydrodynamicsCfg,
    BlueROVThrusterCfg,
    OceanCurrentCfg,
)

from .bluerov_env_cfg import BlueROVEnvCfg, DomainRandomizationCfg
from .tasks import HoverTaskCfg


@configclass
class BlueROVHoverEnvCfg(BlueROVEnvCfg):
    """BlueROV hover environment configuration.

    The vehicle must hover at a target position while maintaining upright attitude.
    Uses HoverTask which provides both position and attitude control objectives.
    """

    # Robot configuration from isaaclab_assets
    robot = BLUEROV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Hydrodynamics (BlueROV-specific parameters from isaaclab_assets)
    hydrodynamics = BlueROVHydrodynamicsCfg()

    # Thrusters (BlueROV has 6 thrusters)
    thrusters = BlueROVThrusterCfg()

    # Action space matches number of thrusters
    action_space: int = 6

    # Observation space: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + goal_pos_b(3) + up(2) = 18
    observation_space: int = 18

    # Task configuration
    task: HoverTaskCfg = HoverTaskCfg(
        goal_pos_range=(2.0, 2.0, 1.0),
        initial_height=2.0,
    )

    # Ocean current (disabled by default)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    # Reward scales for hover task
    position_reward_scale: float = 15.0
    position_reward_exp_scale: float = 2.0
    orientation_reward_scale: float = 5.0
    orientation_exp_scale: float = 0.5
    linear_velocity_penalty_scale: float = -0.01
    angular_velocity_penalty_scale: float = -0.005
    action_rate_penalty_scale: float = -0.005
    action_magnitude_penalty_scale: float = -0.0005
    alive_reward_scale: float = 0.1


@configclass
class BlueROVHoverTrainEnvCfg(BlueROVHoverEnvCfg):
    """BlueROV hover training environment with domain randomization.

    Enables domain randomization for robust policy learning that can
    transfer to real-world conditions.
    """

    # Enable domain randomization
    # Note: mass_scale removed - weight is now handled by PhysX (disable_gravity=False)
    randomization = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization
        position_x_range=(-2.5, 2.5),
        position_y_range=(-2.5, 2.5),
        position_z_range=(1.5, 2.5),
        roll_range=(-0.628, 0.628),
        pitch_range=(-0.628, 0.628),
        yaw_range=(0.0, 6.283),
        # Hydrodynamic parameter randomization
        added_mass_scale=(0.8, 1.2),
        linear_damping_scale=(0.8, 1.2),
        quadratic_damping_scale=(0.8, 1.2),
        volume_scale=(0.95, 1.05),
        # Thruster randomization
        thrust_coefficient_scale=(0.9, 1.1),
        time_constant_scale=(0.9, 1.1),
        # Center of Buoyancy offset randomization (meters)
        cob_offset_x=(-0.02, 0.02),
        cob_offset_y=(-0.02, 0.02),
        cob_offset_z=(-0.03, 0.03),
        # Inertia randomization
        inertia_scale=(0.9, 1.1),
    )

    # Ocean currents for disturbance rejection training
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.3, 0.3, 0.1, 0.0, 0.0, 0.0),
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )


@configclass
class BlueROVHoverEvalEnvCfg(BlueROVHoverEnvCfg):
    """BlueROV hover evaluation environment with aggressive randomization.

    Uses wider randomization ranges than training to stress-test the
    learned policy's robustness and generalization.
    """

    # Aggressive domain randomization for stress testing
    # Note: mass_scale removed - weight is now handled by PhysX (disable_gravity=False)
    randomization = DomainRandomizationCfg(
        enable=True,
        # Wider initial pose randomization
        position_x_range=(-3.0, 3.0),
        position_y_range=(-3.0, 3.0),
        position_z_range=(1.0, 3.0),
        roll_range=(-0.785, 0.785),
        pitch_range=(-0.785, 0.785),
        yaw_range=(0.0, 6.283),
        # Wider hydrodynamic parameter randomization
        added_mass_scale=(0.5, 1.5),
        linear_damping_scale=(0.5, 1.5),
        quadratic_damping_scale=(0.5, 1.5),
        volume_scale=(0.85, 1.15),
        # Wider thruster randomization
        thrust_coefficient_scale=(0.7, 1.3),
        time_constant_scale=(0.7, 1.3),
        # Aggressive CoB offset randomization (meters)
        cob_offset_x=(-0.04, 0.04),
        cob_offset_y=(-0.04, 0.04),
        cob_offset_z=(-0.05, 0.05),
        # Inertia randomization
        inertia_scale=(0.8, 1.2),
    )

    # Stronger ocean currents for evaluation
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.5, 0.5, 0.2, 0.0, 0.0, 0.0),
        noise_scale=(0.15, 0.15, 0.1, 0.0, 0.0, 0.0),
    )

    # Harder reward shaping
    position_reward_scale: float = 20.0
    position_reward_exp_scale: float = 2.5
    orientation_reward_scale: float = 8.0
    orientation_exp_scale: float = 0.4
