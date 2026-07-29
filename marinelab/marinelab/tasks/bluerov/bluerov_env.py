# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV Environment with modular task and reward systems.

This environment provides:
- Pluggable task system (HoverTask, AttitudeTask)
- Composition-based reward terms via RewardManager
- Separated ThrusterModel for cleaner abstraction
- Full 6-DOF Fossen hydrodynamics model
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

# Import shared marine physics models from marinelab.core
from marinelab.core import HydrodynamicsModel, ThrusterModel

from .bluerov_env_cfg import BlueROVEnvCfg
from .rewards import (
    RewardManager,
    RewardTermCfg,
    action_magnitude_penalty,
    action_rate_penalty,
    alive_bonus,
    angular_velocity_penalty,
    linear_velocity_penalty,
    orientation_upright_exp,
    position_tracking_exp,
)
from .tasks import AttitudeTask, AttitudeTaskCfg, HoverTask, TaskBase

if TYPE_CHECKING:
    pass


class BlueROVEnvWindow(BaseEnvWindow):
    """Window manager for the BlueROV environment."""

    def __init__(self, env: BlueROVEnv, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


class BlueROVEnv(DirectRLEnv):
    """BlueROV environment with modular task and reward systems.

    This environment provides:
    - Task abstraction for easy addition of new tasks (hover, attitude, track)
    - Composition-based reward terms via RewardManager
    - Full 6-DOF Fossen hydrodynamics model
    - Separated ThrusterModel for cleaner code structure

    Observation Space (configurable via task):
        Base observations:
        - Root position in world frame (3)
        - Root orientation quaternion (4)
        - Root linear velocity in body frame (3)
        - Root angular velocity in body frame (3)
        - Up vector projection (2)
        Task observations:
        - Goal position in body frame (3) [for HoverTask]
        Total: 18 dimensions (default)

    Action Space:
        - Thruster commands normalized to [-1, 1] (num_thrusters)
        Default: 6 thrusters for BlueROV
    """

    cfg: BlueROVEnvCfg

    def __init__(self, cfg: BlueROVEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the BlueROV environment.

        Args:
            cfg: Environment configuration.
            render_mode: Render mode for visualization.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs)

        # Get body ID for force application
        self._body_id = self._robot.find_bodies(self.cfg.body_link_name)[0]

        # Get gravity magnitude (for reference only - PhysX handles gravity)
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # Initialize hydrodynamics model
        # Note: robot_mass parameter removed - PhysX handles gravity naturally
        self._hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.hydrodynamics,
            current_cfg=self.cfg.ocean_current,
            dt=self.physics_dt,
            articulation_prim_path=self.cfg.robot.prim_path.replace("env_.*", "env_0"),
        )

        # Initialize thruster model
        self._thruster = ThrusterModel(
            cfg=self.cfg.thrusters,
            num_envs=self.num_envs,
            device=self.device,
            enable_randomization=self.cfg.randomization.enable,
        )

        # Initialize task from config
        self._task = self._create_task_from_config()

        # Initialize reward manager with configurable terms
        self._reward_manager = RewardManager(
            cfg=self._build_reward_cfg(),
            num_envs=self.num_envs,
            device=self.device,
        )

        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Force/torque buffers for visualization
        self._thrust_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

        # Cached world up vector for the body-frame up observation (hot path)
        self._up_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)

        # Setup debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _create_task_from_config(self) -> TaskBase:
        """Create task instance from environment configuration.

        The task type is determined by the task configuration class:
        - AttitudeTaskCfg -> AttitudeTask
        - HoverTaskCfg (or any other TaskBaseCfg) -> HoverTask

        Returns:
            Task instance matching the configuration.
        """
        task_cfg = self.cfg.task
        if isinstance(task_cfg, AttitudeTaskCfg):
            return AttitudeTask(cfg=task_cfg, num_envs=self.num_envs, device=self.device)
        return HoverTask(cfg=task_cfg, num_envs=self.num_envs, device=self.device)

    def _build_reward_cfg(self) -> dict[str, RewardTermCfg]:
        """Build reward configuration from environment config.

        Returns:
            Dictionary mapping term names to RewardTermCfg.
        """
        return {
            "position_tracking": RewardTermCfg(
                func=position_tracking_exp,
                weight=self.cfg.position_reward_scale,
                params={"exp_scale": self.cfg.position_reward_exp_scale},
            ),
            "orientation_upright": RewardTermCfg(
                func=orientation_upright_exp,
                weight=self.cfg.orientation_reward_scale,
                params={"exp_scale": getattr(self.cfg, "orientation_exp_scale", 0.5)},
            ),
            "linear_velocity": RewardTermCfg(
                func=linear_velocity_penalty,
                weight=self.cfg.linear_velocity_penalty_scale,
            ),
            "angular_velocity": RewardTermCfg(
                func=angular_velocity_penalty,
                weight=self.cfg.angular_velocity_penalty_scale,
            ),
            "action_rate": RewardTermCfg(
                func=action_rate_penalty,
                weight=self.cfg.action_rate_penalty_scale,
            ),
            "action_magnitude": RewardTermCfg(
                func=action_magnitude_penalty,
                weight=self.cfg.action_magnitude_penalty_scale,
            ),
            "alive_bonus": RewardTermCfg(
                func=alive_bonus,
                weight=self.cfg.alive_reward_scale,
            ),
        }

    def _setup_scene(self):
        """Setup the simulation scene with robot and terrain."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.4, 0.6, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Apply thruster dynamics. _pre_physics_step runs once per env step ->
        # elapsed time between calls is step_dt (= physics_dt * decimation).
        self._thruster.apply_dynamics(self._actions, self.step_dt)

    def _apply_action(self):
        """Apply thruster forces and hydrodynamic forces to the robot."""
        # Compute thruster wrench
        thrust_forces, thrust_torques = self._thruster.compute_wrench()
        self._thrust_forces[:, 0, :] = thrust_forces
        self._thrust_torques[:, 0, :] = thrust_torques

        # Compute hydrodynamic forces
        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )

        # Combine forces in body frame
        total_forces = self._thrust_forces.clone()
        total_forces[:, 0, :] += self._hydro_forces

        total_torques = self._thrust_torques.clone()
        total_torques[:, 0, :] += self._hydro_torques

        # Apply to robot via wrench composer
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces,
            torques=total_torques,
        )

    def _get_observations(self) -> dict:
        """Compute observations."""
        # Update hydrodynamics PhysX state for added mass force calculation
        # This must be called after robot.update() which happens at the start of each step
        if self._hydro.apply_added_mass:
            self._hydro.update_physx_state(
                body_com_acc_w=self._robot.data.body_com_acc_w[:, self._body_id[0], :],
                root_quat_w=self._robot.data.root_quat_w,
            )

        # Base state observations
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        # Up vector in body frame for orientation
        up_b = quat_apply_inverse(root_quat_w, self._up_w)

        # Task-specific goal observations
        goal_obs = self._task.get_goal_observations(self._robot)

        # Concatenate observations
        obs = torch.cat(
            [
                root_pos_w,  # 3
                root_quat_w,  # 4
                root_lin_vel_b,  # 3
                root_ang_vel_b,  # 3
                goal_obs,  # 3 (task-dependent)
                up_b[:, :2],  # 2
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using the reward manager."""
        goal_state = self._task.get_goal_state()

        return self._reward_manager.compute(
            robot=self._robot,
            dt=self.step_dt,
            goal_pos_w=goal_state.get("goal_pos_w", torch.zeros(self.num_envs, 3, device=self.device)),
            actions=self._actions,
            prev_actions=self._prev_actions,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        height = self._robot.data.root_pos_w[:, 2]
        too_low = height < self.cfg.min_height
        too_high = height > self.cfg.max_height

        distance_from_origin = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        too_far = distance_from_origin > self.cfg.max_distance_from_origin

        # Task-specific terminations
        task_terminated = self._task.get_terminations(self._robot)

        terminated = too_low | too_high | too_far | task_terminated

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Log episode statistics
        reward_sums = self._reward_manager.reset(env_ids)

        self.extras["log"] = {}
        for name, value in reward_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = value

        # Log position tracking metric only for tasks with goal_pos_w
        if hasattr(self._task, "goal_pos_w"):
            final_distance = torch.linalg.norm(
                self._task.goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
            ).mean()
            self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance.item()
        self.extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        # Reset components
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # Reset thruster and hydrodynamics
        self._thruster.reset(env_ids)
        self._hydro.reset(env_ids)

        # Apply domain randomization
        # Note: mass_scale removed - weight is now handled by PhysX (disable_gravity=False)
        if self.cfg.randomization.enable:
            from .mdp.events import randomize_hydrodynamics

            r = self.cfg.randomization
            randomize_hydrodynamics(
                self,
                env_ids,
                added_mass_scale=r.added_mass_scale,
                linear_damping_scale=r.linear_damping_scale,
                quadratic_damping_scale=r.quadratic_damping_scale,
                volume_scale=r.volume_scale,
                cob_offset_x=r.cob_offset_x,
                cob_offset_y=r.cob_offset_y,
                cob_offset_z=r.cob_offset_z,
                cog_offset_x=r.cog_offset_x,
                cog_offset_y=r.cog_offset_y,
                cog_offset_z=r.cog_offset_z,
                inertia_scale=r.inertia_scale,
            )
            self._thruster.randomize_parameters(
                env_ids=env_ids,
                thrust_coeff_scale=r.thrust_coefficient_scale,
                time_constant_scale=r.time_constant_scale,
            )

        # Randomize ocean current
        if any(v > 0 for v in self.cfg.ocean_current.max_velocity):
            self._hydro.set_ocean_current(env_ids)

        # Reset task (samples new goals)
        self._task.reset(env_ids, self._terrain.env_origins)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Apply initial pose randomization
        if self.cfg.randomization.enable:
            num_reset = len(env_ids)
            r = self.cfg.randomization

            pos_lo = torch.tensor(
                [r.position_x_range[0], r.position_y_range[0], r.position_z_range[0]], device=self.device
            )
            pos_hi = torch.tensor(
                [r.position_x_range[1], r.position_y_range[1], r.position_z_range[1]], device=self.device
            )
            pos = torch.rand(num_reset, 3, device=self.device) * (pos_hi - pos_lo) + pos_lo
            default_root_state[:, 0] += pos[:, 0]
            default_root_state[:, 1] += pos[:, 1]
            default_root_state[:, 2] = pos[:, 2]

            rot_lo = torch.tensor([r.roll_range[0], r.pitch_range[0], r.yaw_range[0]], device=self.device)
            rot_hi = torch.tensor([r.roll_range[1], r.pitch_range[1], r.yaw_range[1]], device=self.device)
            rpy = torch.rand(num_reset, 3, device=self.device) * (rot_hi - rot_lo) + rot_lo
            random_quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            default_root_state[:, 3:7] = quat_mul(default_root_state[:, 3:7], random_quat)
        else:
            default_root_state[:, 2] = self.cfg.task.initial_height

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

            if not hasattr(self, "current_visualizer"):
                arrow_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                arrow_cfg.prim_path = "/Visuals/Command/ocean_current"
                arrow_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.current_visualizer = VisualizationMarkers(arrow_cfg)
            self.current_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "current_visualizer"):
                self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if not hasattr(self, "scene"):  # 종료 중 씬이 이미 파괴됨 → 콜백 스킵 (teardown race guard)
            return
        goal_state = self._task.get_goal_state()
        goal_pos = goal_state.get("goal_pos_w", torch.zeros(self.num_envs, 3, device=self.device))
        self.goal_pos_visualizer.visualize(goal_pos)

        # Visualize ocean current
        current_vel = self._hydro.current.velocity_w[:, :3]
        current_magnitude = torch.linalg.norm(current_vel, dim=1, keepdim=True)
        current_scale = current_magnitude.clamp(min=0.01) * 2.0

        robot_pos = self._robot.data.root_pos_w
        arrow_pos = robot_pos + torch.tensor([[0.0, 0.0, 0.5]], device=self.device)

        from isaaclab.utils.math import quat_from_angle_axis

        default_dir = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        current_dir = current_vel / current_magnitude.clamp(min=1e-6)

        cross = torch.cross(default_dir, current_dir, dim=1)
        dot = torch.sum(default_dir * current_dir, dim=1)
        angle = torch.atan2(torch.linalg.norm(cross, dim=1), dot)
        axis = cross / torch.linalg.norm(cross, dim=1, keepdim=True).clamp(min=1e-6)

        arrow_quat = quat_from_angle_axis(angle, axis)

        self.current_visualizer.visualize(
            translations=arrow_pos,
            orientations=arrow_quat,
            scales=current_scale.expand(-1, 3),
        )
