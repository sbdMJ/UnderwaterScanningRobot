# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for BlueROV environments.

Hyperparameters are tuned for underwater vehicle control tasks with:
- Hover: 18-dimensional observation space (pose, velocity, goal)
- Attitude: 12-dimensional observation space (orientation, angular velocity, goal)
- 6-dimensional action space (thruster commands)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

##
# Hover Task Configurations
##


@configclass
class BlueROVHoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for BlueROV hover task.

    This configuration uses larger networks and longer horizons compared to
    aerial vehicles due to the more complex dynamics of underwater vehicles.
    """

    num_steps_per_env = 48
    max_iterations = 500
    save_interval = 50
    experiment_name = "bluerov_hover"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BlueROVHoverTrainPPORunnerCfg(BlueROVHoverPPORunnerCfg):
    """PPO configuration for BlueROV hover training with domain randomization."""

    max_iterations = 600
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BlueROVHoverEvalPPORunnerCfg(BlueROVHoverPPORunnerCfg):
    """PPO configuration for BlueROV hover evaluation with aggressive randomization."""

    max_iterations = 800
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


##
# Attitude Task Configurations
##


@configclass
class BlueROVAttitudePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for BlueROV attitude-only task.

    Uses a smaller network since attitude control is simpler than
    full position + attitude control.
    """

    num_steps_per_env = 32  # Shorter horizon for simpler task
    max_iterations = 300
    save_interval = 50
    experiment_name = "bluerov_attitude"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 64],  # Smaller network for simpler task
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BlueROVAttitudeTrainPPORunnerCfg(BlueROVAttitudePPORunnerCfg):
    """PPO configuration for BlueROV attitude training with domain randomization."""

    max_iterations = 400
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
