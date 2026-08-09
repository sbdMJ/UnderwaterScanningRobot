#!/usr/bin/env python3
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab play entry for the public marine-vehicle environments (BlueROV, ...).

Mirror of the marinelab train entry: isaaclab stays pristine, so this overlay entry
adds the single overlay concern — ``import marinelab`` (after AppLauncher boots, so the
USD ``pxr`` runtime exists) to trigger marinelab's gym.register() calls. marinelab tasks
use the stock rsl-rl ``OnPolicyRunner``, so no custom runner dispatch is needed.

The main() body is replicated from upstream/main
``scripts/reinforcement_learning/rsl_rl/play.py`` so this entry tracks upstream behavior.

Usage (run via isaaclab's runtime):
    cd /workspace/isaaclab && ./isaaclab.sh -p \
        /workspace/marinelab/scripts/play.py \
        --task Isaac-BlueROV-Hover-Direct-v0 --num_envs 8 --checkpoint <path/model_N.pt>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Make upstream cli_args importable (it lives next to upstream play.py and uses
# `import cli_args  # isort: skip`, relying on sys.path).
ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab")
UPSTREAM_RL_DIR = os.path.join(ISAACLAB_PATH, "scripts", "reinforcement_learning", "rsl_rl")
if UPSTREAM_RL_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_RL_DIR)

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# --- trajectory logging / scan metrics (wallscan tasks) ------------------------
# Off by default so the plain `play.py` behaviour (run forever, watch it) is unchanged.
parser.add_argument(
    "--log_traj",
    action="store_true",
    default=False,
    help="Log the scan trajectory, compute scan-quality metrics, and exit (PKRC-WallScan tasks).",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=None,
    help="Control steps to run with --log_traj. Default: one full episode (episode_length_s / step_dt).",
)
parser.add_argument(
    "--out_dir", type=str, default=None, help="Where --log_traj writes npz/json/png. Default: <repo>/results."
)
parser.add_argument("--tag", type=str, default="eval", help="Filename suffix for --log_traj outputs.")
parser.add_argument(
    "--score_episode",
    type=int,
    default=0,
    help="Which per-env episode index --log_traj scores (0 = the first). -1 pools every logged step.",
)
parser.add_argument("--no_plot", action="store_true", default=False, help="Skip the PNG with --log_traj.")
# --- plant corrections, so an RL checkpoint can be scored on the SAME vehicle as the NMPC -------
# Both default to the legacy classes, i.e. omitting them reproduces every published RL number.
# The NMPC results are all measured under `fixed` + `z_slender`; comparing across a plant
# difference would be comparing two different robots, so these exist to close that gap.
parser.add_argument("--tam", choices=["shipped", "fixed"], default="shipped",
                    help="Thruster allocation matrix. 'fixed' = PKRCThrusterCfgFixedTAM, which puts "
                         "the sway moment arm on roll where a heave differential can cancel it. "
                         "NOTE the policy was TRAINED on 'shipped', so 'fixed' is off-distribution "
                         "for it -- that is the deployment question, not a like-for-like one.")
parser.add_argument("--hydro", choices=["shipped", "z_slender"], default="shipped",
                    help="Hydrodynamic coefficients. 'z_slender' matches the USD mesh's long axis "
                         "and roughly halves heave drag; also off-distribution for the checkpoint.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401

import marinelab  # noqa: F401  # triggers marinelab gym.register() (overlay concern, after pxr exists)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _run_with_trajectory_logging(env, policy, policy_nn, simulation_app) -> None:
    """Run a BOUNDED rollout, logging the scan trajectory, then write metrics/npz/png.

    Everything logged here is read straight off state the env already computed for
    this step, or derived from ``root_pos_w``/``root_quat_w`` with the pure helpers in
    ``geometry``/``sensors``. The env's ``_read_state()`` is deliberately NOT called:
    ``apply_sensors`` draws from the global torch RNG, so an extra call would shift
    the sensor-noise stream and make the logged run differ from an unlogged one.
    """
    import json

    import numpy as np

    from isaaclab.utils.math import euler_xyz_from_quat

    from marinelab.tasks.pkrc_wallscan import eval_metrics as em
    from marinelab.tasks.pkrc_wallscan import geometry
    from marinelab.tasks.pkrc_wallscan.sensors import _body_up

    u = env.unwrapped
    if not hasattr(u, "_scan_state"):
        raise SystemExit(
            f"--log_traj only supports the PKRC-WallScan tasks (got --task {args_cli.task!r}, "
            "whose env has no scan state machine)."
        )

    cfg = u.cfg
    dt = u.step_dt
    steps = args_cli.eval_steps if args_cli.eval_steps is not None else int(round(cfg.episode_length_s / dt))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.abspath(args_cli.out_dir or os.path.join(repo_root, "results"))
    os.makedirs(out_dir, exist_ok=True)
    score_episode = None if args_cli.score_episode < 0 else args_cli.score_episode

    log = em.TrajectoryLog()
    obs = env.get_observations()
    print(f"[INFO] logging {steps} control steps x {u.num_envs} envs ({steps * dt:.0f} s of sim) -> {out_dir}")

    for i in range(steps):
        if not simulation_app.is_running():
            print(f"[WARN] simulation app closed early at step {i}/{steps}")
            break
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            pos = u._robot.data.root_pos_w - u.scene.env_origins
            quat = u._robot.data.root_quat_w
            _, _, yaw = euler_xyz_from_quat(quat)
            up_z = _body_up(quat)[:, 2].clamp(-1.0, 1.0)
            # Nominal mount => the clean GT sonar reading, matching the reward's wall_dist.
            wall_dist = geometry.sonar_wall_distance(
                pos[:, :2], yaw, u._sonar_mount_nom, u._sonar_yaw_nom, cfg.tank_radius
            )
            log.record(
                phase=u._scan_state.phase,
                cycles=u._cycles,
                searching=u._search_active,
                done=dones,
                # The wrapper collapses dones to terminated|truncated, so keep the two
                # apart: these distinguish a success/failure termination from a time-out.
                terminated=u.reset_terminated,
                time_out=u.reset_time_outs,
                term_collided=u._term_collided,
                term_oob=u._term_oob,
                term_tilted=u._term_tilted,
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                yaw=yaw,
                theta=torch.atan2(pos[:, 1], pos[:, 0]),
                tilt_deg=torch.rad2deg(torch.arccos(up_z)),
                s=u._s,
                s_gt=u._s_gt,
                s_ref=u._s_ref,
                z_ref=u._z_ref,
                wall_dist=wall_dist,
                clearance=u._clearance,
            )

    traj = log.as_arrays(step_dt=dt)
    sway_step = cfg.scan.ref_step_s if cfg.scan.ref_step_s > 0.0 else cfg.scan.ref_step
    metrics = em.compute_metrics(
        traj,
        step_dt=dt,
        d_ref=cfg.d_ref,
        heave_target=cfg.scan.ref_step / dt if cfg.scan.ref_step > 0.0 else None,
        sway_target=sway_step / dt if sway_step > 0.0 else None,
        episode=score_episode,
        episode_length_s=cfg.episode_length_s,
    )
    metrics["task"] = args_cli.task
    metrics["checkpoint"] = args_cli.checkpoint

    tag = args_cli.tag
    npz_path = os.path.join(out_dir, f"trajectory_{tag}.npz")
    json_path = os.path.join(out_dir, f"metrics_{tag}.json")
    np.savez_compressed(npz_path, **{k: v for k, v in traj.items()})
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(em.format_metrics(metrics))
    print(f"[INFO] trajectory -> {npz_path}")
    print(f"[INFO] metrics    -> {json_path}")
    if not args_cli.no_plot:
        png = em.plot_trajectory(
            traj,
            os.path.join(out_dir, f"trajectory_{tag}.png"),
            episode=score_episode,
            title=f"{args_cli.task}  ({tag})",
        )
        print(f"[INFO] plot       -> {png}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.tam == "fixed" or args_cli.hydro == "z_slender":
        from marinelab.assets.pkrc import PKRCHydrodynamicsCfgZSlender, PKRCThrusterCfgFixedTAM

        if args_cli.tam == "fixed":
            env_cfg.thrusters = PKRCThrusterCfgFixedTAM()
        if args_cli.hydro == "z_slender":
            env_cfg.hydrodynamics = PKRCHydrodynamicsCfgZSlender()
        print(f"[INFO] plant overrides: tam={args_cli.tam} hydro={args_cli.hydro}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    if args_cli.log_traj:
        _run_with_trajectory_logging(env, policy, policy_nn, simulation_app)
        env.close()
        return

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
