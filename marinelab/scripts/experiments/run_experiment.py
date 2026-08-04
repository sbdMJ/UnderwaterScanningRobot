# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Single entry point for the competitor experiments (E1/E2/E3/E4a).

Runs every cell of a yaml experiment config — method x condition x seed — through the
``marinelab.control`` adapter layer, writing per-cell trajectory/metrics files in the
naming convention ``marinelab.experiments.protocol`` defines. One process, one Isaac app;
envs are rebuilt per cell (conditions may change the task).

Closed-loop structure for the MPC family is a faithful port of ``run_wallscan_mpc.py``
(controller-owned state machine, no spin search, GT or EKF state); the PPO path mirrors
``play.py --log_traj`` but runs the *exported* policy through ``PPOPolicyController`` so
sim and hardware share one inference implementation.

Examples::

    ./isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py \\
        --config ../marinelab/scripts/experiments/configs/e1_nominal.yaml
    # one cell only:
    ... --config .../e1_nominal.yaml --method fixed --seed 0
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Competitor experiment runner")
parser.add_argument("--config", type=str, required=True, help="experiment yaml (see configs/)")
parser.add_argument("--method", type=str, default=None, help="run only this method")
parser.add_argument("--cond", type=str, default=None, help="run only this condition")
parser.add_argument("--seed", type=int, default=None, help="run only this seed")
parser.add_argument("--results_root", type=str, default=None, help="default: <repo>/results")
parser.add_argument("--log_every", type=int, default=500)
parser.add_argument("--no_plot", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json

import numpy as np
import torch

import gymnasium as gym

from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_tasks  # noqa: F401

import marinelab  # noqa: F401

from marinelab.assets.pkrc import (
    PKRCHydrodynamicsCfg,
    PKRCHydrodynamicsCfgZSlender,
    PKRCThrusterCfg,
    PKRCThrusterCfgFixedTAM,
)
from marinelab.control import (
    DiffWMPCController,
    FixedWeightNMPC,
    PPOPolicyController,
    SensorSample,
    VehicleState,
    WallFrameStateEstimator,
)
from marinelab.control.types import ScanReference
from marinelab.experiments.protocol import ExperimentCell, load_cells
from marinelab.experiments.scoring import ScoreAccumulator
from marinelab.tasks.pkrc_wallscan import eval_metrics as em
from marinelab.tasks.pkrc_wallscan import geometry, mpc_reference as mref, scan_state_machine as ssm
from marinelab.tasks.pkrc_wallscan.mpc_controller import PlantParams
from marinelab.tasks.pkrc_wallscan.sensors import (
    SensorCfg, SensorCfgDatasheet, _body_up, att_noise, gyro_noise, ukfm_in_range,
)
from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import WallFrameEKFCfg
from marinelab.tasks.pkrc_wallscan.wallscan_env_cfg import (
    WallScanEvalCfg,
    WallScanStage3Cfg,
    WallScanTrainCfg,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TASK_CFGS = {
    "Isaac-PKRC-WallScan-Stage3-Direct-v0": WallScanStage3Cfg,
    "Isaac-PKRC-WallScan-Train-Direct-v0": WallScanTrainCfg,
    "Isaac-PKRC-WallScan-Eval-Direct-v0": WallScanEvalCfg,
}

MPC_METHODS = ("fixed", "bo", "diff", "ssi")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def build_env(cell: ExperimentCell):
    opt = cell.options
    task = opt["task"]
    if task not in TASK_CFGS:
        raise SystemExit(f"unknown task {task!r}; add it to TASK_CFGS (or env_variants)")
    cfg = TASK_CFGS[task]()
    cfg.scene.num_envs = int(opt.get("num_envs", 1))
    cfg.seed = cell.seed
    tam = opt.get("tam", "env")
    if tam == "fixed":
        cfg.thrusters = PKRCThrusterCfgFixedTAM()
    elif tam == "shipped":
        cfg.thrusters = PKRCThrusterCfg()
    hydro = opt.get("hydro", "env")
    if hydro == "z_slender":
        cfg.hydrodynamics = PKRCHydrodynamicsCfgZSlender()
    elif hydro == "shipped":
        cfg.hydrodynamics = PKRCHydrodynamicsCfg()
    return gym.make(task, cfg=cfg).unwrapped, cfg


def build_controller(cell: ExperimentCell, env, cfg):
    """Returns (controller, mpc_cfg) — mpc_cfg is None for policy-only methods."""
    opt = cell.options
    if cell.method == "ppo":
        return PPOPolicyController(_resolve(opt["policy"])), None
    prm = PlantParams.from_env(env)
    mpc_cfg = mref.WallScanMPCCfg(
        tank_radius=cfg.tank_radius, d_ref=cfg.d_ref,
        z_top=cfg.scan.z_top, z_bottom=cfg.scan.z_bottom, sway_step=cfg.scan.sway_step,
        ref_step=cfg.scan.ref_step, ref_step_s=cfg.scan.ref_step_s,
        step_dt=env.step_dt, dt_mpc=float(opt.get("dt_mpc", 0.05)),
    )
    nmpc_kwargs = dict(
        plant=prm, mpc_cfg=mpc_cfg, horizon=int(opt.get("horizon", 30)),
        rti_iters=int(opt.get("rti_iters", 8)),
        code_export_root=os.path.join(REPO_ROOT, "isaaclab", "logs", "c_generated_code_wallscan"),
    )
    if cell.method == "fixed":
        return FixedWeightNMPC(**nmpc_kwargs), mpc_cfg
    if cell.method == "bo":
        return FixedWeightNMPC(params_json=_resolve(opt["params_json"]), **nmpc_kwargs), mpc_cfg
    if cell.method == "diff":
        return DiffWMPCController(ckpt_path=_resolve(opt["ckpt"]), **nmpc_kwargs), mpc_cfg
    if cell.method == "ssi":
        raise SystemExit("ssi: port pending (docs/competitor_framework_plan.md §4 ④)")
    raise SystemExit(f"unknown method {cell.method!r}")


class SimSensorStream:
    """Synthesize SensorSample streams from GT — port of run_wallscan_mpc's --state ekf block."""

    def __init__(self, cfg, opt: dict, seed: int):
        scfg_cls = SensorCfgDatasheet if opt.get("sensors", "placeholder") == "datasheet" else SensorCfg
        self.scfg = scfg_cls(ukfm_gate=opt.get("ukfm_gate", "depth_below_surface"),
                             ukfm_surface_z=cfg.tank_height)
        # Bias half-ranges are DR knobs (0 in Stage3); the estimator condition needs them on.
        if scfg_cls is SensorCfgDatasheet:
            self.scfg.sonar_bias_dr = SensorCfgDatasheet.sonar_bias_dr
            self.scfg.ins_att_bias_dr = SensorCfgDatasheet.ins_att_bias_dr
        else:
            self.scfg.sonar_bias_dr = 0.10
            self.scfg.ins_att_bias_dr = 0.04
        self.scfg.depth_bias_dr = 0.10
        self.scfg.dvl_bias_dr = 0.01
        self.gyro_bias = float(opt.get("gyro_bias", self.scfg.ins_gyro_bias_dr
                                       if scfg_cls is SensorCfgDatasheet else 0.02))
        self.gyro_noise_std = gyro_noise(self.scfg)
        self.rng = np.random.default_rng(seed)
        self.tank_radius = cfg.tank_radius
        self.bias: dict = {}
        self.redraw_bias()

    def redraw_bias(self) -> None:
        s, rng = self.scfg, self.rng
        self.bias = {
            "sonar": float(rng.uniform(-s.sonar_bias_dr, s.sonar_bias_dr)) if s.sonar_bias_dr else 0.0,
            "dvl": rng.uniform(-s.dvl_bias_dr, s.dvl_bias_dr, size=2) if s.dvl_bias_dr else np.zeros(2),
            "depth": float(rng.uniform(-s.depth_bias_dr, s.depth_bias_dr)) if s.depth_bias_dr else 0.0,
            "att": (rng.uniform(-s.ins_att_bias_dr, s.ins_att_bias_dr, size=2)
                    if s.ins_att_bias_dr else np.zeros(2)),
        }

    def sample(self, env, stamp: float) -> tuple[SensorSample, dict]:
        """One tick's synthesized readings for env 0, plus the truth used to make them."""
        rng, s = self.rng, self.scfg
        pos = (env._robot.data.root_pos_w - env.scene.env_origins)
        quat = env._robot.data.root_quat_w
        roll_t, pitch_t, yaw = euler_xyz_from_quat(quat)
        theta = torch.atan2(pos[:, 1], pos[:, 0])
        truth = {
            "r": float(torch.linalg.norm(pos[0, :2])),
            "phi": float(mref._wrap_to_pi(yaw[0] - theta[0])),
        }
        # Truth from the nominal mount: the 0.10 m offset stays unmodelled by the filter,
        # a deliberate realistic model error (see run_wallscan_mpc.py).
        sonar_true = float(geometry.sonar_wall_distance(
            pos[:, :2], yaw, env._sonar_mount_nom, env._sonar_yaw_nom, self.tank_radius)[0])
        v_b = env._robot.data.root_lin_vel_b[0]
        w_b = env._robot.data.root_ang_vel_b[0]
        v_meas = v_b[:2].cpu().numpy() + rng.normal(0, s.dvl_noise, 2) + self.bias["dvl"]
        ukfm = None
        if bool(ukfm_in_range(pos[:, 2], s)[0]):
            ukfm = (truth["r"] + float(rng.normal(0, s.ukfm_noise)),
                    truth["phi"] + float(rng.normal(0, s.ukfm_noise)))
        sample = SensorSample(
            v_bx=float(v_meas[0]), v_by=float(v_meas[1]),
            gyro_z=float(w_b[2]) + float(rng.normal(0, self.gyro_noise_std)) + self.gyro_bias,
            sonar=sonar_true + float(rng.normal(0, s.sonar_noise)) + self.bias["sonar"],
            depth=float(pos[0, 2]) + float(rng.normal(0, s.depth_noise)) + self.bias["depth"],
            roll=float(roll_t[0]) + float(rng.normal(0, att_noise(s))) + self.bias["att"][0],
            pitch=float(pitch_t[0]) + float(rng.normal(0, att_noise(s))) + self.bias["att"][1],
            v_bz=float(v_b[2]), gyro_x=float(w_b[0]), gyro_y=float(w_b[1]),
            ukfm=ukfm, stamp=stamp,
        )
        return sample, truth


def _gt_errors(env, pos, quat, z_ref, s_ref, prev_z, prev_s, theta, s_anchor, dt, mpc_cfg):
    """(n, 12) wallscan error vector on GROUND TRUTH — the scoring input (plan §10-2)."""
    x = torch.cat([pos, quat, env._robot.data.root_lin_vel_b,
                   env._robot.data.root_ang_vel_b], dim=-1)
    v_z_des = (z_ref - prev_z) / dt if prev_z is not None else torch.zeros_like(z_ref)
    v_tan_des = (s_ref - prev_s) / dt if prev_s is not None else torch.zeros_like(s_ref)
    return mref.wallscan_errors(x, z_ref=z_ref, s_ref=s_ref, v_tan_des=v_tan_des,
                                v_z_des=v_z_des, theta_anchor=theta, s_anchor=s_anchor,
                                cfg=mpc_cfg).cpu().numpy()


def run_mpc_cell(cell: ExperimentCell, env, cfg, ctl, steps: int, mpc_cfg,
                 score: ScoreAccumulator) -> dict:
    """Controller-owned closed loop (no spin search) — port of run_wallscan_mpc.main."""
    if env.num_envs != 1:
        raise SystemExit(f"MPC methods run num_envs=1 (acados is sequential); got {env.num_envs}")
    opt = cell.options
    dev, dt = env.device, env.step_dt
    use_ekf = opt.get("state", "gt") == "ekf"
    scan_cfg = cfg.scan
    state_sm = ssm.ScanState(1, device=dev)
    s_gt = torch.zeros(1, device=dev)
    theta_prev = torch.zeros(1, device=dev)
    cycles = torch.zeros(1, dtype=torch.long, device=dev)
    stream = SimSensorStream(cfg, opt, cell.seed) if use_ekf else None
    estimator = WallFrameStateEstimator(WallFrameEKFCfg(
        tank_radius=cfg.tank_radius,
        r_sonar=stream.scfg.sonar_noise, sonar_mount_pos=stream.scfg.sonar_mount_pos,
        sonar_yaw_offset=stream.scfg.sonar_yaw_offset)) if use_ekf else None
    est_err = {"r": [], "phi": [], "s": []}

    def gt_pose():
        pos = env._robot.data.root_pos_w - env.scene.env_origins
        quat = env._robot.data.root_quat_w
        _, _, yaw = euler_xyz_from_quat(quat)
        return pos, quat, yaw

    def reset_internal():
        pos, _, yaw = gt_pose()
        theta = torch.atan2(pos[:, 1], pos[:, 0])
        state_sm.phase[:] = 0
        state_sm.s_ref[:] = 0.0
        state_sm.sway_dir[:] = 1.0
        state_sm.z_hold[:] = 0.0
        state_sm._hold[:] = 0
        state_sm.z_ramp[:] = pos[:, 2]
        state_sm.s_ramp[:] = 0.0
        s_gt[:] = 0.0
        cycles[:] = 0
        theta_prev[:] = theta
        ctl.reset(VehicleState.from_x13(np.zeros(13)))
        if use_ekf:
            stream.redraw_bias()
            estimator.reset(r0=float(torch.linalg.norm(pos[0, :2])),
                            phi0=float(mref._wrap_to_pi(yaw[0] - theta[0])),
                            theta0=float(theta[0]))

    env.reset()
    env.episode_length_buf[:] = 0  # own the whole window (see run_wallscan_mpc / CLAUDE.md)
    reset_internal()

    log = em.TrajectoryLog()
    action = torch.zeros(1, 6, device=dev)
    prev_z_ref = prev_s_ref = None
    for i in range(steps):
        if not simulation_app.is_running():
            print(f"[WARN] app closed early at {i}/{steps}")
            break
        pos, quat, yaw = gt_pose()
        theta = torch.atan2(pos[:, 1], pos[:, 0])
        s_gt += mref._wrap_to_pi(theta - theta_prev) * cfg.tank_radius
        theta_prev = theta.clone()

        if use_ekf:
            sample, truth = stream.sample(env, stamp=i * dt)
            veh = estimator.step(sample, dt)
            est_err["r"].append(estimator.ekf.r - truth["r"])
            est_err["phi"].append(float(mref._wrap_to_pi(
                torch.tensor(estimator.ekf.phi - truth["phi"]))))
            est_err["s"].append(estimator.s_hat - float(s_gt[0]))
            z_sm = torch.tensor([veh.pos_w[2]], device=dev)
            s_sm = torch.tensor([estimator.s_hat], device=dev)
            theta_anchor, s_anchor = estimator.theta_hat, estimator.s_hat
        else:
            x13 = torch.cat([pos, quat,
                             env._robot.data.root_lin_vel_b,
                             env._robot.data.root_ang_vel_b], dim=-1)[0].cpu().numpy()
            veh = VehicleState.from_x13(x13, stamp=i * dt)
            z_sm, s_sm = pos[:, 2], s_gt
            theta_anchor, s_anchor = float(theta[0]), float(s_gt[0])

        z_ref, s_ref, _phase_sc, advanced = ssm.step(state_sm, z_sm, s_sm, scan_cfg, z_latch=z_sm)
        cycles += (advanced & (state_sm.phase == 0)).long()
        preview = mref.reference_preview(
            state_sm.phase, state_sm.z_ramp, state_sm.s_ramp, state_sm.s_ref, state_sm.z_hold,
            mpc_cfg, int(cell.options.get("horizon", 30)),
        )
        ref = ScanReference(
            z_ref=preview["z_ref"][0].cpu().numpy(), s_ref=preview["s_ref"][0].cpu().numpy(),
            v_tan_des=preview["v_tan_des"][0].cpu().numpy(),
            v_z_des=preview["v_z_des"][0].cpu().numpy(),
            theta_anchor=theta_anchor, s_anchor=s_anchor, phase=int(state_sm.phase[0]),
        )
        out = ctl.step(veh, ref)
        e_score = _gt_errors(env, pos, quat, z_ref, s_ref, prev_z_ref, prev_s_ref,
                             theta, s_gt, dt, mpc_cfg)
        prev_z_ref, prev_s_ref = z_ref.clone(), s_ref.clone()
        action[0] = torch.as_tensor(out.u_cmd, dtype=torch.float32, device=dev)
        _obs, _rew, terminated, truncated, _info = env.step(action)
        dones = terminated | truncated
        score.add(e_score, out.u_cmd[None], done=dones.cpu().numpy(),
                  collided=env._term_collided.cpu().numpy())

        pos_n, quat_n, yaw_n = gt_pose()
        up_z = _body_up(quat_n)[:, 2].clamp(-1.0, 1.0)
        wall_dist = geometry.sonar_wall_distance(
            pos_n[:, :2], yaw_n, env._sonar_mount_nom, env._sonar_yaw_nom, cfg.tank_radius)
        log.record(
            phase=state_sm.phase, cycles=cycles,
            searching=torch.zeros(1, dtype=torch.bool, device=dev),
            done=dones, terminated=env.reset_terminated, time_out=env.reset_time_outs,
            term_collided=env._term_collided, term_oob=env._term_oob, term_tilted=env._term_tilted,
            x=pos_n[:, 0], y=pos_n[:, 1], z=pos_n[:, 2], yaw=yaw_n,
            theta=torch.atan2(pos_n[:, 1], pos_n[:, 0]),
            tilt_deg=torch.rad2deg(torch.arccos(up_z)),
            s=s_gt, s_gt=s_gt, s_ref=s_ref, z_ref=z_ref,
            wall_dist=wall_dist, clearance=env._clearance,
        )
        if dones.any():
            reset_internal()
            prev_z_ref = prev_s_ref = None
        if args_cli.log_every and i % args_cli.log_every == 0:
            print(f"  t={i * dt:6.1f}s ph={int(state_sm.phase[0])} z={float(pos_n[0, 2]):5.2f} "
                  f"s={float(s_gt[0]):+6.2f} solve={out.solve_ms:.1f}ms")

    extras = {}
    if use_ekf:
        e = {k: np.asarray(v) for k, v in est_err.items()}
        extras["estimator"] = {
            "sensors": opt.get("sensors", "placeholder"),
            "r_rmse_m": float(np.sqrt(np.mean(e["r"] ** 2))),
            "phi_rmse_deg": float(np.degrees(np.sqrt(np.mean(e["phi"] ** 2)))),
            "s_rmse_m": float(np.sqrt(np.mean(e["s"] ** 2))),
        }
    return {"log": log, "extras": extras}


def run_ppo_cell(cell: ExperimentCell, env, cfg, ctl, steps: int, mpc_cfg,
                 score: ScoreAccumulator) -> dict:
    """Env-owned scan (spin search included) with the exported policy — mirrors play.py."""
    dev, dt = env.device, env.step_dt
    obs_dict, _ = env.reset()
    env.episode_length_buf[:] = 0  # same protocol as the MPC path: full-length first episode
    log = em.TrajectoryLog()
    n = env.num_envs
    action = torch.zeros(n, 6, device=dev)
    prev_z_ref = prev_s_ref = None
    for i in range(steps):
        if not simulation_app.is_running():
            print(f"[WARN] app closed early at {i}/{steps}")
            break
        obs = obs_dict["policy"].cpu().numpy()
        for e in range(n):
            action[e] = torch.as_tensor(ctl.step(None, None, obs[e]).u_cmd,
                                        dtype=torch.float32, device=dev)
        # pre-step snapshot for scoring: the state/references this action was taken against
        pos_b = env._robot.data.root_pos_w - env.scene.env_origins
        quat_b = env._robot.data.root_quat_w
        theta_b = torch.atan2(pos_b[:, 1], pos_b[:, 0])
        z_ref_b, s_ref_b = env._z_ref.clone(), env._s_ref.clone()
        s_gt_b = env._s_gt.clone()
        e_score = _gt_errors(env, pos_b, quat_b, z_ref_b, s_ref_b, prev_z_ref, prev_s_ref,
                             theta_b, s_gt_b, dt, mpc_cfg)
        prev_z_ref, prev_s_ref = z_ref_b, s_ref_b
        obs_dict, _rew, terminated, truncated, _info = env.step(action)
        dones = terminated | truncated
        score.add(e_score, action.cpu().numpy(), done=dones.cpu().numpy(),
                  collided=env._term_collided.cpu().numpy())
        if dones.any():
            prev_z_ref = prev_s_ref = None

        pos = env._robot.data.root_pos_w - env.scene.env_origins
        quat = env._robot.data.root_quat_w
        _, _, yaw = euler_xyz_from_quat(quat)
        up_z = _body_up(quat)[:, 2].clamp(-1.0, 1.0)
        wall_dist = geometry.sonar_wall_distance(
            pos[:, :2], yaw, env._sonar_mount_nom, env._sonar_yaw_nom, cfg.tank_radius)
        log.record(
            phase=env._scan_state.phase, cycles=env._cycles, searching=env._search_active,
            done=dones, terminated=env.reset_terminated, time_out=env.reset_time_outs,
            term_collided=env._term_collided, term_oob=env._term_oob, term_tilted=env._term_tilted,
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], yaw=yaw,
            theta=torch.atan2(pos[:, 1], pos[:, 0]),
            tilt_deg=torch.rad2deg(torch.arccos(up_z)),
            s=env._s, s_gt=env._s_gt, s_ref=env._s_ref, z_ref=env._z_ref,
            wall_dist=wall_dist, clearance=env._clearance,
        )
        if args_cli.log_every and i % args_cli.log_every == 0:
            print(f"  t={i * dt:6.1f}s ph={int(env._scan_state.phase[0])} "
                  f"cycles={int(env._cycles[0])}")
    return {"log": log, "extras": {}}


def run_cell(cell: ExperimentCell, results_root: str) -> None:
    opt = cell.options
    steps = int(opt.get("steps", 9000))
    print(f"\n[CELL] {cell.exp} / {cell.tag}  task={opt['task']}  steps={steps}")
    env, cfg = build_env(cell)
    try:
        ctl, mpc_cfg = build_controller(cell, env, cfg)
        if mpc_cfg is None:  # policy-only methods still need the error-vector cfg for scoring
            mpc_cfg = mref.WallScanMPCCfg(
                tank_radius=cfg.tank_radius, d_ref=cfg.d_ref,
                z_top=cfg.scan.z_top, z_bottom=cfg.scan.z_bottom, sway_step=cfg.scan.sway_step,
                ref_step=cfg.scan.ref_step, ref_step_s=cfg.scan.ref_step_s,
                step_dt=env.step_dt, dt_mpc=float(opt.get("dt_mpc", 0.05)),
            )
        score = ScoreAccumulator(env.num_envs)
        runner = run_ppo_cell if cell.method == "ppo" else run_mpc_cell
        with torch.inference_mode():
            result = runner(cell, env, cfg, ctl, steps, mpc_cfg, score)

        dt = env.step_dt
        traj = result["log"].as_arrays(step_dt=dt)
        sway_step = cfg.scan.ref_step_s if cfg.scan.ref_step_s > 0.0 else cfg.scan.ref_step
        score_episode = opt.get("score_episode", 0)
        metrics = em.compute_metrics(
            traj, step_dt=dt, d_ref=cfg.d_ref,
            heave_target=cfg.scan.ref_step / dt if cfg.scan.ref_step > 0.0 else None,
            sway_target=sway_step / dt if sway_step > 0.0 else None,
            episode=None if score_episode < 0 else score_episode,
            episode_length_s=cfg.episode_length_s,
        )
        score.finalize()
        score_summary = score.summary(score_episode=max(0, score_episode))
        metrics.update({"exp": cell.exp, "method": cell.method, "cond": cell.cond,
                        "seed": cell.seed, "options": {k: v for k, v in opt.items()},
                        "controller_cost": ctl.stats.summary(),
                        "score": {k: score_summary[k] for k in
                                  ("objective", "collided", "scored_losses")}})
        metrics.update(result["extras"])

        out_dir = cell.out_dir(results_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cell.trajectory_path(results_root), **traj)
        with open(cell.metrics_path(results_root), "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(em.format_metrics(metrics))
        cost = metrics["controller_cost"]
        print(f"[COST] {cost['solve_ms_mean']:.2f} ms/step (p95 {cost['solve_ms_p95']:.2f})  "
              f"fail {100 * cost['fail_frac']:.2f}%  sat {100 * cost['saturated_frac']:.2f}%")
        if not args_cli.no_plot:
            em.plot_trajectory(traj, str(out_dir / f"trajectory_{cell.tag}.png"),
                               episode=None if score_episode < 0 else score_episode,
                               title=f"{cell.method} [{cell.tag}]")
    finally:
        env.close()


def main() -> None:
    cells = load_cells(args_cli.config, only_method=args_cli.method,
                       only_cond=args_cli.cond, only_seed=args_cli.seed)
    if not cells:
        raise SystemExit("no cells match the given filters")
    results_root = os.path.abspath(args_cli.results_root or os.path.join(REPO_ROOT, "results"))
    print(f"[INFO] {len(cells)} cell(s) -> {results_root}")
    for cell in cells:
        run_cell(cell, results_root)


if __name__ == "__main__":
    main()
    simulation_app.close()
