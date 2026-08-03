# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Phase 3: pure-NMPC wallscan baseline (no learning), scored with the RL run's own metrics.

Drives ``WallScanEnv`` with actions from :class:`WallScanMPC` instead of a policy. The env is
NOT modified: it still computes its 15-term reward and ``Scan/*`` telemetry, which are simply
ignored, so ``eval_metrics`` numbers land on exactly the same footing as
``play.py --log_traj`` and can be put in one table next to the trained policy.

Two things this baseline deliberately does differently from the RL policy:

* **No spin search.** The RL env sweeps yaw for ~10 s to find the nearest wall by sonar
  minimum, and a 07-27 audit found that bearing landed 5-84 deg off the true normal. An MPC
  that knows its position does not need it: on a cylinder the wall normal IS the outward
  radial, so the heading reference is closed-loop from position (``yaw_offset = pi``).
* **Reference preview.** The state machine's ramp is rolled forward over the whole horizon
  (``mpc_reference.reference_preview``) rather than presented as a single frozen setpoint,
  so the solver decelerates into a phase endpoint instead of overshooting it.

State comes from GROUND TRUTH here (plan Phase 2a). That is on purpose: it isolates "does
the MPC formulation win" from "is the state estimator good enough", which is the separate
and harder Phase 2. Numbers from this script are therefore an upper bound, directly
comparable to the RL policy's own GT-referenced heading (``_yaw_ref_cur = theta_gt``).

Example (inside the container, needs the acados mount):

    ./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/run_wallscan_mpc.py \\
        --tam fixed --steps 9000 --tag mpc_fixed_tam'
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pure-NMPC wallscan baseline.")
parser.add_argument("--tam", choices=["shipped", "fixed"], default="fixed",
                    help="Thruster allocation matrix. 'fixed' moves the sway moment arm to the "
                         "roll row, where a heave differential can cancel it (measured 2026-07-30).")
parser.add_argument("--hydro", choices=["shipped", "z_slender"], default="shipped",
                    help="Hydrodynamic coefficients. 'shipped' claims an x-slender hull, which "
                         "contradicts the USD mesh; 'z_slender' matches the actual (vertical) hull "
                         "and roughly halves the heave drag -- the wallscan's primary axis.")
parser.add_argument("--thruster_tau", type=float, default=None,
                    help="Override BOTH thruster time constants [s]. Diagnostic: the MPC model "
                         "assumes the commanded force appears instantly, while the plant lags "
                         "(tau_up 0.1 / tau_down 0.05). Passing the control step (0.02) makes the "
                         "actuator instantaneous, so plant and model agree and the lag's "
                         "contribution to the heave overshoot is isolated. Values below the "
                         "control step are rejected: alpha = dt/tau > 1 makes the first-order "
                         "update overshoot its own target.")
parser.add_argument("--steps", type=int, default=9000, help="Control steps (9000 = one 180 s episode).")
parser.add_argument("--num_envs", type=int, default=1, help="acados solves sequentially; keep small.")
parser.add_argument("--horizon", type=int, default=30, help="MPC stages.")
parser.add_argument("--dt_mpc", type=float, default=0.05, help="MPC stage length [s].")
parser.add_argument("--rti_iters", type=int, default=8)
parser.add_argument("--tag", type=str, default="mpc")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--score_episode", type=int, default=0)
parser.add_argument("--settle_s", type=float, default=20.0,
                    help="Seconds dropped from the start of each episode for the additional "
                         "'settled' metric block. The spawn hands the vehicle up to 180 deg of "
                         "heading error and correcting it is a transient: measured 2026-07-31, "
                         "the worst of three seeds averaged 18.7 deg of crab over its first 20 s "
                         "and 0.000-0.140 deg for the remaining 160 s. Full-window figures are "
                         "still reported alongside. 0 disables.")
parser.add_argument("--no_plot", action="store_true", default=False)
parser.add_argument("--log_every", type=int, default=250)
parser.add_argument("--seed", type=int, default=None,
                    help="Seeds the env (hence the spawn disk sample and attitude draw) AND this "
                         "script's sensor-noise stream. Without it every run gets a different "
                         "spawn, and the initial transient dominates the transient-sensitive "
                         "metrics — run-to-run spread was measured as large as the differences "
                         "between the configurations being compared.")
parser.add_argument("--state", choices=["gt", "ekf"], default="gt",
                    help="State fed to the MPC. 'gt' is the Phase 2a upper bound; 'ekf' runs the "
                         "wall-frame (r, phi, s) filter off synthesized sensor streams, which is "
                         "what a real deployment would have.")
parser.add_argument("--ukfm_gate", choices=["legacy_height", "depth_below_surface"],
                    default="depth_below_surface",
                    help="UKF-M validity semantics for --state ekf. The legacy gate is inverted; "
                         "see SensorCfg.ukfm_gate.")
parser.add_argument("--sensors", choices=["placeholder", "datasheet"], default="placeholder",
                    help="Sensor model for --state ekf. 'datasheet' uses the 3DM-GV7 and Ping1D "
                         "published figures instead of the conservative guesses in SensorCfg; the "
                         "gyro turn-on bias alone is 213x smaller.")
parser.add_argument("--gyro_bias", type=float, default=None,
                    help="Override the gyro bias [rad/s]. Default: taken from the chosen --sensors "
                         "model (0.02 placeholder, 9.4e-5 datasheet).")
parser.add_argument("--policy_ckpt", type=str, default=None,
                    help="Diff-WMPC weight-policy checkpoint. When given, the MPC cost weights come "
                         "from the policy every step instead of the fixed DEFAULT_WERR, so the same "
                         "metrics path compares learned vs hand-tuned weights directly.")
parser.add_argument("--w_roll", type=float, default=None,
                    help="Override the roll/pitch cost weights. Under the FIXED TAM a high value is\n                         what makes the solver spend heave differential on cancelling the sway\n                         leg's parasitic moment; under the shipped TAM it cannot help (pitch has\n                         no other actuator), so this is the decisive A/B.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import time

import numpy as np
import torch

import gymnasium as gym

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import isaaclab_tasks  # noqa: F401

import marinelab  # noqa: F401

from marinelab.assets.pkrc import (
    PKRCHydrodynamicsCfg,
    PKRCHydrodynamicsCfgZSlender,
    PKRCThrusterCfg,
    PKRCThrusterCfgFixedTAM,
)
from marinelab.tasks.pkrc_wallscan import eval_metrics as em
from marinelab.tasks.pkrc_wallscan import geometry, mpc_reference as mref, scan_state_machine as ssm
from marinelab.tasks.pkrc_wallscan.mpc_controller import DEFAULT_WERR, DEFAULT_WU, PlantParams, WallScanMPC
from marinelab.tasks.pkrc_wallscan.sensors import (
    SensorCfg, SensorCfgDatasheet, _body_up, att_noise, ukfm_in_range,
)
from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import WallFrameEKF, WallFrameEKFCfg
from marinelab.tasks.pkrc_wallscan.wallscan_env_cfg import WallScanStage3Cfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    cfg = WallScanStage3Cfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed  # DirectRLEnv seeds torch from this in __init__
    cfg.thrusters = PKRCThrusterCfgFixedTAM() if args_cli.tam == "fixed" else PKRCThrusterCfg()
    cfg.hydrodynamics = (PKRCHydrodynamicsCfgZSlender() if args_cli.hydro == "z_slender"
                         else PKRCHydrodynamicsCfg())
    if args_cli.thruster_tau is not None:
        if args_cli.thruster_tau < cfg.sim.dt * cfg.decimation:
            raise SystemExit(f"--thruster_tau must be >= the control step "
                             f"{cfg.sim.dt * cfg.decimation}; alpha = dt/tau > 1 is unstable")
        cfg.thrusters.time_constant_up = args_cli.thruster_tau
        cfg.thrusters.time_constant_down = args_cli.thruster_tau

    env = gym.make("Isaac-PKRC-WallScan-Stage3-Direct-v0", cfg=cfg).unwrapped
    dev, N = env.device, env.num_envs
    dt = env.step_dt

    prm = PlantParams.from_env(env)
    mpc_cfg = mref.WallScanMPCCfg(
        tank_radius=cfg.tank_radius, d_ref=cfg.d_ref,
        z_top=cfg.scan.z_top, z_bottom=cfg.scan.z_bottom, sway_step=cfg.scan.sway_step,
        ref_step=cfg.scan.ref_step, ref_step_s=cfg.scan.ref_step_s,
        step_dt=dt, dt_mpc=args_cli.dt_mpc,
    )
    print(f"[INFO] TAM={args_cli.tam}  r_des={mpc_cfg.r_des:.2f} m  "
          f"horizon={args_cli.horizon}x{args_cli.dt_mpc}s={args_cli.horizon * args_cli.dt_mpc:.2f}s")
    print(f"[INFO] plant: m={prm.mass} I={prm.rigid_body_inertia} B={prm.buoyancy_force:.2f} N "
          f"max_thrust={prm.max_thrust} N")

    mpc = WallScanMPC(
        prm, mpc_cfg, N=args_cli.horizon, rti_iters=args_cli.rti_iters,
        with_sensitivity=False,  # the baseline needs no gradients; Phase 4 turns this on
        code_export_root=os.path.join(REPO_ROOT, "isaaclab", "logs", "c_generated_code_wallscan"),
    )
    werr = np.asarray(DEFAULT_WERR, float)
    if args_cli.w_roll is not None:
        werr[8] = werr[9] = args_cli.w_roll
    weights = np.concatenate([werr, np.full(mpc.nu, DEFAULT_WU)])
    print(f"[INFO] w_roll/pitch = {werr[8]:.1f}")

    policy = None
    if args_cli.policy_ckpt:
        from marinelab.algorithms.diff_wmpc import WeightPolicy
        from marinelab.tasks.pkrc_wallscan.mpc_reference import NE as _NE

        policy = WeightPolicy(_NE + 2, _NE, mpc.nu, werr_init=werr,
                              wu_init=np.full(mpc.nu, DEFAULT_WU))
        st = torch.load(args_cli.policy_ckpt, map_location="cpu")
        policy.load_state_dict(st["policy"] if "policy" in st else st)
        policy.eval()
        print(f"[INFO] weights from Diff-WMPC policy {args_cli.policy_ckpt}")

    # Our own scan state machine: the env runs one too, but its phase timing is gated by the
    # spin search we are skipping, so the controller must own the reference it tracks.
    scan_cfg = cfg.scan
    state = ssm.ScanState(N, device=dev)
    s_gt = torch.zeros(N, device=dev)
    theta_prev = torch.zeros(N, device=dev)
    # Our own cycle count. env._cycles belongs to the ENV's state machine, which is gated by
    # the spin search this controller skips, so it stays 0 and would misreport the metric.
    cycles = torch.zeros(N, dtype=torch.long, device=dev)

    def reset_internal():
        pos = env._robot.data.root_pos_w - env.scene.env_origins
        theta = torch.atan2(pos[:, 1], pos[:, 0])
        state.phase[:] = 0
        state.s_ref[:] = 0.0
        state.sway_dir[:] = 1.0
        state.z_hold[:] = 0.0
        state._hold[:] = 0
        state.z_ramp[:] = pos[:, 2]
        state.s_ramp[:] = 0.0
        s_gt[:] = 0.0
        cycles[:] = 0
        theta_prev[:] = theta

    # --- estimator (only used with --state ekf) -----------------------------
    scfg_cls = SensorCfgDatasheet if args_cli.sensors == "datasheet" else SensorCfg
    scfg = scfg_cls(ukfm_gate=args_cli.ukfm_gate, ukfm_surface_z=cfg.tank_height)
    # Bias half-ranges are DR knobs (0 in Stage3); this measurement needs the biases present,
    # so use each model's published magnitude directly.
    if args_cli.sensors == "datasheet":
        scfg.sonar_bias_dr = SensorCfgDatasheet.sonar_bias_dr
        scfg.depth_bias_dr = 0.10
        scfg.dvl_bias_dr = 0.01
        scfg.ins_att_bias_dr = SensorCfgDatasheet.ins_att_bias_dr
    else:
        scfg.sonar_bias_dr = 0.10
        scfg.depth_bias_dr = 0.10
        scfg.dvl_bias_dr = 0.01
        scfg.ins_att_bias_dr = 0.04
    gyro_bias = (args_cli.gyro_bias if args_cli.gyro_bias is not None
                 else scfg_cls.ins_gyro_bias_dr if args_cli.sensors == "datasheet" else 0.02)
    print(f"[INFO] sensors={args_cli.sensors}  gyro_bias={gyro_bias:.3e}  "
          f"sonar_noise={scfg.sonar_noise}  att_bias={scfg.ins_att_bias_dr:.4f}")
    from marinelab.tasks.pkrc_wallscan.sensors import gyro_noise as _gyro_noise
    gyro_noise_std = _gyro_noise(scfg)
    ekf: WallFrameEKF | None = None
    theta_hat = torch.zeros(N, device=dev)
    rng = np.random.default_rng(args_cli.seed if args_cli.seed is not None else 0)
    sensor_bias = {}

    def harvest_counts():
        if ekf is not None:
            est_counts["ukfm"] += ekf.n_ukfm
            est_counts["gated"] += ekf.n_gated

    def reset_estimator():
        """Fresh filter + a new per-episode sensor bias draw, seeded at the true state.

        Seeding at truth is deliberate: this measurement isolates how the estimate DEGRADES
        over a mission, not how it converges from a cold start (the spin search would own
        that, and this controller skips it).
        """
        nonlocal ekf, sensor_bias
        harvest_counts()
        pos = env._robot.data.root_pos_w - env.scene.env_origins
        r0 = float(torch.linalg.norm(pos[0, :2]))
        th0 = float(torch.atan2(pos[0, 1], pos[0, 0]))
        _, _, yaw0 = euler_xyz_from_quat(env._robot.data.root_quat_w)
        ekf = WallFrameEKF(WallFrameEKFCfg(
            tank_radius=cfg.tank_radius, r_sonar=scfg.sonar_noise,
            # The transducer is not at the body origin; omitting this put the whole 10 cm
            # offset into the r estimate (measured +5.3 cm, worsening to ~10 cm once the sonar
            # noise was set from the Ping1D datasheet).
            sonar_mount_pos=scfg.sonar_mount_pos, sonar_yaw_offset=scfg.sonar_yaw_offset,
            initial=(r0, float(mref._wrap_to_pi(yaw0[0] - torch.tensor(th0))), 0.0),
        ))
        theta_hat[:] = th0
        sensor_bias = {
            "sonar": float(rng.uniform(-scfg.sonar_bias_dr, scfg.sonar_bias_dr)) if scfg.sonar_bias_dr else 0.0,
            "dvl": rng.uniform(-scfg.dvl_bias_dr, scfg.dvl_bias_dr, size=2) if scfg.dvl_bias_dr else np.zeros(2),
            "depth": float(rng.uniform(-scfg.depth_bias_dr, scfg.depth_bias_dr)) if scfg.depth_bias_dr else 0.0,
            "att": (rng.uniform(-scfg.ins_att_bias_dr, scfg.ins_att_bias_dr, size=2)
                    if scfg.ins_att_bias_dr else np.zeros(2)),
        }

    env.reset()
    env.episode_length_buf[:] = 0  # own the whole window; see compute_metrics' episode-0 note
    reset_internal()
    if args_cli.state == "ekf":
        reset_estimator()

    log = em.TrajectoryLog()
    est_err = {"r": [], "phi": [], "s": []}
    # Accumulated across segments: reset_estimator() builds a FRESH filter, so reading
    # ekf.n_ukfm at the end would report only what happened after the last reset (the timeout
    # fires on the final step, which made this read 0-1 instead of thousands).
    est_counts = {"ukfm": 0, "gated": 0}
    action = torch.zeros(N, 6, device=dev)
    first_solve = True
    t_solve = 0.0
    n_solve = 0
    n_fail = 0
    sat_steps = 0

    print(f"[INFO] {args_cli.steps} steps x {N} envs ({args_cli.steps * dt:.0f} s of sim)")
    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            print(f"[WARN] app closed early at {i}/{args_cli.steps}")
            break

        pos = env._robot.data.root_pos_w - env.scene.env_origins
        quat = env._robot.data.root_quat_w
        v_b = env._robot.data.root_lin_vel_b
        w_b = env._robot.data.root_ang_vel_b
        _, _, yaw = euler_xyz_from_quat(quat)

        theta = torch.atan2(pos[:, 1], pos[:, 0])
        s_gt += mref._wrap_to_pi(theta - theta_prev) * cfg.tank_radius
        theta_prev = theta

        # --- estimator: synthesize the sensor streams, filter, rebuild the MPC state -------
        if args_cli.state == "ekf":
            r_true = float(torch.linalg.norm(pos[0, :2]))
            phi_true = float(mref._wrap_to_pi(yaw[0] - theta[0]))
            # Sonar TRUTH is the env's own nominal-mount reading, so the 0.10 m forward mount
            # offset stays UNMODELLED by the filter (whose model puts the beam at the body
            # origin). That is a deliberate, realistic model error of the same magnitude as
            # the bias DR, and it shows up in the reported r error.
            sonar_true = float(geometry.sonar_wall_distance(
                pos[:, :2], yaw, env._sonar_mount_nom, env._sonar_yaw_nom, cfg.tank_radius)[0])
            sonar_m = sonar_true + float(rng.normal(0, scfg.sonar_noise)) + sensor_bias["sonar"]
            v_meas = (env._robot.data.root_lin_vel_b[0, :2].cpu().numpy()
                      + rng.normal(0, scfg.dvl_noise, 2) + sensor_bias["dvl"])
            gyro_m = (float(env._robot.data.root_ang_vel_b[0, 2])
                      + float(rng.normal(0, gyro_noise_std)) + gyro_bias)
            z_meas = float(pos[0, 2]) + float(rng.normal(0, scfg.depth_noise)) + sensor_bias["depth"]

            ukfm = None
            if bool(ukfm_in_range(pos[:, 2], scfg)[0]):
                ukfm = (r_true + float(rng.normal(0, scfg.ukfm_noise)),
                        phi_true + float(rng.normal(0, scfg.ukfm_noise)))
            ekf.step(v_bx=float(v_meas[0]), v_by=float(v_meas[1]), gyro_z=gyro_m, dt=dt,
                     sonar=sonar_m, ukfm=ukfm)

            est_err["r"].append(ekf.r - r_true)
            est_err["phi"].append(float(mref._wrap_to_pi(torch.tensor(ekf.phi - phi_true))))
            est_err["s"].append(ekf.s - float(s_gt[0]))

            # theta_hat comes straight from the arc-length state (s = R * dtheta), so no second
            # integrator can drift away from it.
            th_hat = theta_hat[0] + ekf.s / cfg.tank_radius
            roll_m, pitch_m, _ = euler_xyz_from_quat(quat)
            # Attitude BIAS was missing here until 2026-07-31, which made every tilt figure
            # optimistic: the MPC regulates the measured roll/pitch, so a constant offset is a
            # floor on the real tilt it can hold.
            roll_m = roll_m[0] + float(rng.normal(0, att_noise(scfg))) + sensor_bias["att"][0]
            pitch_m = pitch_m[0] + float(rng.normal(0, att_noise(scfg))) + sensor_bias["att"][1]
            yaw_hat = th_hat + ekf.phi
            pos_est = torch.tensor([[ekf.r * math.cos(float(th_hat)),
                                     ekf.r * math.sin(float(th_hat)), z_meas]], device=dev)
            quat_est = quat_from_euler_xyz(roll_m.reshape(1), pitch_m.reshape(1),
                                           yaw_hat.reshape(1).to(dev))
            v_b_est = env._robot.data.root_lin_vel_b.clone()
            v_b_est[0, :2] = torch.as_tensor(v_meas, dtype=torch.float32, device=dev)
            w_b_est = env._robot.data.root_ang_vel_b.clone()
            w_b_est[0, 2] = gyro_m
            # Everything the controller sees from here on is the ESTIMATE, including the
            # phase timing inputs -- otherwise the comparison would be half-cheating.
            pos_c, quat_c, v_c, w_c = pos_est, quat_est, v_b_est, w_b_est
            z_for_sm, s_for_sm = pos_est[:, 2], torch.tensor([ekf.s], device=dev)
            theta_a_src, s_a_src = th_hat.reshape(1), torch.tensor([ekf.s], device=dev)
        else:
            pos_c, quat_c, v_c, w_c = pos, quat, v_b, w_b
            z_for_sm, s_for_sm = pos[:, 2], s_gt
            theta_a_src, s_a_src = theta, s_gt

        # Advance the reference one control step, then preview it over the horizon.
        z_ref, s_ref, _phase_sc, advanced = ssm.step(state, z_for_sm, s_for_sm, scan_cfg,
                                                     z_latch=z_for_sm)
        cycles += (advanced & (state.phase == 0)).long()  # wrap SWAY_B(3) -> DESCEND(0)
        ref = mref.reference_preview(
            state.phase, state.z_ramp, state.s_ramp, state.s_ref, state.z_hold,
            mpc_cfg, args_cli.horizon,
        )

        if policy is not None:
            with torch.no_grad():
                e_now = mref.wallscan_errors(
                    torch.cat([pos_c, quat_c, v_c, w_c], dim=-1),
                    z_ref=ref["z_ref"][:, 0], s_ref=ref["s_ref"][:, 0],
                    v_tan_des=ref["v_tan_des"][:, 0], v_z_des=ref["v_z_des"][:, 0],
                    theta_anchor=theta, s_anchor=s_gt, cfg=mpc_cfg,
                )[0]
                phf = state.phase.float()[0]
                feat = torch.cat([e_now.cpu(), torch.stack([
                    torch.sin(2 * math.pi * phf / 4), torch.cos(2 * math.pi * phf / 4)]).cpu()])
                weights = policy(feat).numpy()

        x0 = torch.cat([pos_c, quat_c, v_c, w_c], dim=-1).cpu().numpy()
        for e in range(N):
            P = mpc.param_matrix(
                {k: v[e] for k, v in ref.items()},
                theta_anchor=float(theta[e]), s_anchor=float(s_gt[e]),
            )
            t0 = time.perf_counter()
            out = mpc.solve(x0[e], P, weights, want_sensitivity=False, init_state_traj=first_solve)
            t_solve += time.perf_counter() - t0
            n_solve += 1
            if out["status"] != 0:
                n_fail += 1
            if np.abs(out["u0_cmd"]).max() > 0.98:
                sat_steps += 1
            action[e] = torch.as_tensor(out["u0_cmd"], dtype=torch.float32, device=dev)
        first_solve = False

        _obs, _rew, terminated, truncated, _info = env.step(action)
        dones = terminated | truncated

        pos_n = env._robot.data.root_pos_w - env.scene.env_origins
        quat_n = env._robot.data.root_quat_w
        _, _, yaw_n = euler_xyz_from_quat(quat_n)
        up_z = _body_up(quat_n)[:, 2].clamp(-1.0, 1.0)
        wall_dist = geometry.sonar_wall_distance(
            pos_n[:, :2], yaw_n, env._sonar_mount_nom, env._sonar_yaw_nom, cfg.tank_radius
        )
        log.record(
            phase=state.phase, cycles=cycles, searching=torch.zeros(N, dtype=torch.bool, device=dev),
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
            if args_cli.state == "ekf":
                reset_estimator()
            first_solve = True

        if args_cli.log_every and i % args_cli.log_every == 0:
            clr = float(cfg.tank_radius - torch.linalg.norm(pos_n[0, :2]))
            crab = float(torch.rad2deg(mref._wrap_to_pi(yaw_n[0] - torch.atan2(pos_n[0, 1], pos_n[0, 0]))))
            print(f"  t={i * dt:6.1f}s ph={int(state.phase[0])} z={float(pos_n[0, 2]):5.2f}"
                  f"(ref {float(z_ref[0]):5.2f}) s={float(s_gt[0]):+6.2f}(ref {float(s_ref[0]):+6.2f})"
                  f" clr={clr:5.2f} crab={crab:+6.2f}deg tilt={float(torch.rad2deg(torch.arccos(up_z[0]))):5.2f}deg"
                  f" |u|max={float(action[0].abs().max()):.2f}")

    traj = log.as_arrays(step_dt=dt)
    sway_step = cfg.scan.ref_step_s if cfg.scan.ref_step_s > 0.0 else cfg.scan.ref_step
    metrics = em.compute_metrics(
        traj, step_dt=dt, d_ref=cfg.d_ref,
        heave_target=cfg.scan.ref_step / dt if cfg.scan.ref_step > 0.0 else None,
        sway_target=sway_step / dt if sway_step > 0.0 else None,
        episode=None if args_cli.score_episode < 0 else args_cli.score_episode,
        episode_length_s=cfg.episode_length_s,
        settle_s=args_cli.settle_s,
    )
    metrics["controller"] = "pure-NMPC"
    metrics["tam"] = args_cli.tam
    metrics["hydro"] = args_cli.hydro
    metrics["thruster_tau"] = args_cli.thruster_tau
    metrics["policy_ckpt"] = args_cli.policy_ckpt
    metrics["state_source"] = args_cli.state
    metrics["seed"] = args_cli.seed
    if args_cli.state == "ekf":
        harvest_counts()
        e = {k: np.asarray(v) for k, v in est_err.items()}
        metrics["estimator"] = {
            "sensors": args_cli.sensors,
            "ukfm_gate": args_cli.ukfm_gate, "gyro_bias": gyro_bias,
            "r_rmse_m": float(np.sqrt(np.mean(e["r"] ** 2))),
            "r_bias_m": float(np.mean(e["r"])),
            "phi_rmse_deg": float(np.degrees(np.sqrt(np.mean(e["phi"] ** 2)))),
            "s_rmse_m": float(np.sqrt(np.mean(e["s"] ** 2))),
            "sonar_gated": int(est_counts["gated"]), "ukfm_fixes": int(est_counts["ukfm"]),
        }
        print(f"\n[EKF] r RMSE {metrics['estimator']['r_rmse_m'] * 100:.1f} cm "
              f"(bias {metrics['estimator']['r_bias_m'] * 100:+.1f} cm)  "
              f"phi RMSE {metrics['estimator']['phi_rmse_deg']:.2f} deg  "
              f"s RMSE {metrics['estimator']['s_rmse_m'] * 100:.1f} cm  "
              f"ukfm {metrics['estimator']['ukfm_fixes']}  gated {metrics['estimator']['sonar_gated']}")
    metrics["mpc"] = {
        "horizon": args_cli.horizon, "dt_mpc": args_cli.dt_mpc, "rti_iters": args_cli.rti_iters,
        "werr": werr.tolist(), "wu": DEFAULT_WU,
        "solve_ms_mean": 1e3 * t_solve / max(1, n_solve),
        "solve_fail_frac": n_fail / max(1, n_solve),
        "saturated_step_frac": sat_steps / max(1, n_solve),
        "state_source": args_cli.state,
        "spin_search": False,
    }

    out_dir = os.path.abspath(args_cli.out_dir or os.path.join(REPO_ROOT, "results"))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"trajectory_{args_cli.tag}.npz"), **traj)
    with open(os.path.join(out_dir, f"metrics_{args_cli.tag}.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(em.format_metrics(metrics))
    print(f"\n[MPC] solve {metrics['mpc']['solve_ms_mean']:.2f} ms/step  "
          f"fail {100 * metrics['mpc']['solve_fail_frac']:.2f}%  "
          f"saturated {100 * metrics['mpc']['saturated_step_frac']:.2f}%")
    print(f"[INFO] -> {out_dir}/trajectory_{args_cli.tag}.npz, metrics_{args_cli.tag}.json")

    if not args_cli.no_plot:
        png = em.plot_trajectory(
            traj, os.path.join(out_dir, f"trajectory_{args_cli.tag}.png"),
            episode=None if args_cli.score_episode < 0 else args_cli.score_episode,
            title=f"pure-NMPC ({args_cli.tam} TAM)  [{args_cli.tag}]",
        )
        print(f"[INFO] plot -> {png}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
