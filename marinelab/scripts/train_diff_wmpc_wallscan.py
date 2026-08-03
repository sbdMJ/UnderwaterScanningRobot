# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Phase 4: Diff-WMPC weight learning on the wallscan NMPC.

A small policy maps the current error/phase situation to the MPC's diagonal cost weights;
acados reports ``d z*/d p_global``; a differentiable task loss is backpropagated through that
Jacobian. No RL, no critic, no reward — see ``marinelab.algorithms.diff_wmpc``.

## Why short randomized segments instead of episodes

The reference implementation trains for ~40k steps. A wallscan episode is 9000 steps (180 s),
so 40k steps would be **4.4 episodes** — nowhere near enough coverage of the four phases,
depths and headings. But the Diff-WMPC gradient is local (one loss evaluation at one shooting
node), so what matters is the VARIETY of situations, not episode continuity. This script
therefore resets to a random (radius, bearing, depth, attitude, phase) every
``--segment_steps`` control steps, turning the same step budget into hundreds of independent
conditions.

## What we already know it should discover

A hand sweep on 2026-07-30 found the roll weight has to reach ~2000 before the solver spends
heave differential on cancelling the sway leg's parasitic moment (measured closed-loop: sway
tilt 1.11 deg at w_roll=20 -> 0.14 deg at w_roll=2000, and under the SHIPPED TAM the same
change does almost nothing because pitch has no other actuator). The policy starts from the
hand-tuned ``DEFAULT_WERR``, so watch ``w_roll`` in the log: if Diff-WMPC is working it
should climb on its own, per phase, without anybody telling it that heave differential
trims roll.

Run (inside the container, needs the acados mount):

    ./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/train_diff_wmpc_wallscan.py \\
        --steps 20000 --tam fixed --ckpt_dir ../checkpoints/diff_wmpc'
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diff-WMPC weight learning for the wallscan NMPC.")
parser.add_argument("--steps", type=int, default=20000, help="Total control steps.")
parser.add_argument("--segment_steps", type=int, default=600,
                    help="Steps per randomized segment (12 s). Coverage beats continuity here.")
parser.add_argument("--tam", choices=["shipped", "fixed"], default="fixed")
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument("--dt_mpc", type=float, default=0.05)
parser.add_argument("--rti_iters", type=int, default=8)
parser.add_argument("--sens_nodes", type=str, default=None,
                    help="Comma-separated shooting nodes to sum the loss over (default: N/6, N/3, "
                         "2N/3, N). Summing is what stops the learner from optimizing only the "
                         "fast axes; a single node 1.0 s out made it abandon radial tracking.")
parser.add_argument("--w_radial_floor", type=float, default=10.0,
                    help="Per-entry lower bound on the radial and heading cost weights. The "
                         "measured collapse drove w_radial to 2-3; this is cheap insurance that "
                         "costs nothing while it is not binding. 0 disables.")
parser.add_argument("--l_u", type=float, default=None,
                    help="Task-loss weight on the NORMALIZED control effort. This is the only term "
                         "that opposes larger cost weights, and at the default 1e-3 it lost: a 40k "
                         "run parked w_roll at 4998/5000 and 6.5%% of steps were dropped as "
                         "saturated, which turns 'weights-varying' MPC into a constant "
                         "max-weight controller. Raising it is the direct counter-pressure.")
parser.add_argument("--l_v_z", type=float, default=None,
                    help="Task-loss weight on the heave-rate error. Default comes from "
                         "WallScanLossCfg (2.0); pass 0.2 to reproduce the pre-2026-07-31 value.")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--grad_clip", type=float, default=0.1)
parser.add_argument("--history_len", type=int, default=4)
parser.add_argument("--ckpt", type=str, default=None, help="Evaluate this checkpoint (learning off).")
parser.add_argument("--ckpt_dir", type=str, default=None)
parser.add_argument("--ckpt_every", type=int, default=5000)
parser.add_argument("--log_every", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401

import marinelab  # noqa: F401

from marinelab.algorithms.diff_wmpc import DiffWMPCLearner, WallScanLossCfg, WeightPolicy
from marinelab.assets.pkrc import PKRCThrusterCfg, PKRCThrusterCfgFixedTAM
from marinelab.tasks.pkrc_wallscan import mpc_reference as mref
from marinelab.tasks.pkrc_wallscan import scan_state_machine as ssm
from marinelab.tasks.pkrc_wallscan.mpc_controller import DEFAULT_WERR, DEFAULT_WU, PlantParams, WallScanMPC
from marinelab.tasks.pkrc_wallscan.mpc_reference import NE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROLL_IDX = 8  # index of `roll` in mpc_reference.ERROR_NAMES


def _atomic_save(state, path):
    """Temp file + rename: an interrupted write must never leave a 0-byte checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    rng = np.random.default_rng(args_cli.seed)

    cfg_env = __import__("marinelab.tasks.pkrc_wallscan.wallscan_env_cfg", fromlist=["x"]).WallScanStage3Cfg()
    cfg_env.scene.num_envs = 1
    cfg_env.thrusters = PKRCThrusterCfgFixedTAM() if args_cli.tam == "fixed" else PKRCThrusterCfg()

    env = gym.make("Isaac-PKRC-WallScan-Stage3-Direct-v0", cfg=cfg_env).unwrapped
    dev = env.device
    dt = env.step_dt
    ALL = torch.arange(1, device=dev)

    prm = PlantParams.from_env(env)
    mpc_cfg = mref.WallScanMPCCfg(
        tank_radius=cfg_env.tank_radius, d_ref=cfg_env.d_ref,
        z_top=cfg_env.scan.z_top, z_bottom=cfg_env.scan.z_bottom, sway_step=cfg_env.scan.sway_step,
        ref_step=cfg_env.scan.ref_step, ref_step_s=cfg_env.scan.ref_step_s,
        step_dt=dt, dt_mpc=args_cli.dt_mpc,
    )
    nodes = ([int(v) for v in args_cli.sens_nodes.split(",")]
             if args_cli.sens_nodes else None)
    mpc = WallScanMPC(prm, mpc_cfg, N=args_cli.horizon, rti_iters=args_cli.rti_iters,
                      sens_nodes=nodes, with_sensitivity=True,
                      code_export_root=os.path.join(REPO_ROOT, "isaaclab", "logs",
                                                    "c_generated_code_wallscan_diff"))

    # feature = the NE error entries + phase(sin, cos). Phase matters: the useful weight
    # schedule is phase-dependent (a sway leg wants roll authority, a heave leg wants z),
    # and without it the policy would have to emit one compromise for all four.
    feat_dim = NE + 2
    werr_lb = np.full(NE, 0.1)
    if args_cli.w_radial_floor > 0.0:
        werr_lb[[0, 6, 7]] = args_cli.w_radial_floor  # radial, head_x, head_y
    policy = WeightPolicy(feat_dim, NE, mpc.nu, history_len=args_cli.history_len,
                          werr_init=np.asarray(DEFAULT_WERR, float),
                          wu_init=np.full(mpc.nu, DEFAULT_WU), werr_lb=werr_lb)
    print(f"[diff-wmpc] loss nodes {mpc.sens_nodes} "
          f"(= {[round(k * args_cli.dt_mpc, 2) for k in mpc.sens_nodes]} s ahead)  "
          f"radial/heading floor {args_cli.w_radial_floor}")
    learner = DiffWMPCLearner(policy, n_pglobal=mpc.n_pglobal, lr=args_cli.lr,
                              batch_size=args_cli.batch_size, grad_clip=args_cli.grad_clip)
    loss_cfg = WallScanLossCfg(max_thrust=prm.max_thrust)
    if args_cli.l_v_z is not None:
        loss_cfg.l_v_z = args_cli.l_v_z
    if args_cli.l_u is not None:
        loss_cfg.l_u = args_cli.l_u
    print(f"[diff-wmpc] l_v_z = {loss_cfg.l_v_z}  l_u = {loss_cfg.l_u}")

    evaluating = args_cli.ckpt is not None
    if evaluating:
        state = torch.load(args_cli.ckpt, map_location="cpu")
        learner.load_state_dict(state if "policy" in state else {"policy": state})
        policy.eval()
        print(f"[diff-wmpc] EVAL mode from {args_cli.ckpt}; learning OFF")
    ckpt_dir = args_cli.ckpt_dir or os.path.join(REPO_ROOT, "checkpoints", "diff_wmpc")
    os.makedirs(ckpt_dir, exist_ok=True)

    state_m = ssm.ScanState(1, device=dev)
    s_gt = torch.zeros(1, device=dev)
    theta_prev = torch.zeros(1, device=dev)

    def new_segment():
        """Random (radius, bearing, depth, attitude, phase) so a step budget buys coverage."""
        env.reset()
        env.episode_length_buf[:] = 0
        r = float(rng.uniform(3.5, 5.3))          # radial error in [-1.0, +0.8] m
        th = float(rng.uniform(0.0, 2 * math.pi))
        z = float(rng.uniform(cfg_env.scan.z_bottom + 0.2, cfg_env.scan.z_top - 0.2))
        # Yaw within the envelope where the solve stays interior: a huge initial heading
        # error just produces saturated steps, which the learner discards anyway.
        yaw = th + float(rng.uniform(-0.7, 0.7))
        roll, pitch = (float(rng.uniform(-0.09, 0.09)) for _ in range(2))
        phase = int(rng.integers(0, 4))

        root = env._robot.data.default_root_state.clone()
        root[:, 0:3] = env.scene.env_origins + torch.tensor([r * math.cos(th), r * math.sin(th), z],
                                                           device=dev)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        root[:, 3] = cr * cp * cy + sr * sp * sy
        root[:, 4] = sr * cp * cy - cr * sp * sy
        root[:, 5] = cr * sp * cy + sr * cp * sy
        root[:, 6] = cr * cp * sy - sr * sp * cy
        root[:, 7:13] = 0.0
        env._robot.write_root_pose_to_sim(root[:, :7], ALL)
        env._robot.write_root_velocity_to_sim(root[:, 7:], ALL)
        env._thruster.reset(ALL)

        state_m.phase[:] = phase
        state_m._hold[:] = 0
        state_m.sway_dir[:] = 1.0
        state_m.z_ramp[:] = z
        state_m.z_hold[:] = z
        s_gt[:] = 0.0
        state_m.s_ramp[:] = 0.0
        state_m.s_ref[:] = cfg_env.scan.sway_step if phase in (1, 3) else 0.0
        theta_prev[:] = th
        policy.reset_history()

    new_segment()
    first_solve = True
    loss_ema = None
    w_roll_ema = float(DEFAULT_WERR[ROLL_IDX])
    ceil_ema = 0.0

    print(f"[diff-wmpc] TAM={args_cli.tam} steps={args_cli.steps} segment={args_cli.segment_steps} "
          f"feat_dim={feat_dim} n_pglobal={mpc.n_pglobal}")
    for i in range(args_cli.steps):
        if not simulation_app.is_running():
            print(f"[WARN] app closed early at {i}/{args_cli.steps}")
            break

        pos = env._robot.data.root_pos_w - env.scene.env_origins
        quat = env._robot.data.root_quat_w
        x0_t = torch.cat([pos, quat, env._robot.data.root_lin_vel_b,
                          env._robot.data.root_ang_vel_b], dim=-1)
        theta = torch.atan2(pos[:, 1], pos[:, 0])
        s_gt += mref._wrap_to_pi(theta - theta_prev) * cfg_env.tank_radius
        theta_prev = theta

        ssm.step(state_m, pos[:, 2], s_gt, cfg_env.scan, z_latch=pos[:, 2])
        ref = mref.reference_preview(state_m.phase, state_m.z_ramp, state_m.s_ramp,
                                     state_m.s_ref, state_m.z_hold, mpc_cfg, args_cli.horizon)
        theta_a = theta.detach().clone()
        s_a = s_gt.detach().clone()

        def errors_at(x, stage):
            return mref.wallscan_errors(
                x.reshape(1, -1),
                z_ref=ref["z_ref"][:, stage], s_ref=ref["s_ref"][:, stage],
                v_tan_des=ref["v_tan_des"][:, stage], v_z_des=ref["v_z_des"][:, stage],
                theta_anchor=theta_a, s_anchor=s_a, cfg=mpc_cfg,
            )[0]

        with torch.no_grad():
            e_now = errors_at(x0_t[0], 0)
            ph = state_m.phase.float()
            feat = torch.cat([e_now.cpu(),
                              torch.stack([torch.sin(2 * math.pi * ph[0] / 4),
                                           torch.cos(2 * math.pi * ph[0] / 4)]).cpu()])

        w = learner.compute_weights(feat)
        P = mpc.param_matrix({k: v[0] for k, v in ref.items()},
                             theta_anchor=float(theta_a), s_anchor=float(s_a))
        out = mpc.solve(x0_t[0].cpu().numpy(), P, w.detach().cpu().numpy(),
                        want_sensitivity=not evaluating, init_state_traj=first_solve)
        first_solve = False

        if not evaluating:
            # Each node is scored against the reference THAT node tracks; reusing the stage-0
            # setpoint everywhere would reintroduce the myopia the multi-node sum removes.
            L = learner.learn_step(
                w, out, lambda x, node: errors_at(x.to(dev), node).cpu(), loss_cfg
            )
            if L is not None:
                loss_ema = L if loss_ema is None else 0.99 * loss_ema + 0.01 * L
        w_roll_ema = 0.99 * w_roll_ema + 0.01 * float(w[ROLL_IDX])
        # Fraction of cost weights pinned at their ceiling. This is the pathology itself: if it
        # goes to 1.0 the policy has stopped varying the weights and the method has degenerated
        # into a constant max-weight controller, whatever the loss curve says.
        with torch.no_grad():
            at_ceiling = float((w[:NE] > 0.99 * policy.werr_ub).float().mean())
        ceil_ema = 0.99 * ceil_ema + 0.01 * at_ceiling

        action = torch.as_tensor(out["u0_cmd"], dtype=torch.float32, device=dev).unsqueeze(0)
        env.step(action)

        if (i + 1) % args_cli.segment_steps == 0:
            new_segment()
            first_solve = True

        if args_cli.log_every and i % args_cli.log_every == 0:
            print(f"  t={i:6d} ph={int(state_m.phase[0])} loss={L if not evaluating else None} "
                  f"ema={loss_ema if loss_ema is None else round(loss_ema, 5)} "
                  f"upd={learner.n_updates} skip(st/sat/nan)="
                  f"{learner.n_skipped_status}/{learner.n_skipped_sat}/{learner.n_skipped_nan} "
                  f"w[rad,z,head,roll]=[{float(w[0]):.0f},{float(w[1]):.0f},"
                  f"{float(w[6]):.0f},{float(w[ROLL_IDX]):.0f}] w_roll_ema={w_roll_ema:.0f} "
                  f"at_ceiling={ceil_ema:.2f}")

        if not evaluating and args_cli.ckpt_every and i > 0 and i % args_cli.ckpt_every == 0:
            _atomic_save(learner.state_dict(), os.path.join(ckpt_dir, f"policy_{i}.pt"))

    if not evaluating:
        _atomic_save(learner.state_dict(), os.path.join(ckpt_dir, "policy_final.pt"))
        print(f"[diff-wmpc] done. updates={learner.n_updates} skipped={learner.n_skipped} "
              f"loss_ema={loss_ema} -> {ckpt_dir}/policy_final.pt")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
