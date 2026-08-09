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
parser.add_argument("--task", choices=["stage3", "eval"], default="stage3",
                    help="Training condition. 'stage3' is the clean tank. 'eval' turns on the\n"
                         "stress DR the policy is evaluated under (added mass / damping x0.5-1.5,\n"
                         "volume x0.85-1.15 = buoyancy +-34 N, thrust and tau x0.7-1.3, CoB/CoG\n"
                         "+-5 cm, inertia x0.8-1.2, spawn attitude +-45 deg), removing the\n"
                         "train/test distribution gap. Note what this can and cannot fix: under\n"
                         "DR the plant saturates 39% of steps (measured 2026-08-04), and cost\n"
                         "weights choose what to prioritize, not how hard the thrusters can push."),
parser.add_argument("--steps", type=int, default=20000, help="Total control steps.")
parser.add_argument("--segment_steps", type=int, default=600,
                    help="Steps per randomized segment (12 s). Coverage beats continuity here.")
parser.add_argument("--tam", choices=["shipped", "fixed"], default="fixed")
parser.add_argument("--hydro", choices=["shipped", "z_slender"], default="shipped",
                    help="Hydrodynamic coefficients; keep this matched to the evaluation runs.")
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
parser.add_argument("--feat", choices=["error_phase", "preview", "both"], default="error_phase",
                    help="Policy input. 'error_phase' (default, legacy) = current 12-D error +\n"
                         "current phase: purely REACTIVE, and dw_ekf was measured to collapse to a\n"
                         "constant weight vector under it. 'preview' is the source paper's design --\n"
                         "a 5-point look-ahead of (v_z_des, v_tan_des) over the horizon and nothing\n"
                         "else -- which is what makes the weights ANTICIPATE a phase change instead of\n"
                         "reacting to it. 'both' concatenates the two, to ablate whether dropping the\n"
                         "error term costs anything. Default stays legacy so published checkpoints keep\n"
                         "their meaning; see diff_wmpc.FEAT_MODES.")
parser.add_argument("--spawn_yaw_err", type=float, default=0.7,
                    help="Half-width of the training spawn's heading error [rad]. The evaluation\n"
                         "env spawns up to pi, so the 0.7 default leaves a plant-conditioned policy\n"
                         "extrapolating from its first step -- measured as the regime both DR\n"
                         "divergences began in. Use 3.15 to match evaluation. A fixed vector is\n"
                         "indifferent to this: it has no input to be out of distribution on.")
parser.add_argument("--snap_ramp", action="store_true", default=False,
                    help="Snap the arc reference onto the vehicle when a VERTICAL phase begins.\n"
                         "Without it the s ramp keeps slewing toward a sway target the leg cleared\n"
                         "short of, so the reference commands ~55 cm of sideways motion through the\n"
                         "descent -- measured 2026-08-08 as a third of the coverage error, and it is\n"
                         "in the REFERENCE, so a policy trained without this learns to serve a\n"
                         "diagonal. Keep it matched between training and evaluation.")
parser.add_argument("--dr_trim_cm", type=float, default=None, metavar="CM",
                    help="Narrow the CoB/CoG offset DR to +-CM centimetres (all three axes),\n"
                         "leaving every other channel alone. The Eval default is +-5 cm, which\n"
                         "the closed form theta_eq = asin(trim / z_cb) with the measured\n"
                         "z_cb = 0.150 m turns into a 19.5 deg standing pitch -- on an axis with no\n"
                         "actuator, so no controller can answer it, and the tangential drift that\n"
                         "wrecks scan coverage follows it at ~16 cm per degree (r = 0.92 over 10\n"
                         "configurations). Trim is a build property: it is measured by floating the\n"
                         "vehicle and reading its rest angle, and corrected with ballast. Narrowing\n"
                         "this to what assembly can actually hold removes conditions the vehicle is\n"
                         "physically incapable of, instead of training against them.")
parser.add_argument("--pin_spec", type=str, default=None, metavar="VOL,COB_X,COG_X",
                    help="Pin the DR draw to an ARBITRARY vehicle, e.g. '0.85,-0.05,0.05'. Same\n"
                         "semantics as --pin_dr (every other channel nominal) but anywhere in the\n"
                         "box, so the region where the ceiling opens can be mapped. Screening is\n"
                         "cheap because the shared vector's own error UPPER-BOUNDS the ceiling at\n"
                         "that vehicle: an optimal vector cannot do better than zero error, so a\n"
                         "cell where the shared vector is already good needs no per-vehicle training.")
parser.add_argument("--pin_dr", choices=["A", "B", "C"], default=None,
                    help="Collapse every DR range to a SINGLE vehicle, so 'the optimal weight\n"
                         "vector for this vehicle' is well defined. A = buoyancy -34 N with the\n"
                         "CoB trimmed one way, B = nominal, C = +34 N trimmed the other. Only\n"
                         "volume_scale and the CoB/CoG offsets differ between them; every other\n"
                         "channel is pinned at nominal, so the three isolate exactly the axis a\n"
                         "plant-conditioned policy would have to exploit. Spawn attitude stays\n"
                         "randomized -- pinning it would remove the transient the controller has\n"
                         "to survive. Used to measure the CEILING on conditioning: the gap between\n"
                         "a per-vehicle optimum and one shared vector is the most any policy can win.")
parser.add_argument("--static", action="store_true", default=False,
                    help="Train Diff-MPC instead of Diff-WMPC: ONE weight vector optimised\n"
                         "directly (Algorithm 1 line 4, theta_k = theta), no network, features\n"
                         "ignored. This is the honest control for 'does the schedule earn its\n"
                         "keep' -- freezing a trained policy at its emitted mean is not the same\n"
                         "thing, since a mean is not the argmin of anything. Prints a ready-to-use\n"
                         "--werr/--wu flag string at the end.")
parser.add_argument("--plant_feat", action="store_true", default=False,
                    help="Append the PLANT block to the policy input: body moments (3), the\n"
                         "buoyancy residual d_f_z, and the recent saturated fraction, from\n"
                         "wrench_observer. This is the axis on which weights-varying can beat a\n"
                         "fixed vector HERE: the reference barely varies and the solver already\n"
                         "previews it, but the plant varies per episode under DR, and one vector\n"
                         "must compromise across every drawn vehicle while a policy that can\n"
                         "identify its vehicle need not. Pairs with --task eval.")
parser.add_argument("--sym_sway", action="store_true", default=False,
                    help="Feed |v_tan| instead of v_tan in the preview, so SWAY_A and SWAY_B -- the\n"
                         "same motion mirrored, and weights multiply SQUARED errors -- cannot be\n"
                         "given different weights. dw_both_lu1 split them 0.743 vs 1.059, which is\n"
                         "overfitting. Off by default so earlier checkpoints keep their meaning.")
parser.add_argument("--fixed_wu", type=float, default=None,
                    help="Pin the control-effort weights to this constant instead of letting the\n"
                         "policy emit them. Measured 2026-08-07: dw_both_lu1's raw weights swing\n"
                         "2.8x across phases while q/r -- the only thing a quadratic cost can see --\n"
                         "swings 1.27x, so most of the network's capacity was going into a scaling\n"
                         "the solver is invariant to. Fixing wu removes that redundancy entirely.")
parser.add_argument("--l_u", type=float, default=None,
                    help="Task-loss weight on the NORMALIZED control effort. This is the only term "
                         "that opposes larger cost weights, and at the default 1e-3 it lost: a 40k "
                         "run parked w_roll at 4998/5000 and 6.5%% of steps were dropped as "
                         "saturated, which turns 'weights-varying' MPC into a constant "
                         "max-weight controller. Raising it is the direct counter-pressure.")
parser.add_argument("--l_s", type=float, default=None,
                    help="Task-loss weight on the ARC-LENGTH error, i.e. scan coverage. Default\n"
                         "(WallScanLossCfg) is 1.0 against l_radial 2.0, so coverage was the lowest\n"
                         "priority in the objective -- and it shows: measured 2026-08-08, the\n"
                         "DR-optimised vector drifts 1.64 m tangentially through each vertical leg\n"
                         "while holding wall distance to 5 cm, because nothing in the loss objected.\n"
                         "Note the gradient is also myopic here: tangential drift accumulates over\n"
                         "tens of seconds while the loss nodes sit at 0.25-1.5 s, so raising this may\n"
                         "be necessary but not sufficient.")
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
parser.add_argument("--state", choices=["gt", "ekf"], default="ekf",
                    help="State the learner trains against. Diff-WMPC tunes weights for the state "
                         "QUALITY it sees, so training on ground truth and deploying on an "
                         "estimate is a train/test mismatch -- measured as crab 0.074 deg WORSE "
                         "than hand-tuned weights. 'ekf' closes it by driving the same estimator "
                         "run_wallscan_mpc.py evaluates with.")
parser.add_argument("--sensors", choices=["placeholder", "datasheet"], default="datasheet")
parser.add_argument("--ukfm_gate", choices=["legacy_height", "depth_below_surface"],
                    default="depth_below_surface")
parser.add_argument("--full_3axis", action="store_true", default=False,
                    help="Model the DVL and gyro as the 3-AXIS instruments they are. By default\n"
                         "estimator_loop passes body v_z and the roll/pitch rates through from\n"
                         "GROUND TRUTH, which flatters the wallscan's primary axis and, because\n"
                         "wrench_observer computes its force channel from v_z, feeds any\n"
                         "plant-conditioned policy a partly ground-truth signal. Turning this on is\n"
                         "the observer-degradation test AND the sim2real fix: it is what the real\n"
                         "vehicle actually has. Off by default so published numbers keep meaning.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import gymnasium as gym

from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_tasks  # noqa: F401

import marinelab  # noqa: F401

from marinelab.algorithms.diff_wmpc import (
    DiffWMPCLearner,
    StaticWeights,
    WallScanLossCfg,
    WeightPolicy,
    build_features,
)
from marinelab.algorithms.diff_wmpc import feature_dim as dw_feature_dim
from marinelab.assets.pkrc import (
    PKRCHydrodynamicsCfg,
    PKRCHydrodynamicsCfgZSlender,
    PKRCThrusterCfg,
    PKRCThrusterCfgFixedTAM,
)
from marinelab.tasks.pkrc_wallscan import mpc_reference as mref
from marinelab.tasks.pkrc_wallscan.estimator_loop import WallFrameEstimator
from marinelab.tasks.pkrc_wallscan.wrench_observer import WrenchObserver
from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, SensorCfgDatasheet
from marinelab.tasks.pkrc_wallscan import scan_state_machine as ssm
from marinelab.tasks.pkrc_wallscan.mpc_controller import DEFAULT_WERR, DEFAULT_WU, PlantParams, WallScanMPC
from marinelab.tasks.pkrc_wallscan.mpc_reference import NE


PIN_DR = {
    "A": dict(volume_scale=(0.85, 0.85), cob_offset_x=(-0.05, -0.05), cog_offset_x=(0.05, 0.05)),
    "B": dict(volume_scale=(1.00, 1.00), cob_offset_x=(0.0, 0.0), cog_offset_x=(0.0, 0.0)),
    "C": dict(volume_scale=(1.15, 1.15), cob_offset_x=(0.05, 0.05), cog_offset_x=(-0.05, -0.05)),
}


def apply_pin_dr(cfg, name, spec=None):
    """Freeze the DR draw to one deterministic vehicle (see --pin_dr / --pin_spec)."""
    r = cfg.randomization
    for f in ("added_mass_scale", "linear_damping_scale", "quadratic_damping_scale",
              "thrust_coefficient_scale", "time_constant_scale", "inertia_scale"):
        setattr(r, f, (1.0, 1.0))
    for f in ("cob_offset_y", "cob_offset_z", "cog_offset_y", "cog_offset_z"):
        setattr(r, f, (0.0, 0.0))
    if spec:
        vol, cob, cog = (float(x) for x in spec.split(","))
        r.volume_scale = (vol, vol)
        r.cob_offset_x = (cob, cob)
        r.cog_offset_x = (cog, cog)
    else:
        for f, v in PIN_DR[name].items():
            setattr(r, f, v)
    print(f"[INFO] DR pinned to vehicle {name}: volume={r.volume_scale[0]} "
          f"cob_x={r.cob_offset_x[0]} cog_x={r.cog_offset_x[0]}; all other channels nominal")
    return cfg


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROLL_IDX = 8  # index of `roll` in mpc_reference.ERROR_NAMES


def _atomic_save(state, path):
    """Temp file + rename: an interrupted write must never leave a 0-byte checkpoint.

    Stamps the policy's input layout into the checkpoint. Without it, loading a `preview`
    policy into a runner that builds `error_phase` features is a silent behaviour change --
    the same failure the `WeightPolicy` bounds buffers were made to close, and here it would
    not even raise, because `both` and `error_phase` differ only in width.
    """
    state = dict(state)
    state["feat_mode"] = args_cli.feat
    state["horizon"] = args_cli.horizon
    state["plant_feat"] = args_cli.plant_feat
    state["sym_sway"] = args_cli.sym_sway
    state["fixed_wu"] = args_cli.fixed_wu
    state["static"] = args_cli.static
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    rng = np.random.default_rng(args_cli.seed)

    _cfgmod = __import__("marinelab.tasks.pkrc_wallscan.wallscan_env_cfg", fromlist=["x"])
    TASKS = {"stage3": ("Isaac-PKRC-WallScan-Stage3-Direct-v0", _cfgmod.WallScanStage3Cfg),
             "eval": ("Isaac-PKRC-WallScan-Eval-Direct-v0", _cfgmod.WallScanEvalCfg)}
    gym_id, cfg_cls = TASKS[args_cli.task]
    cfg_env = cfg_cls()
    cfg_env.scene.num_envs = 1
    cfg_env.thrusters = PKRCThrusterCfgFixedTAM() if args_cli.tam == "fixed" else PKRCThrusterCfg()
    if args_cli.snap_ramp:
        cfg_env.scan.snap_ramp_on_vertical = True
        print('[INFO] arc ramp snaps to the vehicle on vertical-phase entry')
    if args_cli.dr_trim_cm is not None:
        _t = args_cli.dr_trim_cm / 100.0
        for _f in ('cob_offset_x', 'cob_offset_y', 'cob_offset_z',
                   'cog_offset_x', 'cog_offset_y', 'cog_offset_z'):
            setattr(cfg_env.randomization, _f, (-_t, _t))
        print(f'[INFO] CoB/CoG trim DR narrowed to +-{args_cli.dr_trim_cm} cm '
              f'(predicted max standing pitch {__import__("math").degrees(__import__("math").asin(min(_t / 0.150, 1.0))):.1f} deg)')
    if args_cli.pin_dr or args_cli.pin_spec:
        apply_pin_dr(cfg_env, args_cli.pin_dr or 'B', args_cli.pin_spec)
    cfg_env.hydrodynamics = (PKRCHydrodynamicsCfgZSlender() if args_cli.hydro == "z_slender"
                             else PKRCHydrodynamicsCfg())

    env = gym.make(gym_id, cfg=cfg_env).unwrapped
    dev = env.device
    dt = env.step_dt
    ALL = torch.arange(1, device=dev)

    prm = PlantParams.from_env(env)
    mpc_cfg = mref.WallScanMPCCfg(
        tank_radius=cfg_env.tank_radius, d_ref=cfg_env.d_ref,
        z_top=cfg_env.scan.z_top, z_bottom=cfg_env.scan.z_bottom, sway_step=cfg_env.scan.sway_step,
        ref_step=cfg_env.scan.ref_step, ref_step_s=cfg_env.scan.ref_step_s,
        step_dt=dt, dt_mpc=args_cli.dt_mpc,
        # d_ref is the SONAR standoff (2026-08-03), so r_des is 4.40 not 4.50. The learner
        # optimises against this reference, which is why the policy has to be retrained.
        sonar_mount_x=cfg_env.sensors.sonar_mount_pos[0],
    )
    nodes = ([int(v) for v in args_cli.sens_nodes.split(",")]
             if args_cli.sens_nodes else None)
    mpc = WallScanMPC(prm, mpc_cfg, N=args_cli.horizon, rti_iters=args_cli.rti_iters,
                      sens_nodes=nodes, with_sensitivity=True,
                      code_export_root=os.path.join(REPO_ROOT, "isaaclab", "logs",
                                                    "c_generated_code_wallscan_diff"))

    # Policy input; see diff_wmpc.FEAT_MODES for why there are three and which one the source
    # paper actually uses. Legacy `error_phase` = NE errors + phase(sin, cos): reactive only,
    # and measured to collapse to a constant weight vector.
    feat_dim = dw_feature_dim(args_cli.feat, NE, plant=args_cli.plant_feat)
    obs = WrenchObserver(prm) if args_cli.plant_feat else None
    sat_ema = 0.0          # recent saturated fraction, the 5th plant channel
    plant_vec = np.zeros(5, dtype=np.float32) if args_cli.plant_feat else None
    werr_lb = np.full(NE, 0.1)
    if args_cli.w_radial_floor > 0.0:
        werr_lb[[0, 6, 7]] = args_cli.w_radial_floor  # radial, head_x, head_y
    _Weights = StaticWeights if args_cli.static else WeightPolicy
    policy = _Weights(feat_dim, NE, mpc.nu, history_len=args_cli.history_len,
                          werr_init=np.asarray(DEFAULT_WERR, float),
                          wu_init=np.full(mpc.nu, DEFAULT_WU), werr_lb=werr_lb)
    print(f"[diff-wmpc] r_des={mpc_cfg.r_des:.3f} (sonar convention)  hydro={args_cli.hydro}")
    print(f"[diff-wmpc] loss nodes {mpc.sens_nodes} "
          f"(= {[round(k * args_cli.dt_mpc, 2) for k in mpc.sens_nodes]} s ahead)  "
          f"radial/heading floor {args_cli.w_radial_floor}")
    learner = DiffWMPCLearner(policy, n_pglobal=mpc.n_pglobal, lr=args_cli.lr,
                              batch_size=args_cli.batch_size, grad_clip=args_cli.grad_clip)
    loss_cfg = WallScanLossCfg(max_thrust=prm.max_thrust)
    if args_cli.l_v_z is not None:
        loss_cfg.l_v_z = args_cli.l_v_z
    if args_cli.l_s is not None:
        loss_cfg.l_s = args_cli.l_s
    if args_cli.l_u is not None:
        loss_cfg.l_u = args_cli.l_u
    print(f"[diff-wmpc] l_v_z = {loss_cfg.l_v_z}  l_u = {loss_cfg.l_u}  l_s = {loss_cfg.l_s}")

    evaluating = args_cli.ckpt is not None
    if evaluating:
        state = torch.load(args_cli.ckpt, map_location="cpu")
        learner.load_state_dict(state if "policy" in state else {"policy": state})
        policy.eval()
        print(f"[diff-wmpc] EVAL mode from {args_cli.ckpt}; learning OFF")
    ckpt_dir = args_cli.ckpt_dir or os.path.join(REPO_ROOT, "checkpoints", "diff_wmpc")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Estimator, sharing the exact sensor model the evaluator uses (see estimator_loop).
    scfg_cls = SensorCfgDatasheet if args_cli.sensors == "datasheet" else SensorCfg
    scfg = scfg_cls(ukfm_gate=args_cli.ukfm_gate, ukfm_surface_z=cfg_env.tank_height)
    if args_cli.sensors == "placeholder":
        scfg.sonar_bias_dr, scfg.depth_bias_dr = 0.10, 0.10
        scfg.dvl_bias_dr, scfg.ins_att_bias_dr = 0.01, 0.04
    est = None
    if args_cli.state == "ekf":
        est = WallFrameEstimator(
            full_3axis=args_cli.full_3axis,
            scfg=scfg, tank_radius=cfg_env.tank_radius, step_dt=dt,
            sonar_mount_nom=env._sonar_mount_nom, sonar_yaw_nom=env._sonar_yaw_nom,
            gyro_bias=(scfg_cls.ins_gyro_bias_dr if args_cli.sensors == "datasheet" else 0.02),
            rng=np.random.default_rng(args_cli.seed),
        )
        print(f"[diff-wmpc] training on the EKF state ({args_cli.sensors} sensors, "
              f"gate {args_cli.ukfm_gate})")

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
        # Heading error. The +-0.7 rad cap was justified by "a huge initial heading error just
        # produces saturated steps, which the learner discards anyway" -- true at the old l_u,
        # where DR training skipped 39.6% of steps, and FALSE at l_u=0.1, where the skip rate is
        # 0-15%. Meanwhile the evaluation spawn hands up to 180 deg, so the cap left the policy
        # untrained on the exact regime both drpl divergences started in (t=0, sonar reading
        # across the tank off a lost heading). --spawn_yaw_err widens it to match.
        yaw = th + float(rng.uniform(-args_cli.spawn_yaw_err, args_cli.spawn_yaw_err))
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
        if obs is not None:
            # env.reset() RE-DRAWS the DR parameters, so the vehicle the observer was estimating
            # no longer exists. Carrying its estimate across the boundary trains the policy on a
            # converged reading of the PREVIOUS vehicle -- and, worse, means the policy never sees
            # the observer's cold start, which is what it gets at every evaluation episode. Both
            # measured 2026-08-07 as the cause of drpl's two divergences, which both begin at t=0.
            obs.reset()
            sat_ema = 0.0
            plant_vec[:] = 0.0
        if est is not None:
            _, _, yaw0 = euler_xyz_from_quat(env._robot.data.root_quat_w)
            est.reset(env._robot.data.root_pos_w - env.scene.env_origins,
                      env._robot.data.root_quat_w, float(yaw0[0]))
            # Under --task eval the env re-draws the physical sonar mount (+-8 cm / +-0.04 rad)
            # on every reset. Synthesize the range from the TRUE mount while the filter keeps
            # predicting from nominal, exactly as the evaluator does -- otherwise training would
            # see a mount error the evaluation does not, reintroducing a train/test gap in the
            # one channel the radial cost depends on.
            est.sonar_mount_true = env._sonar_mount[0:1]
            est.sonar_yaw_true = float(env._sonar_yaw[0])
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

        if est is not None:
            roll_g, pitch_g, yaw_g = euler_xyz_from_quat(quat)
            out_e = est.step(i, pos, quat, env._robot.data.root_lin_vel_b,
                             env._robot.data.root_ang_vel_b,
                             float(roll_g[0]), float(pitch_g[0]), float(yaw_g[0]),
                             float(s_gt[0]))
            x0_t = torch.cat([out_e.pos, out_e.quat, out_e.v_b, out_e.w_b], dim=-1)
            # Phase timing rides on the ESTIMATE too; feeding it truth would leave half the
            # mismatch in place.
            z_sm, s_sm = out_e.pos[:, 2], out_e.s_anchor
            theta_a, s_a = out_e.theta_anchor.clone(), out_e.s_anchor.clone()
        else:
            z_sm, s_sm = pos[:, 2], s_gt
            theta_a = theta.detach().clone()
            s_a = s_gt.detach().clone()

        ssm.step(state_m, z_sm, s_sm, cfg_env.scan, z_latch=z_sm)
        ref = mref.reference_preview(state_m.phase, state_m.z_ramp, state_m.s_ramp,
                                     state_m.s_ref, state_m.z_hold, mpc_cfg, args_cli.horizon)

        def errors_at(x, stage):
            return mref.wallscan_errors(
                x.reshape(1, -1),
                z_ref=ref["z_ref"][:, stage], s_ref=ref["s_ref"][:, stage],
                v_tan_des=ref["v_tan_des"][:, stage], v_z_des=ref["v_z_des"][:, stage],
                theta_anchor=theta_a, s_anchor=s_a, cfg=mpc_cfg,
            )[0]

        with torch.no_grad():
            feat = build_features(args_cli.feat, e_now=errors_at(x0_t[0], 0),
                                  phase=state_m.phase[0], ref=ref, n_stages=args_cli.horizon,
                                  sym_sway=args_cli.sym_sway, plant=plant_vec)

        w = learner.compute_weights(feat)
        if args_cli.fixed_wu is not None:
            # Replace the emitted effort weights with the constant, keeping the tensor
            # differentiable in the werr half only -- the wu half then contributes no gradient,
            # which is exactly the redundant direction we are removing.
            w = torch.cat([w[:NE], torch.full((mpc.nu,), args_cli.fixed_wu, device=w.device)])
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

        if obs is not None:
            u_n = np.clip(out["u0_cmd"], -1.0, 1.0) * prm.max_thrust
            sat_now = bool(np.abs(out["u0_cmd"]).max() > 0.98)
            d = obs.update(x0_t[0, 3:7].cpu().numpy(), x0_t[0, 7:10].cpu().numpy(),
                           x0_t[0, 10:13].cpu().numpy(), u_n, dt, saturated=sat_now)
            sat_ema = 0.99 * sat_ema + 0.01 * float(sat_now)
            # Scaled to O(1) so the plant block does not swamp the rest of the input: moments run
            # to ~10 N*m and the buoyancy residual to ~34 N (Eval's volume DR against a 40 N thruster).
            plant_vec[:3] = d[3:6] / 10.0
            plant_vec[3] = d[2] / 34.0
            plant_vec[4] = sat_ema

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
        if args_cli.static:
            # A static vector IS the deliverable, so emit it in the form the runner consumes
            # instead of leaving 18 numbers to be transcribed by hand.
            with torch.no_grad():
                w_fin = policy(torch.zeros(feat_dim)).cpu().numpy()
            flags = " ".join(f"--werr {n}={v:.1f}"
                             for n, v in zip(mref.ERROR_NAMES, w_fin[:NE]))
            # --fixed_wu replaces the emitted effort weights INSIDE the loop, so the policy's own
            # wu entries never saw a gradient and still hold their init. Printing them would hand
            # back a flag string that does not reproduce the controller that was just trained.
            wu_fin = args_cli.fixed_wu if args_cli.fixed_wu is not None else float(w_fin[NE:].mean())
            print(f"[diff-wmpc] STATIC vector:\n  {flags} --wu {wu_fin:.5f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
