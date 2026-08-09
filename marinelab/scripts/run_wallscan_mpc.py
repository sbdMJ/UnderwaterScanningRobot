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
parser.add_argument("--task", choices=["stage3", "eval"], default="stage3",
                    help="'stage3' is the clean tank the NMPC was tuned in (no DR). 'eval' is the "
                         "stress condition the RL policy's published stress-DR numbers come from: "
                         "spawn attitude +-45 deg, added mass / damping x0.5-1.5, volume x0.85-1.15 "
                         "(i.e. buoyancy +-34 N), thrust and thruster tau x0.7-1.3, CoB/CoG +-5 cm, "
                         "inertia x0.8-1.2. The MPC model stays NOMINAL, so all of that is model "
                         "mismatch it never sees -- which is the point of the test. Pair it with "
                         "--sensors placeholder, whose bias magnitudes match this task's sensor DR.")
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
parser.add_argument("--full_3axis", action="store_true", default=False,
                    help="Model the DVL and gyro as the 3-AXIS instruments they are. By default\n"
                         "estimator_loop passes body v_z and the roll/pitch rates through from\n"
                         "GROUND TRUTH, which flatters the wallscan's primary axis and, because\n"
                         "wrench_observer computes its force channel from v_z, feeds any\n"
                         "plant-conditioned policy a partly ground-truth signal. Turning this on is\n"
                         "the observer-degradation test AND the sim2real fix: it is what the real\n"
                         "vehicle actually has. Off by default so published numbers keep meaning.")
parser.add_argument("--gyro_bias", type=float, default=None,
                    help="Override the gyro bias [rad/s]. Default: taken from the chosen --sensors "
                         "model (0.02 placeholder, 9.4e-5 datasheet).")
parser.add_argument("--policy_ckpt", type=str, default=None,
                    help="Diff-WMPC weight-policy checkpoint. When given, the MPC cost weights come "
                         "from the policy every step instead of the fixed DEFAULT_WERR, so the same "
                         "metrics path compares learned vs hand-tuned weights directly.")
parser.add_argument("--w_roll", type=float, default=None,
                    help="Override the roll/pitch cost weights. Under the FIXED TAM a high value is\n                         what makes the solver spend heave differential on cancelling the sway\n                         leg's parasitic moment; under the shipped TAM it cannot help (pitch has\n                         no other actuator), so this is the decisive A/B.")
parser.add_argument("--werr", type=str, action="append", default=None, metavar="NAME=VALUE",
                    help="Override ONE entry of the error cost weight vector, by its\n"
                         "mpc_reference.ERROR_NAMES name (e.g. --werr pitch=0.1 --werr w_y=0.1).\n"
                         "Repeatable, and applied AFTER --w_roll so it can split that flag's pair.\n"
                         "Exists because --w_roll sets roll and pitch TOGETHER, and under\n"
                         "PKRCThrusterCfgFixedTAM the pitch row of the allocation matrix is all\n"
                         "zeros: weight on pitch/w_y can only perturb the QP and spend heave\n"
                         "authority on an axis with no actuator behind it.")
parser.add_argument("--wu", type=float, default=None,
                    help="Override the control-effort weight (default DEFAULT_WU = 0.01, applied to\n"
                         "all 6 inputs). Only the werr:wu RATIO matters, so this is the knob for how\n"
                         "aggressive the solver is allowed to be -- the Diff-WMPC policy learned\n"
                         "wu ~ 0.005 alongside werr ~ 100x DEFAULT_WERR, i.e. an effort penalty ~200x\n"
                         "weaker than the hand-tuned default, and that ratio is not reachable with\n"
                         "--werr alone because werr is capped at 5000.")
parser.add_argument("--log_weights", action="store_true", default=False,
                    help="Store the PER-STEP, PER-ENV cost weights the policy emits and the plant\n"
                         "features it saw, into the trajectory npz as werr_t/wu_t/plant_t. The\n"
                         "aggregate werr_emitted_mean cannot show a transient, and a run that\n"
                         "diverges in one env of eight is exactly a transient -- diagnosing it from\n"
                         "the mean is guesswork. Adds ~7 MB per 9000-step 8-env run.")
parser.add_argument("--snap_ramp", action="store_true", default=False,
                    help="Snap the arc reference onto the vehicle when a VERTICAL phase begins, so\n"
                         "a sway leg that cleared short does not leave the ramp commanding sideways\n"
                         "motion through the descent. See ScanCfg.snap_ramp_on_vertical: the bleed is\n"
                         "55-56 cm at every trim level and is the whole coverage floor that survives\n"
                         "a perfectly trimmed vehicle. The alternative -- tightening --reach_eps --\n"
                         "removes it by making the vehicle chase the remainder, and costs the scan:\n"
                         "measured 0.6 -> 0.1 takes cycles 1.83 -> 0.08.")
parser.add_argument("--reach_eps", type=float, default=None,
                    help="Override ScanCfg.reach_eps, the band a phase must be inside before the\n"
                         "state machine advances. The 0.6 default exists for RL BOOTSTRAP -- see\n"
                         "wallscan_env_cfg: at 0.3 a fresh policy never reaches a waypoint, sigma\n"
                         "collapses, and training deadlocks. None of that applies to the NMPC: no\n"
                         "exploration, no sigma, no bootstrap.\n"
                         "What 0.6 costs here was already diagnosed on 2026-07-26 and is still\n"
                         "present, measured 2026-08-08: a sway leg may clear up to 0.6 m short, the\n"
                         "s ramp keeps running into the NEXT phase, and the reference itself then\n"
                         "moves 55-56 cm sideways during a vertical leg -- identical across trim\n"
                         "levels, and the whole coverage floor that survives a perfectly trimmed\n"
                         "vehicle. The scan is drawing diagonals because it was asked to.")
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
parser.add_argument("--dobs", action="store_true", default=False,
                    help="Enable the residual-wrench observer (online buoyancy/trim estimation) and\n"
                         "feed its estimate to the MPC as a model parameter. OFF by default so every\n"
                         "already-published number keeps its meaning; the measurements this exists to\n"
                         "attack are the stress-DR standing offsets (tilt 13.83 deg, wall 5.52 cm,\n"
                         "saturated 29%, QP failures 0.00%) -- i.e. a model error, not a solver one.")
parser.add_argument("--dobs_channels", choices=["all", "moment", "z_moment", "none"], default="all",
                    help="Which observer channels are fed to the MPC: 'moment' = body moments only,\n"
                         "'z_moment' adds the vertical (buoyancy) force, 'all' adds the lateral forces.\n"
                         "MEASURED indistinguishable at 3 seeds -- every effect comes from the moment\n"
                         "channels, the force half changes nothing. Kept because it is the knob that\n"
                         "established that. See WrenchObserverCfg.channel_mask for the table.")
parser.add_argument("--dobs_lam_force", type=float, default=1.0,
                    help="Observer lowpass cutoff [rad/s] on the force channel. Must stay well below\n"
                         "the DVL's hold rate (15 Hz -> 3 control steps) so the zero-order-hold stair\n"
                         "averages out while a constant disturbance passes.")
parser.add_argument("--dobs_lam_moment", type=float, default=2.0,
                    help="Observer lowpass cutoff [rad/s] on the moment channel; faster than the force\n"
                         "channel because the gyro is not zero-order held.")
parser.add_argument("--cam", choices=["overview", "follow", "wall"], default="overview",
                    help="Viewport placement for GUI/video runs. 'overview' is a fixed shot of the "
                         "whole tank, which is what shows the boustrophedon pattern; 'follow' rides "
                         "with the vehicle; 'wall' looks along the wall from just behind it, the view "
                         "in which the standoff and the crab angle are actually legible.")
parser.add_argument("--render", action="store_true", default=False,
                    help="Open the Isaac Sim window. Needs a real DISPLAY (local session, not ssh) "
                         "and ./docker/run.sh --gui. Off by default so every scoring command in the "
                         "docs keeps running headless without having to say so.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record the viewport to results/videos/<tag>/. Renders offscreen, so unlike "
                         "--render it works headless and therefore over ssh.")
parser.add_argument("--video_length", type=int, default=3000,
                    help="Recorded steps (3000 = 60 s at the 50 Hz control rate). The run itself "
                         "continues for --steps; only the recording stops.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Headless unless explicitly asked to draw: this script's documented invocations are scoring runs
# that predate the viewing flags and never pass --headless.
args_cli.headless = not args_cli.render
if args_cli.video:
    args_cli.enable_cameras = True  # offscreen render pipeline; independent of the window

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
from marinelab.algorithms.diff_wmpc import build_features
from marinelab.tasks.pkrc_wallscan import eval_metrics as em
from marinelab.tasks.pkrc_wallscan import geometry, mpc_reference as mref, scan_state_machine as ssm
from marinelab.tasks.pkrc_wallscan.mpc_controller import DEFAULT_WERR, DEFAULT_WU, PlantParams, WallScanMPC
from marinelab.tasks.pkrc_wallscan.sensors import (
    SensorCfg, SensorCfgDatasheet, _body_up, dvl_hold_steps,
)
from marinelab.tasks.pkrc_wallscan.estimator_loop import WallFrameEstimator
from marinelab.tasks.pkrc_wallscan.wrench_observer import WrenchObserver, WrenchObserverCfg
from marinelab.tasks.pkrc_wallscan.wallscan_env_cfg import WallScanEvalCfg, WallScanStage3Cfg


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


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """Euler angle -> (-pi, pi]. ``euler_xyz_from_quat`` returns [0, 2pi), where a small negative
    roll reads as ~6.28 rad and any |.|-mean over it is meaningless."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def main() -> None:
    TASKS = {"stage3": ("Isaac-PKRC-WallScan-Stage3-Direct-v0", WallScanStage3Cfg),
             "eval": ("Isaac-PKRC-WallScan-Eval-Direct-v0", WallScanEvalCfg)}
    gym_id, cfg_cls = TASKS[args_cli.task]
    cfg = cfg_cls()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed  # DirectRLEnv seeds torch from this in __init__
    cfg.thrusters = PKRCThrusterCfgFixedTAM() if args_cli.tam == "fixed" else PKRCThrusterCfg()
    if args_cli.snap_ramp:
        cfg.scan.snap_ramp_on_vertical = True
        print('[INFO] arc ramp snaps to the vehicle on vertical-phase entry')
    if args_cli.reach_eps is not None:
        cfg.scan.reach_eps = args_cli.reach_eps
        print(f'[INFO] reach_eps = {cfg.scan.reach_eps} (default 0.6 is an RL-bootstrap value)')
    if args_cli.dr_trim_cm is not None:
        _t = args_cli.dr_trim_cm / 100.0
        for _f in ('cob_offset_x', 'cob_offset_y', 'cob_offset_z',
                   'cog_offset_x', 'cog_offset_y', 'cog_offset_z'):
            setattr(cfg.randomization, _f, (-_t, _t))
        print(f'[INFO] CoB/CoG trim DR narrowed to +-{args_cli.dr_trim_cm} cm '
              f'(predicted max standing pitch {__import__("math").degrees(__import__("math").asin(min(_t / 0.150, 1.0))):.1f} deg)')
    if args_cli.pin_dr or args_cli.pin_spec:
        apply_pin_dr(cfg, args_cli.pin_dr or 'B', args_cli.pin_spec)
    cfg.hydrodynamics = (PKRCHydrodynamicsCfgZSlender() if args_cli.hydro == "z_slender"
                         else PKRCHydrodynamicsCfg())
    if args_cli.thruster_tau is not None:
        if args_cli.thruster_tau < cfg.sim.dt * cfg.decimation:
            raise SystemExit(f"--thruster_tau must be >= the control step "
                             f"{cfg.sim.dt * cfg.decimation}; alpha = dt/tau > 1 is unstable")
        cfg.thrusters.time_constant_up = args_cli.thruster_tau
        cfg.thrusters.time_constant_down = args_cli.thruster_tau

    # Viewport. Only consulted when a window is open or a video is being recorded; the tank is
    # R=6 m / H=10 m and the vehicle rides at r=4.4 m, so the default (7.5, 7.5, 7.5) eye sits
    # inside the wall and shows almost nothing.
    if args_cli.cam == "follow":
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (-3.0, 0.0, 1.5)   # behind and above, in the tank-local frame
        cfg.viewer.lookat = (2.0, 0.0, -0.5)
    elif args_cli.cam == "wall":
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (-1.5, -4.0, 0.5)  # off to the side: standoff and crab angle both visible
        cfg.viewer.lookat = (0.5, 0.0, 0.0)
    else:
        cfg.viewer.origin_type = "env"
        cfg.viewer.eye = (0.0, -14.0, 6.0)  # outside the wall, high enough to see the full 10 m sweep
        cfg.viewer.lookat = (0.0, 2.0, 5.0)

    env_outer = gym.make(gym_id, cfg=cfg,
                         render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = os.path.join(args_cli.out_dir or os.path.join(REPO_ROOT, "results"),
                                 "videos", args_cli.tag)
        print(f"[INFO] recording {args_cli.video_length} steps to {video_dir}")
        env_outer = gym.wrappers.RecordVideo(
            env_outer, video_folder=video_dir, step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length, disable_logger=True)
    env = env_outer.unwrapped
    dev, N = env.device, env.num_envs
    dt = env.step_dt

    prm = PlantParams.from_env(env)
    mpc_cfg = mref.WallScanMPCCfg(
        tank_radius=cfg.tank_radius, d_ref=cfg.d_ref,
        z_top=cfg.scan.z_top, z_bottom=cfg.scan.z_bottom, sway_step=cfg.scan.sway_step,
        ref_step=cfg.scan.ref_step, ref_step_s=cfg.scan.ref_step_s,
        step_dt=dt, dt_mpc=args_cli.dt_mpc,
        # d_ref is the SONAR standoff (confirmed 2026-08-03), so the radial target has to
        # account for the beam sitting forward of the body origin.
        sonar_mount_x=cfg.sensors.sonar_mount_pos[0],
    )
    print(f"[INFO] TAM={args_cli.tam}  r_des={mpc_cfg.r_des:.3f} m "
          f"(d_ref={mpc_cfg.d_ref} as SONAR range, mount +{mpc_cfg.sonar_mount_x} m)  "
          f"horizon={args_cli.horizon}x{args_cli.dt_mpc}s={args_cli.horizon * args_cli.dt_mpc:.2f}s")
    print(f"[INFO] plant: m={prm.mass} I={prm.rigid_body_inertia} B={prm.buoyancy_force:.2f} N "
          f"max_thrust={prm.max_thrust} N")

    # ONE SOLVER PER ENV. An acados RTI solver warm-starts from the state it was left in by the
    # previous solve() call. With one env that is the previous timestep -- exactly the intended
    # warm start. With N envs solved in a loop it is a DIFFERENT VEHICLE's trajectory, 20 m away
    # (env_spacing) at an unrelated bearing, and 8 RTI iterations do not recover from that.
    # Measured 2026-08-03, 8 envs x 9000 steps, shared solver vs per-env solver:
    #   crab 23.41 deg -> see run log;  wall error 116 cm;  saturated 13.6% (1 env: 0.24 deg,
    #   0.47 cm, 0.79%). The single-env numbers were never wrong, but they hid this completely.
    # Instances 1..N-1 reuse the generated code (generate/build only on the first), so this costs
    # one codegen and N cheap solver objects.
    export_root = os.path.join(REPO_ROOT, "isaaclab", "logs", "c_generated_code_wallscan")
    mpcs = [WallScanMPC(
        prm, mpc_cfg, N=args_cli.horizon, rti_iters=args_cli.rti_iters,
        with_sensitivity=False,  # the baseline needs no gradients; Phase 4 turns this on
        code_export_root=export_root, generate=(e == 0), build=(e == 0),
    ) for e in range(N)]
    mpc = mpcs[0]  # for .nu / .param_matrix, which are stateless
    werr = np.asarray(DEFAULT_WERR, float)
    if args_cli.w_roll is not None:
        werr[8] = werr[9] = args_cli.w_roll
    for spec in args_cli.werr or ():
        name, _, val = spec.partition("=")
        if name not in mref.ERROR_NAMES or not _:
            raise SystemExit(f"--werr expects NAME=VALUE with NAME in {mref.ERROR_NAMES}; got {spec!r}")
        werr[mref.ERROR_NAMES.index(name)] = float(val)
    wu_val = DEFAULT_WU if args_cli.wu is None else args_cli.wu
    weights = np.concatenate([werr, np.full(mpc.nu, wu_val)])
    print("[INFO] werr = " + ", ".join(f"{n}={v:g}" for n, v in zip(mref.ERROR_NAMES, werr))
          + f"  wu={wu_val:g}")

    policy = None
    feat_mode = "error_phase"
    if args_cli.policy_ckpt:
        from marinelab.algorithms.diff_wmpc import WeightPolicy
        from marinelab.algorithms.diff_wmpc import feature_dim as dw_feature_dim
        from marinelab.tasks.pkrc_wallscan.mpc_reference import NE as _NE

        st = torch.load(args_cli.policy_ckpt, map_location="cpu")
        # The layout travels WITH the checkpoint; checkpoints written before --feat existed
        # carry no stamp and are legacy `error_phase` by construction.
        feat_mode = st.get("feat_mode", "error_phase") if isinstance(st, dict) else "error_phase"
        plant_feat = bool(st.get("plant_feat", False)) if isinstance(st, dict) else False
        sym_sway = bool(st.get("sym_sway", False)) if isinstance(st, dict) else False
        fixed_wu = st.get("fixed_wu") if isinstance(st, dict) else None
        ck_horizon = st.get("horizon") if isinstance(st, dict) else None
        if ck_horizon is not None and int(ck_horizon) != args_cli.horizon:
            raise SystemExit(
                f"checkpoint was trained with --horizon {ck_horizon} but this run uses "
                f"{args_cli.horizon}; the preview stages would sample different look-ahead times")
        # ONE POLICY INSTANCE PER ENV. WeightPolicy keeps an internal `_history` of past
        # features, so a single instance driven round-robin over N envs interleaves eight
        # vehicles' histories into one buffer -- the same bug class as sharing one acados
        # solver. Invisible with the legacy near-constant weights, fatal with a preview feature.
        sd = st["policy"] if "policy" in st else st
        policies = []
        for _ in range(N):
            p = WeightPolicy(dw_feature_dim(feat_mode, _NE, plant=plant_feat), _NE, mpc.nu,
                             werr_init=werr, wu_init=np.full(mpc.nu, DEFAULT_WU))
            p.load_state_dict(sd)
            p.eval()
            policies.append(p)
        policy = policies[0]
        print(f"[INFO] weights from Diff-WMPC policy {args_cli.policy_ckpt} "
              f"(feat={feat_mode}, plant={plant_feat}, sym_sway={sym_sway}, "
              f"fixed_wu={fixed_wu}, {N} instances)")
        if plant_feat and not args_cli.dobs:
            raise SystemExit(
                "this checkpoint was trained with --plant_feat, so its policy input includes the "
                "wrench-observer channels; run with --dobs so they exist. Without it the plant "
                "block would be all zeros and the policy would silently see a vehicle it is not on. "
                "Use --dobs --dobs_channels none to match training, which gives the estimate to the "
                "POLICY only and still feeds the solver a zero disturbance.")

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
    # The sensor synthesis and the filter live in estimator_loop so the Diff-WMPC trainer drives
    # the IDENTICAL model. That is not tidiness: the learner tunes weights for the state quality
    # it trains against, so any divergence here reappears as a train/test mismatch. This file
    # previously had its own copy and it silently omitted the INS attitude bias, which made
    # every tilt figure it reported optimistic.
    scfg_cls = SensorCfgDatasheet if args_cli.sensors == "datasheet" else SensorCfg
    scfg = scfg_cls(ukfm_gate=args_cli.ukfm_gate, ukfm_surface_z=cfg.tank_height)
    if args_cli.sensors == "placeholder":
        scfg.sonar_bias_dr, scfg.depth_bias_dr = 0.10, 0.10
        scfg.dvl_bias_dr, scfg.ins_att_bias_dr = 0.01, 0.04
    gyro_bias = (args_cli.gyro_bias if args_cli.gyro_bias is not None
                 else scfg_cls.ins_gyro_bias_dr if args_cli.sensors == "datasheet" else 0.02)
    # ONE estimator per env: WallFrameEstimator holds the filter state, the DVL hold and the
    # per-episode sensor biases for a single vehicle, and returns (1, .) tensors. Sharing one
    # across envs silently estimated env 0 and fed it to all 8 (it raised on the tensor cat, but
    # a shape-compatible version of the same bug would have scored nonsense).
    #
    # Each gets its own noise stream, spawned off the run seed so seeds stay reproducible and no
    # two envs draw the identical realization.
    ests: list[WallFrameEstimator] = []
    if args_cli.state == "ekf":
        seeds = np.random.SeedSequence(args_cli.seed).spawn(N)
        ests = [WallFrameEstimator(
            full_3axis=args_cli.full_3axis,
            scfg=scfg, tank_radius=cfg.tank_radius, step_dt=dt,
            sonar_mount_nom=env._sonar_mount_nom[e:e + 1] if env._sonar_mount_nom.shape[0] > 1
                            else env._sonar_mount_nom,
            sonar_yaw_nom=env._sonar_yaw_nom,
            gyro_bias=gyro_bias, rng=np.random.default_rng(ss),
        ) for e, ss in enumerate(seeds)]
        print(f"[INFO] sensors={args_cli.sensors} gyro_bias={gyro_bias:.3e} "
              f"sonar_noise={scfg.sonar_noise} att_bias={scfg.ins_att_bias_dr:.4f} "
              f"beam={math.degrees(scfg.sonar_beam_half_angle):.1f}deg "
              f"dvl_hold={dvl_hold_steps(scfg, dt)} n_est={len(ests)}")

    def reset_estimator(mask: torch.Tensor | None = None):
        """Fresh filter + fresh per-episode biases, for the envs in ``mask`` (all if None).

        Episodes end independently once N > 1, so resetting every estimator on any done would
        discard mid-mission filter state for envs that never reset.
        """
        pos0 = env._robot.data.root_pos_w - env.scene.env_origins
        quat0 = env._robot.data.root_quat_w
        _, _, yaw0 = euler_xyz_from_quat(quat0)
        idx = range(N) if mask is None else torch.nonzero(mask).flatten().tolist()
        for e in idx:
            ests[e].reset(pos0[e:e + 1], quat0[e:e + 1], float(yaw0[e]))
            # The env re-draws the physical sonar mount on every reset (nominal under stage3, and
            # +-8 cm / +-0.04 rad under eval). Hand the estimator the TRUE mount so the synthesized
            # range carries that error, while its filter keeps predicting from the surveyed
            # nominal -- an unmodeled forward mount offset pushes the estimated radius OUT, which
            # is the +5.3 cm closed-loop bias measured before the offset was modelled at all.
            ests[e].sonar_mount_true = env._sonar_mount[e:e + 1]
            ests[e].sonar_yaw_true = float(env._sonar_yaw[e])

    env_outer.reset()  # through the wrapper so RecordVideo arms its step_trigger
    env.episode_length_buf[:] = 0  # own the whole window; see compute_metrics' episode-0 note
    reset_internal()
    if args_cli.state == "ekf":
        reset_estimator()

    log = em.TrajectoryLog()
    rp_log: list[torch.Tensor] = []   # per-step (2, N) roll/pitch [rad]; merged into the npz below
    est_err = {"r": [], "phi": [], "s": []}
    # Accumulated across segments: reset_estimator() builds a FRESH filter, so reading
    # ekf.n_ukfm at the end would report only what happened after the last reset (the timeout
    # fires on the final step, which made this read 0-1 instead of thousands).
    est_counts = {"ukfm": 0, "gated": 0}

    def harvest_counts():
        """Totals across segments. reset_estimator() builds a FRESH filter, so reading the live
        one at the end reported only what happened after the last reset -- the time-out fires on
        the final step, which once made this read 0-1 fixes instead of ~8000."""
        if ests:
            tot = [e.harvest() for e in ests]
            est_counts["ukfm"] = sum(u for u, _ in tot)
            est_counts["gated"] = sum(g for _, g in tot)

    # --- residual-wrench observer (only used with --dobs) --------------------
    # ONE PER ENV, for the same reason as the solver and the estimator: it holds one vehicle's
    # filter state, and under DR each vehicle carries a DIFFERENT disturbance (buoyancy, CoB and
    # thrust coefficient are re-drawn per env per episode). Sharing one would average 8 vehicles.
    obs_list: list[WrenchObserver] = []
    if args_cli.dobs:
        mask = {
            "all": (True,) * 6,
            "z_moment": (False, False, True, True, True, True),
            "moment": (False, False, False, True, True, True),
            # 'none' RUNS the observer but exports nothing to the solver. Needed by --plant_feat
            # checkpoints: the trainer feeds the solver a zero disturbance and gives the estimate
            # only to the POLICY, so exporting it here as well would be a train/test mismatch --
            # and it would confound "weights conditioned on the plant" with "model corrected by
            # the plant", which is a different experiment that was already measured (not a win).
            "none": (False,) * 6,
        }[args_cli.dobs_channels]
        ocfg = WrenchObserverCfg(lam_force=args_cli.dobs_lam_force,
                                 lam_moment=args_cli.dobs_lam_moment, channel_mask=mask)
        obs_list = [WrenchObserver(prm, ocfg) for _ in range(N)]
        print(f"[INFO] wrench observer ON: channels={args_cli.dobs_channels} "
              f"lam_force={ocfg.lam_force} lam_moment={ocfg.lam_moment} "
              f"bounds=+-{ocfg.max_force} N / +-{ocfg.max_moment} N*m")
    # (N, ND) world force + body moment handed to the solver as a stage parameter. Stays all-zero
    # without --dobs, which reproduces the pre-observer controller exactly.
    d_wrench = np.zeros((N, mref.ND))
    # Thrust APPLIED over the interval that ends at the velocity sample the observer differences.
    # Must be the CLIPPED command: feeding the desired one makes the observer credit undeliverable
    # thrust to the environment and wind up, and under DR 29% of steps saturate.
    u_prev_newton = np.zeros((N, mpc.nu))
    sat_prev = np.zeros(N, dtype=bool)
    d_sum = np.zeros(mref.ND)
    d_absmax = np.zeros(mref.ND)
    n_dobs = 0

    action = torch.zeros(N, 6, device=dev)
    # Per-env, not global: a cold start is only needed by the env that actually reset, and
    # forcing it on all N throws away every other env's valid warm start.
    cold = np.ones(N, dtype=bool)
    t_solve = 0.0
    n_solve = 0
    n_fail = 0
    sat_steps = 0
    # Mean of the weights a policy actually EMITS in closed loop. Two uses: it is the closed-loop
    # version of the paper's Fig. 5 (the offline probe only shows what the net does on synthetic
    # inputs), and it is the only defensible vector to freeze for a static control -- freezing at
    # one hand-picked operating point misrepresents a policy whose output swings 90% across phases.
    w_sum = np.zeros(mpc.n_pglobal)
    n_w = 0
    # Per-env saturated-step EMA, the 5th plant channel. Per env for the same reason the solvers
    # and estimators are: the envs are different vehicles running independent episodes.
    sat_ema = np.zeros(N)
    w_log: list[np.ndarray] = []      # per-step (N, n_pglobal), only with --log_weights
    plant_log: list[np.ndarray] = []  # per-step (N, N_PLANT)

    def _plant_row(e: int) -> np.ndarray:
        """The PLANT feature block for env `e`, scaled EXACTLY as train_diff_wmpc_wallscan.py
        scales it -- moments by 10 N*m, the buoyancy residual by 34 N (Eval's volume DR against a
        40 N thruster), saturation already in [0, 1]. Reads the observer's RAW estimate, not
        `exported()`: --dobs_channels masks what reaches the SOLVER, and applying it here too
        would feed the policy zeros for channels it was trained to see."""
        d = obs_list[e].d
        return np.array([d[3] / 10.0, d[4] / 10.0, d[5] / 10.0, d[2] / 34.0, sat_ema[e]],
                        dtype=np.float32)

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

        if ests:
            roll_g, pitch_g, _ = euler_xyz_from_quat(quat)
            outs = [ests[e].step(i, pos[e:e + 1], quat[e:e + 1], v_b[e:e + 1], w_b[e:e + 1],
                                 float(roll_g[e]), float(pitch_g[e]), float(yaw[e]),
                                 float(s_gt[e])) for e in range(N)]
            # Error stats pool every env: the RMSEs reported are over the whole run, not env 0.
            est_err["r"].extend(o.err_r for o in outs)
            est_err["phi"].extend(o.err_phi for o in outs)
            est_err["s"].extend(o.err_s for o in outs)
            # Everything the controller sees is the estimate, phase timing included.
            pos_c = torch.cat([o.pos for o in outs])
            quat_c = torch.cat([o.quat for o in outs])
            v_c = torch.cat([o.v_b for o in outs])
            w_c = torch.cat([o.w_b for o in outs])
            theta_a_src = torch.cat([o.theta_anchor for o in outs])
            s_a_src = torch.cat([o.s_anchor for o in outs])
            z_for_sm, s_for_sm = pos_c[:, 2], s_a_src
        else:
            pos_c, quat_c, v_c, w_c = pos, quat, v_b, w_b
            z_for_sm, s_for_sm = pos[:, 2], s_gt
            theta_a_src, s_a_src = theta, s_gt

        # Residual wrench, from THIS step's velocity sample and the thrust applied since the
        # previous one. Placed here, before the solve, so the estimate the solver uses was formed
        # from the same state the solver is handed -- and fed the CONTROLLER's velocities
        # (estimates under --state ekf), not ground truth, so nothing downstream sees truth.
        if obs_list:
            v_np = v_c.detach().cpu().numpy()
            w_np = w_c.detach().cpu().numpy()
            q_np = quat_c.detach().cpu().numpy()
            for e in range(N):
                d_wrench[e] = obs_list[e].update(q_np[e], v_np[e], w_np[e], u_prev_newton[e], dt,
                                                 saturated=bool(sat_prev[e]))
            # Diagnostics track the RAW estimate, not the masked export: a channel that is being
            # withheld is exactly the one whose magnitude you want to see in the log.
            d_raw = np.abs(np.stack([o.d for o in obs_list]))
            d_sum += d_raw.mean(axis=0)
            d_absmax = np.maximum(d_absmax, d_raw.max(axis=0))
            n_dobs += 1

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
                e_all = mref.wallscan_errors(
                    torch.cat([pos_c, quat_c, v_c, w_c], dim=-1),
                    z_ref=ref["z_ref"][:, 0], s_ref=ref["s_ref"][:, 0],
                    v_tan_des=ref["v_tan_des"][:, 0], v_z_des=ref["v_z_des"][:, 0],
                    theta_anchor=theta, s_anchor=s_gt, cfg=mpc_cfg,
                )
                weights_e = np.stack([
                    policies[e](build_features(
                        feat_mode, e_now=e_all[e], phase=state.phase[e], ref=ref,
                        n_stages=args_cli.horizon, env=e, sym_sway=sym_sway,
                        plant=_plant_row(e) if plant_feat else None)).numpy()
                    for e in range(N)])
                if fixed_wu is not None:
                    weights_e[:, _NE:] = fixed_wu
                w_sum += weights_e.mean(axis=0)
                n_w += 1
                if args_cli.log_weights:
                    w_log.append(weights_e.astype(np.float32).copy())
                    plant_log.append(np.stack([_plant_row(e) for e in range(N)])
                                     if plant_feat else np.zeros((N, 5), dtype=np.float32))

        x0 = torch.cat([pos_c, quat_c, v_c, w_c], dim=-1).cpu().numpy()
        for e in range(N):
            P = mpc.param_matrix(
                {k: v[e] for k, v in ref.items()},
                theta_anchor=float(theta[e]), s_anchor=float(s_gt[e]),
                d_wrench=d_wrench[e],
            )
            t0 = time.perf_counter()
            out = mpcs[e].solve(x0[e], P, weights if policy is None else weights_e[e],
                                want_sensitivity=False, init_state_traj=bool(cold[e]))
            t_solve += time.perf_counter() - t0
            n_solve += 1
            if out["status"] != 0:
                n_fail += 1
            sat_prev[e] = np.abs(out["u0_cmd"]).max() > 0.98
            sat_ema[e] = 0.99 * sat_ema[e] + 0.01 * float(sat_prev[e])
            if sat_prev[e]:
                sat_steps += 1
            action[e] = torch.as_tensor(out["u0_cmd"], dtype=torch.float32, device=dev)
        cold[:] = False
        # The env clamps the action to [-1, 1] and scales by the thrust coefficient, so this is the
        # thrust that actually reaches the thrusters (modulo the first-order lag, which the MPC
        # model does not carry either -- see the wrench_observer docstring).
        if obs_list:
            u_prev_newton = np.clip(action.detach().cpu().numpy(), -1.0, 1.0) * prm.max_thrust

        # Through the wrapper: RecordVideo captures a frame on each step it sees, so stepping the
        # unwrapped env would silently produce an empty video.
        _obs, _rew, terminated, truncated, _info = env_outer.step(action)
        dones = terminated | truncated

        pos_n = env._robot.data.root_pos_w - env.scene.env_origins
        quat_n = env._robot.data.root_quat_w
        roll_n, pitch_n, yaw_n = euler_xyz_from_quat(quat_n)
        up_z = _body_up(quat_n)[:, 2].clamp(-1.0, 1.0)
        # Roll and pitch SEPARATELY, not just the combined tilt: under the fixed TAM the pitch row
        # of the allocation matrix is all zeros, so an attitude result is uninterpretable until the
        # actuated axis (roll) is split from the unactuated one (pitch). Kept out of
        # ``TrajectoryLog.FIELDS`` on purpose -- that tuple is a contract play.py must also fill.
        rp_log.append(torch.stack([_wrap_pi(roll_n), _wrap_pi(pitch_n)]).detach().cpu())
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
            if policy is not None:
                # Same rule as the estimator: the feature history describes the episode that just
                # ended, so carrying it across a reset feeds the new spawn the old vehicle's past.
                for e in torch.nonzero(dones).flatten().tolist():
                    policies[e].reset_history()
            if ests:
                reset_estimator(dones)  # only the envs that actually reset
            if obs_list:
                # Only the envs that reset: the disturbance is re-drawn per episode, so keeping the
                # old estimate would trim the new vehicle for the previous one -- but the envs that
                # did NOT reset still carry a valid, converged estimate worth ~1 s of settling.
                for e in torch.nonzero(dones).flatten().tolist():
                    obs_list[e].reset()
                    u_prev_newton[e] = 0.0
                    sat_prev[e] = False
            cold |= dones.cpu().numpy()

        if args_cli.log_every and i % args_cli.log_every == 0:
            clr = float(cfg.tank_radius - torch.linalg.norm(pos_n[0, :2]))
            crab = float(torch.rad2deg(mref._wrap_to_pi(yaw_n[0] - torch.atan2(pos_n[0, 1], pos_n[0, 0]))))
            print(f"  t={i * dt:6.1f}s ph={int(state.phase[0])} z={float(pos_n[0, 2]):5.2f}"
                  f"(ref {float(z_ref[0]):5.2f}) s={float(s_gt[0]):+6.2f}(ref {float(s_ref[0]):+6.2f})"
                  f" clr={clr:5.2f} crab={crab:+6.2f}deg tilt={float(torch.rad2deg(torch.arccos(up_z[0]))):5.2f}deg"
                  f" |u|max={float(action[0].abs().max()):.2f}")

    traj = log.as_arrays(step_dt=dt)
    if w_log:
        traj["werr_t"] = np.stack(w_log)[:, :, :len(werr)]     # (T, N, NE)
        traj["wu_t"] = np.stack(w_log)[:, :, len(werr):]       # (T, N, nu)
        traj["plant_t"] = np.stack(plant_log)                  # (T, N, N_PLANT)
    if rp_log:
        rp = torch.stack(rp_log).numpy()          # (T, 2, N)
        traj["roll_deg"] = np.rad2deg(rp[:, 0, :])
        traj["pitch_deg"] = np.rad2deg(rp[:, 1, :])
    sway_step = cfg.scan.ref_step_s if cfg.scan.ref_step_s > 0.0 else cfg.scan.ref_step
    metrics = em.compute_metrics(
        traj, step_dt=dt, d_ref=cfg.d_ref,
        heave_target=cfg.scan.ref_step / dt if cfg.scan.ref_step > 0.0 else None,
        sway_target=sway_step / dt if sway_step > 0.0 else None,
        # NOTE the local `sway_step` above is the per-STEP ramp rate; `cfg.scan.sway_step` is the
        # arc length one sway leg is meant to advance. The coverage metric needs the latter.
        sway_step=cfg.scan.sway_step,
        episode=None if args_cli.score_episode < 0 else args_cli.score_episode,
        episode_length_s=cfg.episode_length_s,
        settle_s=args_cli.settle_s,
    )
    metrics["controller"] = "pure-NMPC"
    metrics["task"] = gym_id  # provenance: stage3 and eval are not comparable conditions
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
        # `werr`/`wu` are the FIXED vector; with --policy_ckpt they are only the init and the
        # solver never sees them, so `werr_emitted_mean` is the one to read for a policy run.
        "werr": werr.tolist(), "wu": wu_val,
        "werr_emitted_mean": (w_sum[:len(werr)] / max(n_w, 1)).tolist() if n_w else None,
        "wu_emitted_mean": (w_sum[len(werr):] / max(n_w, 1)).tolist() if n_w else None,
        "solve_ms_mean": 1e3 * t_solve / max(1, n_solve),
        "solve_fail_frac": n_fail / max(1, n_solve),
        "saturated_step_frac": sat_steps / max(1, n_solve),
        "state_source": args_cli.state,
        "spin_search": False,
    }
    if obs_list:
        # Provenance AND diagnosis. |d_fz| is the estimate that should track the DR'd buoyancy
        # error (Eval draws +-34 N), and |d_mx|/|d_my| the CoB/CoG trim moment (+-11.4 N*m at a
        # 5 cm offset). If a run fails with these pinned at the bounds, the observer is winding up,
        # not estimating -- check that the APPLIED (clipped) command is what reaches it.
        metrics["dobs"] = {
            "channels": args_cli.dobs_channels,
            "channel_mask": [bool(b) for b in obs_list[0].cfg.channel_mask],
            "lam_force": args_cli.dobs_lam_force, "lam_moment": args_cli.dobs_lam_moment,
            "abs_mean": (d_sum / max(1, n_dobs)).tolist(),
            "abs_max": d_absmax.tolist(),
            "updates": int(sum(o.n_update for o in obs_list)),
            "clipped": int(sum(o.n_clipped for o in obs_list)),
        }
        am = metrics["dobs"]["abs_mean"]
        print(f"\n[DOBS] |d_f| mean ({am[0]:.1f}, {am[1]:.1f}, {am[2]:.1f}) N  "
              f"|d_m| mean ({am[3]:.2f}, {am[4]:.2f}, {am[5]:.2f}) N*m  "
              f"clipped {metrics['dobs']['clipped']}/{metrics['dobs']['updates']}")

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

    env_outer.close()  # flushes the video file; env.close() would leave it truncated


if __name__ == "__main__":
    main()
    simulation_app.close()
