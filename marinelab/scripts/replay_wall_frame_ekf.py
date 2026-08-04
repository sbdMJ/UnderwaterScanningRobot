# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Phase 2 validation: replay a logged trajectory through the wall-frame EKF.

Takes a ``results/trajectory_*.npz`` written by ``run_wallscan_mpc.py`` (or
``play.py --log_traj``), synthesizes the sensor streams from its ground truth using the very
noise/bias figures in ``sensors.py``, runs :class:`WallFrameEKF`, and reports how far the
estimate drifts. No Isaac Sim and no acados — pure numpy, so it runs natively in seconds and
can sweep parameters that a closed-loop run could only probe one at a time.

The question it exists to answer: **the wallscan spends ~39 s at a time on vertical legs,
which give the sonar no new information about (r, phi), so the filter coasts on gyro
integration. Is the gyro good enough?** ``sensors.py`` flags its noise figures as
"conservative placeholders pending real datasheets", and ``ins_gyro_bias_dr = 0.02 rad/s`` is
roughly 400x a real 3DM-GV7's ~10 deg/hr bias stability — so this sweep is really asking
which of those two numbers the project has to live with.

Usage (native, no container):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python marinelab/scripts/replay_wall_frame_ekf.py \\
        --traj results/trajectory_mpc_fixed.npz
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# Register bare parent packages so ``marinelab/__init__.py`` (which imports the bluerov envs
# and therefore isaaclab) never runs. Same trick as ``tests/conftest.py``; the EKF is pure
# numpy, so this is all it needs and the script stays runnable without Isaac Sim.
for _name, _sub in (("marinelab", ""), ("marinelab.tasks", "tasks"),
                    ("marinelab.tasks.pkrc_wallscan", "tasks/pkrc_wallscan")):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(_ROOT, "marinelab", _sub) if _sub else os.path.join(_ROOT, "marinelab")]
        sys.modules[_name] = _m

from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import (  # noqa: E402
    WallFrameEKF,
    WallFrameEKFCfg,
    gyro_drift_over_leg,
    range_information,
    sonar_range,
)

# sensors.py figures (SensorCfg + WallScanTrainCfg overrides)
SONAR_NOISE, SONAR_BIAS = 0.05, 0.10
DVL_NOISE, DVL_BIAS = 0.02, 0.01
GYRO_NOISE, GYRO_BIAS = 0.01, 0.02
UKFM_NOISE, UKFM_MAX_DEPTH = 0.03, 8.0
TANK_R = 6.0


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def load_gt(path: str, env: int = 0) -> dict:
    """Ground-truth (r, phi, s) plus the body velocity / yaw rate a DVL and gyro would see."""
    d = np.load(path)
    x, y, z = d["x"][:, env], d["y"][:, env], d["z"][:, env]
    yaw, theta = d["yaw"][:, env], d["theta"][:, env]
    r = np.hypot(x, y)
    phi = _wrap(yaw - theta)

    # The log does not carry velocities, so reconstruct what the sensors would measure.
    dt = float(d["t"][1, env] - d["t"][0, env])
    vx, vy = np.gradient(x, dt), np.gradient(y, dt)
    v_bx = vx * np.cos(yaw) + vy * np.sin(yaw)
    v_by = -vx * np.sin(yaw) + vy * np.cos(yaw)
    gyro_z = np.gradient(_unwrap(yaw), dt)
    return dict(r=r, phi=phi, s=d["s_gt"][:, env], z=z, v_bx=v_bx, v_by=v_by,
                gyro_z=gyro_z, phase=d["phase"][:, env], dt=dt, n=len(x))


def _unwrap(a):
    return np.unwrap(a)


def synth_sensors(gt: dict, rng: np.random.Generator, *, gyro_bias: float,
                  use_ukfm: bool, sonar_bias_on: bool = True,
                  ukfm_gate: str = "as_shipped") -> dict:
    """One episode's sensor realization: white noise per step + a constant per-episode bias."""
    n = gt["n"]
    b_sonar = rng.uniform(-SONAR_BIAS, SONAR_BIAS) if sonar_bias_on else 0.0
    b_dvl = rng.uniform(-DVL_BIAS, DVL_BIAS, size=2)
    b_gyro = gyro_bias
    true_range = np.array([sonar_range(gt["r"][k], gt["phi"][k], TANK_R) for k in range(n)])
    out = dict(
        sonar=true_range + rng.normal(0, SONAR_NOISE, n) + b_sonar,
        v_bx=gt["v_bx"] + rng.normal(0, DVL_NOISE, n) + b_dvl[0],
        v_by=gt["v_by"] + rng.normal(0, DVL_NOISE, n) + b_dvl[1],
        gyro_z=gt["gyro_z"] + rng.normal(0, GYRO_NOISE, n) + b_gyro,
        true_range=true_range,
    )
    if use_ukfm:
        # sensors.py:91 gates on ``|z| < ukfm_valid_max_depth`` and wallscan's z is HEIGHT
        # above the floor (spawn 9.5, z_bottom 1.0), so as shipped the fix is valid in the
        # lower 8 m and invalid in the top 2 m -- the opposite of what the field's comment
        # ("tank is 10 m deep, marker view lost before that") describes. Which reading is
        # intended decides whether most of the scan has absolute position or none of it.
        if ukfm_gate == "as_shipped":
            valid = np.abs(gt["z"]) < UKFM_MAX_DEPTH              # lower 8 m (INVERTED)
        elif ukfm_gate == "corrected":
            # Physically correct: marker at the surface, camera looking up, so validity is
            # bounded by DEPTH BELOW THE SURFACE, not by height. Confirmed by the user and by
            # wallscan_env.py:702 ("marker view also valid there" at the spawn depth).
            valid = (10.0 - gt["z"]) < UKFM_MAX_DEPTH            # z > 2.0
        elif ukfm_gate == "corrected_with_near_limit":
            # wallscan_env.py:485 also notes the camera degrades within ~1 m of the surface,
            # which is why the operating ceiling is surface - 1 m. Model both ends.
            valid = ((10.0 - gt["z"]) < UKFM_MAX_DEPTH) & (gt["z"] < 9.0)
        else:
            valid = gt["z"] > (10.0 - 2.0)                        # pessimistic: top 2 m only
        out["ukfm_valid"] = valid
        out["ukfm_r"] = gt["r"] + rng.normal(0, UKFM_NOISE, n)
        out["ukfm_phi"] = _wrap(gt["phi"] + rng.normal(0, UKFM_NOISE, n))
    else:
        out["ukfm_valid"] = np.zeros(n, bool)
    return out


def run_filter(gt: dict, sens: dict, *, q_phi: float = 0.02) -> dict:
    ekf = WallFrameEKF(WallFrameEKFCfg(
        tank_radius=TANK_R, r_sonar=SONAR_NOISE, q_phi=q_phi,
        initial=(float(gt["r"][0]), float(gt["phi"][0]), float(gt["s"][0])),
    ))
    est = np.zeros((gt["n"], 3))
    for k in range(gt["n"]):
        ukfm = ((float(sens["ukfm_r"][k]), float(sens["ukfm_phi"][k]))
                if sens["ukfm_valid"][k] else None)
        ekf.step(v_bx=float(sens["v_bx"][k]), v_by=float(sens["v_by"][k]),
                 gyro_z=float(sens["gyro_z"][k]), dt=gt["dt"],
                 sonar=float(sens["sonar"][k]), ukfm=ukfm)
        est[k] = (ekf.r, ekf.phi, ekf.s)
    err_r = est[:, 0] - gt["r"]
    err_phi = _wrap(est[:, 1] - gt["phi"])
    err_s = est[:, 2] - gt["s"]
    vert = np.isin(gt["phase"], (0, 2))
    return dict(
        est=est, gated=ekf.n_gated, n_sonar=ekf.n_sonar, n_ukfm=ekf.n_ukfm,
        r_rmse=float(np.sqrt(np.mean(err_r**2))), r_max=float(np.abs(err_r).max()),
        phi_rmse_deg=float(np.degrees(np.sqrt(np.mean(err_phi**2)))),
        phi_max_deg=float(np.degrees(np.abs(err_phi).max())),
        phi_vert_deg=float(np.degrees(np.sqrt(np.mean(err_phi[vert] ** 2)))) if vert.any() else float("nan"),
        s_rmse=float(np.sqrt(np.mean(err_s**2))), s_final=float(err_s[-1]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="results/trajectory_mpc_fixed.npz")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    gt = load_gt(args.traj)
    print(f"trajectory: {args.traj}  {gt['n']} steps @ {gt['dt'] * 1e3:.0f} ms "
          f"({gt['n'] * gt['dt']:.0f} s)")
    print(f"GT ranges : r {gt['r'].min():.2f}-{gt['r'].max():.2f} m, "
          f"|phi| max {math.degrees(np.abs(gt['phi']).max()):.1f} deg, "
          f"z {gt['z'].min():.2f}-{gt['z'].max():.2f} m")

    print("\n--- how much heading information does one range carry? ---")
    for deg in (0, 1, 5, 15, 30):
        print(f"  |dt/dphi| at phi={deg:2d} deg, r=4.5: "
              f"{range_information(4.5, math.radians(deg), TANK_R):.4f} m/rad")

    print("\n--- open-loop gyro drift over one unexcited 39 s vertical leg ---")
    for bias, label in ((GYRO_BIAS, "shipped placeholder"),
                        (2e-3, "10x better"),
                        (4.8e-5, "real 3DM-GV7 ~10 deg/hr")):
        print(f"  bias {bias:.1e} rad/s ({label:24s}): "
              f"{math.degrees(gyro_drift_over_leg(bias, 39.0)):7.2f} deg")

    print("\n--- can one range even SEE a small heading error? ---")
    for deg in (1, 5, 10, 20):
        from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import sonar_range as _sr
        sig = _sr(4.5, math.radians(deg), TANK_R) - (TANK_R - 4.5)
        print(f"  phi={deg:2d} deg -> range signature {1e3 * sig:7.2f} mm  "
              f"vs noise {1e3 * SONAR_NOISE:.0f} mm (SNR {sig / SONAR_NOISE:5.2f}), "
              f"per-episode bias {1e3 * SONAR_BIAS:.0f} mm (ratio {sig / SONAR_BIAS:5.3f})")

    print("\n--- EKF replay: mean over seeds ---")
    header = f"{'config':<40}{'r RMSE':>9}{'r max':>8}{'phi RMSE':>10}{'phi vert':>10}{'phi max':>9}{'s RMSE':>9}{'gated':>7}"
    print(header)
    CONFIGS = [
        ("UKF-M as shipped (lower 8 m, INVERTED)", GYRO_BIAS, True, "as_shipped"),
        ("UKF-M corrected (z > 2, depth < 8)", GYRO_BIAS, True, "corrected"),
        ("UKF-M corrected + near limit (2<z<9)", GYRO_BIAS, True, "corrected_with_near_limit"),
        ("UKF-M corrected, gyro=real GV7", 4.8e-5, True, "corrected"),
        ("UKF-M pessimistic (top 2 m only)", GYRO_BIAS, True, "surface_only"),
        ("UKF-M OFF, gyro=placeholder", GYRO_BIAS, False, "off"),
        ("UKF-M OFF, gyro=real GV7", 4.8e-5, False, "off"),
        ("UKF-M OFF, gyro=0 (noise only)", 0.0, False, "off"),
    ]
    for label, bias, ukfm, gate in CONFIGS:
        acc = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            acc.append(run_filter(gt, synth_sensors(gt, rng, gyro_bias=bias, use_ukfm=ukfm,
                                                    ukfm_gate=gate)))
        m = {k: float(np.mean([a[k] for a in acc])) for k in
             ("r_rmse", "r_max", "phi_rmse_deg", "phi_vert_deg", "phi_max_deg", "s_rmse", "gated")}
        print(f"{label:<40}{m['r_rmse']:9.3f}{m['r_max']:8.3f}{m['phi_rmse_deg']:10.2f}"
              f"{m['phi_vert_deg']:10.2f}{m['phi_max_deg']:9.2f}{m['s_rmse']:9.3f}{m['gated']:7.0f}")
    print("\nunits: r in m, phi in deg, s in m.  'phi vert' = RMSE on DESCEND/ASCEND only")
    print("(the unexcited legs, where the sonar adds no (r, phi) information)")


if __name__ == "__main__":
    main()
