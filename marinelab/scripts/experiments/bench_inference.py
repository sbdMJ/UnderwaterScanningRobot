#!/usr/bin/env python3
"""E4(c) inference-latency bench — acados solve + SSI RFF update, no Isaac Lab.

Runs the SAME shared closed-loop core the hardware node drives (WallScanControlLoop:
phase machine + reference preview + controller step) against a synthetic plant built
from the SAME CasADi model the MPC solves with (RK4, plus a small constant world-frame
disturbance so the SSI learner does real work). What is measured is therefore the full
per-tick compute the Jetson must fit into the 20 ms control budget (50 Hz), not a bare
solver call.

Needs casadi + acados (and torch for the state machine) but NOT isaaclab, so it runs
    desktop:  ./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/experiments/bench_inference.py'
    jetson:   python3 marinelab/scripts/experiments/bench_inference.py   (after the
              aarch64 acados build, docs/experiments/sim-to-real/jetson_acados_build.md)

Reported per method: total step wall time, the acados solve inside it, and their
difference (reference/state-machine overhead; for ssi additionally the RFF update +
casadi RK4 residual predictor). Percentiles over --steps ticks after --warmup.
Writes experimental_results/e4_inference/bench_<label>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import types

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _shim_marinelab_package():
    """Register bare ``marinelab``/``marinelab.tasks`` packages so their __init__
    (which imports the bluerov envs -> isaaclab -> pxr) never runs — same trick as
    tests/conftest.py. Everything this bench touches (control/, tasks.pkrc_wallscan,
    third_party.ssi_mpc_gpl) is pure numpy/casadi/torch underneath."""
    pkg_root = os.path.join(REPO, "marinelab", "marinelab")
    for name, path in (("marinelab", pkg_root),
                       ("marinelab.tasks", os.path.join(pkg_root, "tasks"))):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod


_shim_marinelab_package()
DEF_PLANT = os.path.join(REPO, "marinelab", "config", "pkrc_plant_hw2026.json")
DEF_BO = os.path.join(REPO, "experimental_results", "tuning", "bo_nmpc", "best_params.json")


def build_loop(method: str, plant, args):
    """Mirror _sim_loop.build_controller + WallScanControlLoop for the e5 protocol."""
    from marinelab.control.fixed_nmpc import FixedWeightNMPC
    from marinelab.control.scan_loop import WallScanControlLoop
    from marinelab.tasks.pkrc_wallscan import mpc_reference as mref
    from marinelab.tasks.pkrc_wallscan import scan_state_machine as ssm

    mpc_cfg = mref.WallScanMPCCfg(
        tank_radius=6.0, d_ref=1.5, z_top=8.5, z_bottom=1.0, sway_step=1.0,
        ref_step=0.004, ref_step_s=0.002, step_dt=args.step_dt, dt_mpc=0.05)
    scan_cfg = ssm.ScanCfg(z_top=8.5, z_bottom=1.0, reach_eps=0.6, reach_hold=10,
                           sway_step=1.0, ref_step=0.004, ref_step_s=0.002)
    kw = dict(plant=plant, mpc_cfg=mpc_cfg, horizon=args.horizon, rti_iters=args.rti_iters,
              code_export_root=args.export_root)
    if os.path.exists(DEF_BO):
        kw["params_json"] = DEF_BO  # E5 protocol: both methods run the BO-tuned weights
    if method == "nominal":
        ctl = FixedWeightNMPC(**kw)
    else:
        from marinelab.third_party.ssi_mpc_gpl.ssi_controller import SSIMPCController

        ctl = SSIMPCController(step_dt=args.step_dt, ssi_lr=0.14733286466312384,
                               ssi_kernel_std=0.18398034704266503, **kw)
    return WallScanControlLoop(ctl, scan_cfg, mpc_cfg, horizon=args.horizon, device="cpu")


def make_plant_stepper(plant, max_thrust: float):
    """RK4 one-step propagator from the controller's own CasADi model + a fixed
    world-frame disturbance the model does NOT know about (SSI's residual target)."""
    import casadi as ca

    from marinelab.tasks.pkrc_wallscan.mpc_controller import _continuous_dynamics

    B = np.asarray(plant.allocation_matrix, float)
    x = ca.SX.sym("x", 13)
    u = ca.SX.sym("u", 6)
    d = ca.SX.sym("d", 3)
    dt = ca.SX.sym("dt")
    k1 = _continuous_dynamics(x, u, plant, B, d)
    k2 = _continuous_dynamics(x + 0.5 * dt * k1, u, plant, B, d)
    k3 = _continuous_dynamics(x + 0.5 * dt * k2, u, plant, B, d)
    k4 = _continuous_dynamics(x + dt * k3, u, plant, B, d)
    fn = ca.Function("plant", [x, u, d, dt], [x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])
    dist = np.array([0.3, -0.2, 0.1])  # N, world frame — modest unmodeled push

    def step(x_np, u_cmd, dt_s):
        u_n = np.clip(np.asarray(u_cmd, float), -1, 1) * max_thrust
        out = np.asarray(fn(x_np, u_n, dist, dt_s)).reshape(-1)
        out[3:7] /= np.linalg.norm(out[3:7])
        return out

    return step


def run(method: str, plant, args) -> dict:
    from marinelab.control.types import VehicleState

    loop = build_loop(method, plant, args)
    stepper = make_plant_stepper(plant, plant.max_thrust)
    R = 6.0
    x = np.zeros(13)
    x[0], x[2] = R - 1.5, 5.0   # on-station at theta=0, mid-depth
    x[3] = 1.0                  # identity quat (body +x -> wall)
    veh = VehicleState.from_x13(x)
    loop.reset(veh=None, z0=float(x[2]))
    s, theta_prev = 0.0, 0.0
    total_ms, solve_ms = [], []
    for k in range(args.warmup + args.steps):
        theta = float(np.arctan2(x[1], x[0]))
        dth = (theta - theta_prev + np.pi) % (2 * np.pi) - np.pi
        s += R * dth
        theta_prev = theta
        t0 = time.perf_counter()
        out = loop.step(VehicleState.from_x13(x), s_hat=s, theta_hat=theta)
        t_step = 1e3 * (time.perf_counter() - t0)
        if k >= args.warmup:
            total_ms.append(t_step)
            solve_ms.append(out.solve_ms)
        x = stepper(x, out.u_cmd, args.step_dt)

    def stats(a):
        a = np.asarray(a)
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                "max": float(a.max())}

    total, solve = np.asarray(total_ms), np.asarray(solve_ms)
    res = {"method": method, "steps": args.steps, "total_ms": stats(total),
           "solve_ms": stats(solve), "overhead_ms": stats(total - solve),
           "budget_ms": 1e3 * args.step_dt,
           "over_budget_pct": float((total > 1e3 * args.step_dt).mean() * 100)}
    print(f"\n[{method}] per-tick over {args.steps} steps (budget {res['budget_ms']:.0f} ms):")
    for key in ("total_ms", "solve_ms", "overhead_ms"):
        s_ = res[key]
        print(f"  {key:12s} mean {s_['mean']:6.2f}  p50 {s_['p50']:6.2f}  "
              f"p95 {s_['p95']:6.2f}  p99 {s_['p99']:6.2f}  max {s_['max']:7.2f}")
    print(f"  ticks over budget: {res['over_budget_pct']:.1f}%")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant-json", default=DEF_PLANT)
    ap.add_argument("--methods", default="nominal,ssi")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--rti-iters", type=int, default=8)
    ap.add_argument("--step-dt", type=float, default=0.02)
    ap.add_argument("--label", default=None, help="output tag; default = hostname")
    ap.add_argument("--export-root", default=os.path.join(REPO, "isaaclab", "logs",
                                                          "c_generated_code_bench"))
    args = ap.parse_args()

    from marinelab.tasks.pkrc_wallscan.mpc_controller import PlantParams

    plant = PlantParams.from_json(args.plant_json)
    label = args.label or platform.node() or "host"
    results = {"label": label, "machine": platform.machine(), "plant_json": args.plant_json,
               "horizon": args.horizon, "rti_iters": args.rti_iters,
               "processor": platform.processor(), "methods": {}}
    for m in args.methods.split(","):
        results["methods"][m] = run(m.strip(), plant, args)
    out_dir = os.path.join(REPO, "experimental_results", "e4_inference")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"bench_{label}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
