# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Re-score saved ``results/trajectory_*.npz`` with the current metric definitions.

Exists because the leg-speed definition was wrong until 2026-07-31 (it averaged ``|d/dt|`` over
every row of a phase, so a truncated leg where the vehicle oscillated while settling was
counted as forward progress — reporting a 16% heave "overshoot" that did not exist). Every
number published from those runs has to be recomputed, and since the trajectories are all on
disk that needs no Isaac Sim and no acados: pure numpy, seconds instead of hours.

Writes ``metrics_<tag>.json`` back in place and prints an old-vs-new table so the size of the
correction is visible rather than silently swapped in.

Usage (native, no container):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python marinelab/scripts/rescore_trajectories.py
    ... --dry-run          # print the table, write nothing
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import types

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# Bare parent packages so marinelab/__init__.py (which pulls in isaaclab) never runs;
# eval_metrics is pure torch/numpy. Same trick as tests/conftest.py.
for _name, _sub in (("marinelab", ""), ("marinelab.tasks", "tasks"),
                    ("marinelab.tasks.pkrc_wallscan", "tasks/pkrc_wallscan")):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(_ROOT, "marinelab", _sub) if _sub else os.path.join(_ROOT, "marinelab")]
        sys.modules[_name] = _m

from marinelab.tasks.pkrc_wallscan import eval_metrics as em  # noqa: E402

SHOW = ["crab_deg", "tilt_heave_deg", "tilt_sway_deg", "heave_speed_mps", "sway_speed_mps",
        "cycles_mean", "wall_dist_err_cm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_ROOT, "..", "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--settle_s", type=float, default=20.0,
                    help="Seconds dropped from each episode start for the 'settled' block.")
    args = ap.parse_args()
    results = os.path.abspath(args.results)

    paths = sorted(glob.glob(os.path.join(results, "trajectory_*.npz")))
    if not paths:
        raise SystemExit(f"no trajectories under {results}")

    hdr = f"{'tag':<28}{'legs h/s':>10}" + "".join(f"{k.replace('_mps','').replace('_deg','').replace('_cm',''):>26}" for k in SHOW)
    print(hdr)
    print("-" * len(hdr))

    for path in paths:
        tag = os.path.basename(path)[len("trajectory_"):-len(".npz")]
        mpath = os.path.join(results, f"metrics_{tag}.json")
        old = {}
        if os.path.isfile(mpath):
            with open(mpath) as fh:
                old = json.load(fh)

        traj = dict(np.load(path))
        step_dt = float(old.get("step_dt", 0.02))
        new = em.compute_metrics(
            traj, step_dt=step_dt,
            d_ref=old.get("d_ref_m"),
            heave_target=old.get("heave_speed_target_mps"),
            sway_target=old.get("sway_speed_target_mps"),
            episode=old.get("scored_episode", 0),
            episode_length_s=old.get("episode_s_nominal"),
            settle_s=args.settle_s,
        )
        # carry forward the provenance fields compute_metrics does not know about
        for k in ("controller", "tam", "hydro", "thruster_tau", "policy_ckpt", "state_source",
                  "seed", "estimator", "mpc", "checkpoint", "task"):
            if k in old:
                new[k] = old[k]

        row = f"{tag:<28}{new.get('heave_legs_scored', 0):>5}/{new.get('sway_legs_scored', 0):<4}"
        for k in SHOW:
            o, n = old.get(k), new.get(k)
            if isinstance(o, (int, float)) and isinstance(n, (int, float)):
                mark = " " if abs(n - o) < 1e-9 else ("*" if abs(n - o) > 0.005 * max(1e-9, abs(o)) else " ")
                row += f"{o:>11.3f}->{n:>11.3f}{mark}"
            else:
                row += f"{'-':>13}{n if n is None else round(float(n), 3):>13} "
        print(row)

        if not args.dry_run:
            with open(mpath, "w") as fh:
                json.dump(new, fh, indent=2)

    print("\n* = changed by more than 0.5%.  'legs h/s' = complete heave/sway legs actually scored.")
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
