#!/usr/bin/env python3
"""Phase C-③ Step 2: reduce bollard-pull measurements to ``newton_per_amp``.

Protocol and rigging: docs/experiments/sim-to-real/thruster_mapping.md §4a.
Runs natively (numpy only) — loads the pure module by file path so the heavy
``marinelab.__init__`` never executes.

Input CSV (header required), one row per steady-state reading::

    thruster,amps,kgf
    1,0.5,0.11      # thruster: 1..6 (T1..T6); amps: SIGNED VESC command;
    1,1.0,0.22      # kgf: unsigned scale reading (magnitude)
    1,-0.5,0.09     # negative amps rows form the reverse-direction fit
    ...

Output: per-thruster k_fwd / k_rev / k_avg with linearity flags, the
``newton_per_amp`` parameter line for the thrust_mapper node, and the suggested
``max_thrust`` (= min_i k_avg_i × amps_limit_i) to align in
``marinelab/config/pkrc_plant_fixed_tam.json``.

Usage::

    python hw_thrust_calibrate.py bollard_pull.csv
    python hw_thrust_calibrate.py bollard_pull.csv --amps-limit 3 3 3 3 5 5
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

G = 9.81  # kgf -> N

_TCM_PATH = (Path(__file__).resolve().parents[2]
             / "marinelab" / "control" / "thrust_current_map.py")
_spec = importlib.util.spec_from_file_location("_tcm", _TCM_PATH)
_tcm = importlib.util.module_from_spec(_spec)
sys.modules["_tcm"] = _tcm  # dataclass machinery needs the module registered
_spec.loader.exec_module(_tcm)
fit_thrust_constant = _tcm.fit_thrust_constant

LINEARITY_WARN = 0.10  # worst fractional residual before we flag the fit


def _fit(rows, direction: int) -> tuple[float, float, int] | None:
    """(k, worst_resid, n) for one thruster and one sign of current, or None."""
    pts = [(a, f) for a, f in rows if (a > 0) == (direction > 0) and a != 0]
    if not pts:
        return None
    amps, newtons = zip(*pts)
    k, resid = fit_thrust_constant(amps, newtons)
    return k, resid, len(pts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="bollard-pull readings (thruster,amps,kgf)")
    ap.add_argument("--amps-limit", nargs=6, type=float,
                    default=[3.0, 3.0, 3.0, 3.0, 5.0, 5.0],
                    help="per-thruster clamp used for the max_thrust suggestion")
    args = ap.parse_args()

    per_thruster: dict[int, list[tuple[float, float]]] = {i: [] for i in range(1, 7)}
    with open(args.csv, newline="") as fh:
        for row in csv.DictReader(fh):
            t = int(row["thruster"])
            if t not in per_thruster:
                raise SystemExit(f"thruster column must be 1..6, got {t}")
            per_thruster[t].append((float(row["amps"]), abs(float(row["kgf"])) * G))

    k_avg = np.full(6, np.nan)
    print(f"{'T':>2} {'k_fwd[N/A]':>11} {'k_rev[N/A]':>11} {'asym':>6} "
          f"{'k_avg[N/A]':>11}  linearity")
    for t in range(1, 7):
        fwd = _fit(per_thruster[t], +1)
        rev = _fit(per_thruster[t], -1)
        if fwd is None and rev is None:
            print(f"{t:>2} {'—':>11} {'—':>11} {'—':>6} {'—':>11}  no data")
            continue
        ks = [x[0] for x in (fwd, rev) if x is not None]
        k_avg[t - 1] = float(np.mean(ks))
        asym = f"{abs(ks[0] - ks[-1]) / max(ks):>5.0%}" if len(ks) == 2 else "    —"
        flags = "; ".join(
            f"{name} worst resid {x[1]:.0%} over {x[2]} pts"
            + (" ⚠ drop the lowest-current point and refit" if x[1] > LINEARITY_WARN else "")
            for name, x in (("fwd", fwd), ("rev", rev)) if x is not None)
        col_f = f"{fwd[0]:>11.3f}" if fwd else f"{'—':>11}"
        col_r = f"{rev[0]:>11.3f}" if rev else f"{'—':>11}"
        print(f"{t:>2} {col_f} {col_r} {asym} {k_avg[t - 1]:>11.3f}  {flags}")

    if np.any(np.isnan(k_avg)):
        print("\nincomplete — fits missing for thrusters "
              f"{[i + 1 for i in np.flatnonzero(np.isnan(k_avg))]}; "
              "parameter lines below assume all six")
        return 1

    limits = np.asarray(args.amps_limit)
    max_thrust = float(np.min(k_avg * limits))
    print("\nthrust_mapper parameters (VESC order T1..T6):")
    print("  newton_per_amp:=\"[" + ", ".join(f"{k:.3f}" for k in k_avg) + "]\"")
    print(f"  max_thrust:={max_thrust:.2f}")
    print(f"\nsuggested max_thrust = min_i(k_avg_i × amps_limit_i) = {max_thrust:.2f} N")
    print("→ set the SAME value in marinelab/config/pkrc_plant_fixed_tam.json "
          "(controller must not plan thrust the drivetrain cannot deliver);")
    binding = int(np.argmin(k_avg * limits)) + 1
    print(f"→ binding thruster: T{binding} at {limits[binding - 1]:.0f} A. "
          "If this lands far below sim's 40 N, raising teleop max_current is the "
          "lever — see thruster_mapping.md §4a step 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
