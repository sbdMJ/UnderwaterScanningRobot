#!/usr/bin/env python3
"""Phase C-③ Step 2: reduce bollard-pull measurements to ``newton_per_amp``.

Protocol and rigging: docs/experiments/sim-to-real/thruster_mapping.md §4a
(per-thruster bridle) and §4b (pair runs — the adopted variant when the bridle
attachment spacing w is too narrow). Runs natively (numpy only) — loads the pure
module by file path so the heavy ``marinelab.__init__`` never executes.

Input CSV (header required), one row per steady-state reading::

    thruster,amps,kgf
    1,0.5,0.11      # single-thruster row: thruster 1..6, SIGNED per-thruster amps,
    1,-0.5,0.09     #   unsigned scale reading (magnitude)
    12,1.0,0.43     # PAIR row (12/34/56): both members at the same per-thruster
    12,-1.0,0.35    #   current `amps`; kgf is the TOTAL force -> slope = k_a + k_b
    56,1.0,0.40     # heave pair via the buoyancy-equilibrium method: kgf is the
    ...             #   residual-buoyancy mass balanced at depth hold

Pair sums are split into per-thruster constants with the zero-moment null
currents from the free-float test (``--null PAIR=Ia,Ib`` — the currents at which
the pair yaws/rolls neither way; k_a/k_b = Ib/Ia). A pair without ``--null`` is
split equally with a warning. Single-thruster rows win over pair rows for the
same thruster.

Usage::

    python hw_thrust_calibrate.py bollard_pull.csv
    python hw_thrust_calibrate.py bollard_pull.csv \
        --null 12=1.35,1.50 --null 34=1.50,1.42 --null 56=2.0,2.1
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
split_pair_constants = _tcm.split_pair_constants

LINEARITY_WARN = 0.10  # worst fractional residual before we flag the fit
KEYS = ("1", "2", "3", "4", "5", "6", "12", "34", "56")
PAIRS = {"12": (0, 1), "34": (2, 3), "56": (4, 5)}


def _fit(rows, direction: int) -> tuple[float, float, int] | None:
    """(k, worst_resid, n) for one key and one sign of current, or None."""
    pts = [(a, f) for a, f in rows if (a > 0) == (direction > 0) and a != 0]
    if not pts:
        return None
    amps, newtons = zip(*pts)
    k, resid = fit_thrust_constant(amps, newtons)
    return k, resid, len(pts)


def _parse_null(spec: str) -> tuple[str, float, float]:
    try:
        pair, amps = spec.split("=")
        ia, ib = (float(v) for v in amps.split(","))
    except ValueError:
        raise SystemExit(f"--null wants PAIR=Ia,Ib (e.g. 12=1.35,1.50), got {spec!r}")
    if pair not in PAIRS:
        raise SystemExit(f"--null pair must be one of {sorted(PAIRS)}, got {pair!r}")
    return pair, ia, ib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="bollard-pull readings (thruster,amps,kgf)")
    ap.add_argument("--null", action="append", default=[], metavar="PAIR=Ia,Ib",
                    help="zero-moment null currents from the free-float ratio test")
    ap.add_argument("--amps-limit", nargs=6, type=float,
                    default=[3.0, 3.0, 3.0, 3.0, 5.0, 5.0],
                    help="per-thruster clamp used for the max_thrust suggestion")
    args = ap.parse_args()
    nulls = dict((p, (a, b)) for p, a, b in (_parse_null(s) for s in args.null))

    rows: dict[str, list[tuple[float, float]]] = {k: [] for k in KEYS}
    with open(args.csv, newline="") as fh:
        for row in csv.DictReader(fh):
            key = row["thruster"].strip()
            if key not in rows:
                raise SystemExit(f"thruster column must be one of {KEYS}, got {key!r}")
            rows[key].append((float(row["amps"]), abs(float(row["kgf"])) * G))

    k_avg = np.full(6, np.nan)
    src = [""] * 6
    print(f"{'key':>5} {'k_fwd[N/A]':>11} {'k_rev[N/A]':>11} {'asym':>6} "
          f"{'k_avg[N/A]':>11}  linearity")
    for key in KEYS:
        fwd, rev = _fit(rows[key], +1), _fit(rows[key], -1)
        if fwd is None and rev is None:
            continue
        ks = [x[0] for x in (fwd, rev) if x is not None]
        k_here = float(np.mean(ks))
        asym = f"{abs(ks[0] - ks[-1]) / max(ks):>5.0%}" if len(ks) == 2 else "    —"
        flags = "; ".join(
            f"{name} worst resid {x[1]:.0%} over {x[2]} pts"
            + (" ⚠ drop the lowest-current point and refit" if x[1] > LINEARITY_WARN else "")
            for name, x in (("fwd", fwd), ("rev", rev)) if x is not None)
        col_f = f"{fwd[0]:>11.3f}" if fwd else f"{'—':>11}"
        col_r = f"{rev[0]:>11.3f}" if rev else f"{'—':>11}"
        label = f"T{key}" if len(key) == 1 else f"T{key[0]}+T{key[1]}"
        print(f"{label:>5} {col_f} {col_r} {asym} {k_here:>11.3f}  {flags}")

        if len(key) == 1:  # single-thruster fits always win
            k_avg[int(key) - 1] = k_here
            src[int(key) - 1] = "single"
        else:
            ia, ib = PAIRS[key]
            if key in nulls:
                k_a, k_b = split_pair_constants(k_here, *nulls[key])
                note = f"null ratio k{ia + 1}/k{ib + 1} = {k_a / k_b:.3f}"
            else:
                k_a = k_b = k_here / 2.0
                note = "⚠ no --null given — split EQUALLY (unit spread unaccounted)"
            print(f"       ↳ T{ia + 1} = {k_a:.3f}, T{ib + 1} = {k_b:.3f} N/A  ({note})")
            for idx, val in ((ia, k_a), (ib, k_b)):
                if src[idx] != "single":
                    k_avg[idx], src[idx] = val, "pair"

    if np.any(np.isnan(k_avg)):
        print("\nincomplete — no data for thrusters "
              f"{[i + 1 for i in np.flatnonzero(np.isnan(k_avg))]}")
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
