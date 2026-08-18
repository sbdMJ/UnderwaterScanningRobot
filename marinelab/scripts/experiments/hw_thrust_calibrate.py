#!/usr/bin/env python3
"""Phase C-③ Step 2: reduce bollard-pull measurements to ``newton_per_amp``.

Protocol and rigging: docs/experiments/sim-to-real/thruster_mapping.md §4b (pair
runs) / §4c (heave buoyancy method). Runs natively (numpy only) — loads the pure
module by file path so the heavy ``marinelab.__init__`` never executes.

Model: the 2026-08-11 measurements read zero thrust at 0.5 A everywhere, so the
fit is the deadzone-affine ``F = k·(I − I₀)`` (friction torque must be overcome
first), not through-origin. Zero-kgf rows are kept as deadzone consistency checks.

Input CSV (header required), one row per steady-state reading::

    thruster,amps,kgf
    1,0.5,0.11      # single-thruster row: thruster 1..6, SIGNED per-thruster amps,
    12,1.0,0.105    # PAIR row (12/34/56): both members at the same per-thruster
    12,0.5,0.0      #   current; kgf is the TOTAL force -> slope = k_a + k_b.
    56,1.0,0.40     #   Zero readings mark the deadzone. Heave pair rows may come
    ...             #   from the buoyancy-equilibrium method (kgf = balanced mass).

Pair sums are split with the zero-moment null currents (``--null PAIR=Ia,Ib``,
k_a/k_b = Ib/Ia); a pair without ``--null`` splits equally with a warning.
Single-thruster rows win over pair rows. ``--fill-missing-from-average`` copies
the average (k, I₀) into unmeasured thrusters — flagged ASSUMED in the output.

Usage::

    python hw_thrust_calibrate.py bollard_pull.csv \
        --null 12=1.35,1.50 --fill-missing-from-average
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
fit_thrust_affine = _tcm.fit_thrust_affine
split_pair_constants = _tcm.split_pair_constants

KEYS = ("1", "2", "3", "4", "5", "6", "12", "34", "56")
PAIRS = {"12": (0, 1), "34": (2, 3), "56": (4, 5)}
DEADZONE_TOL = 0.15  # A — zero-kgf rows may exceed fitted I0 by this before we flag


def _fit(rows, direction: int):
    """(k, i0, worst_resid_N, n_live, zero_amps) for one sign of current, or None."""
    pts = [(a, f) for a, f in rows if (a > 0) == (direction > 0) and a != 0]
    if sum(1 for _, f in pts if f > 0) < 2:
        return None
    amps, newtons = zip(*pts)
    k, i0, resid = fit_thrust_affine(amps, newtons)
    zeros = [abs(a) for a, f in pts if f == 0.0]
    return k, i0, resid, len(pts), zeros


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
    ap.add_argument("--fill-missing-from-average", action="store_true",
                    help="assume unmeasured thrusters match the average (k, I0)")
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
    i0_avg = np.full(6, np.nan)
    src = [""] * 6
    print(f"{'key':>6} {'dir':>3} {'k[N/A]':>8} {'I0[A]':>6} {'resid[N]':>9}  notes")
    for key in KEYS:
        fits = {}
        for name, sign in (("fwd", +1), ("rev", -1)):
            r = _fit(rows[key], sign)
            if r is None:
                continue
            fits[name] = r
            k, i0, resid, n, zeros = r
            notes = []
            if zeros and max(zeros) > i0 + DEADZONE_TOL:
                notes.append(f"⚠ zero reading at {max(zeros):.2f} A above fitted I0")
            label = f"T{key}" if len(key) == 1 else f"T{key[0]}+T{key[1]}"
            print(f"{label:>6} {name:>3} {k:>8.3f} {i0:>6.3f} {resid:>9.3f}  "
                  f"{len(zeros)} deadzone pts; {'; '.join(notes)}")
        if not fits:
            continue
        k_here = float(np.mean([f[0] for f in fits.values()]))
        i0_here = float(np.mean([f[1] for f in fits.values()]))
        if len(fits) == 2:
            ks = [fits["fwd"][0], fits["rev"][0]]
            print(f"{'':>6} avg {k_here:>8.3f} {i0_here:>6.3f} "
                  f"{'':>9}  fwd/rev asym {abs(ks[0] - ks[1]) / max(ks):.0%}")

        if len(key) == 1:  # single-thruster fits always win
            idx = int(key) - 1
            k_avg[idx], i0_avg[idx], src[idx] = k_here, i0_here, "single"
        else:
            ia, ib = PAIRS[key]
            if key in nulls:
                k_a, k_b = split_pair_constants(k_here, *nulls[key])
                note = f"null ratio k{ia + 1}/k{ib + 1} = {k_a / k_b:.3f}"
            else:
                k_a = k_b = k_here / 2.0
                note = "⚠ no --null — split EQUALLY (unit spread unaccounted)"
            print(f"       ↳ T{ia + 1} = {k_a:.3f}, T{ib + 1} = {k_b:.3f} N/A, "
                  f"I0 = {i0_here:.3f} A each  ({note})")
            for idx, val in ((ia, k_a), (ib, k_b)):
                if src[idx] != "single":
                    k_avg[idx], i0_avg[idx], src[idx] = val, i0_here, "pair"

    missing = np.flatnonzero(np.isnan(k_avg))
    if missing.size and args.fill_missing_from_average:
        k_fill = float(np.nanmean(k_avg))
        i0_fill = float(np.nanmean(i0_avg))
        for idx in missing:
            k_avg[idx], i0_avg[idx], src[idx] = k_fill, i0_fill, "ASSUMED"
        print(f"\nASSUMED from average for thrusters {[i + 1 for i in missing]}: "
              f"k = {k_fill:.3f} N/A, I0 = {i0_fill:.3f} A — verify (heave: §4c "
              "buoyancy method) before trusting closed-loop heave authority")
    elif missing.size:
        print(f"\nincomplete — no data for thrusters {[i + 1 for i in missing]} "
              "(use --fill-missing-from-average to proceed anyway)")
        return 1

    limits = np.asarray(args.amps_limit)
    max_thrust = float(np.min(k_avg * (limits - i0_avg)))
    print("\nthrust_mapper parameters (VESC order T1..T6):")
    print("  newton_per_amp:=\"[" + ", ".join(f"{k:.3f}" for k in k_avg) + "]\"")
    print("  amps_offset:=\"[" + ", ".join(f"{v:.3f}" for v in i0_avg) + "]\"")
    print(f"  max_thrust:={max_thrust:.2f}")
    print(f"\nsuggested max_thrust = min_i(k_i × (amps_limit_i − I0_i)) = {max_thrust:.2f} N")
    print("→ set the SAME value in marinelab/config/pkrc_plant_fixed_tam.json "
          "(controller must not plan thrust the drivetrain cannot deliver);")
    binding = int(np.argmin(k_avg * (limits - i0_avg))) + 1
    print(f"→ binding thruster: T{binding} at {limits[binding - 1]:.0f} A. "
          "If this lands far below sim's 40 N, raising teleop max_current is the "
          "lever — see thruster_mapping.md §4a step 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
