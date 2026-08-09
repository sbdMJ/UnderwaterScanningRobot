# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Paired per-seed comparison of two ``results/metrics_<tag>_s<seed>.json`` run families.

The wallscan A/B protocol is paired on the seed, not pooled: a seed fixes the whole
domain-randomisation draw (buoyancy, CoB offset, thruster gains, sensor biases), so the
between-arm difference at one seed is a within-vehicle effect while the between-seed spread is
just which vehicle got drawn. Pooling the two hides an effect of a few degrees under a spread of
ten. Reporting rule used here, matching how the repo's earlier NMPC calls were made: an effect is
``decided`` only when every seed's sign agrees.

Metrics come from the ``settled`` block (first ``settle_s`` dropped), because the spawn hands the
controller up to 180 deg of heading error and correcting it is a transient, not scan quality.

Native-only (json + stdlib). Usage::

    python marinelab/scripts/compare_paired_runs.py p3a p3b --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (json path, label, unit, lower_is_better)
SETTLED = (
    ("tilt_heave_deg", "tilt heave", "deg", True),
    ("tilt_sway_deg", "tilt sway", "deg", True),
    ("crab_deg", "crab", "deg", True),
    ("wall_dist_err_cm", "wall err", "cm", True),
    ("heave_speed_mps", "heave speed", "m/s", False),
    ("sway_speed_mps", "sway speed", "m/s", False),
)
MPC = (
    ("saturated_step_frac", "saturated", "frac", True),
    ("solve_fail_frac", "QP fail", "frac", True),
    ("solve_ms_mean", "solve", "ms", True),
)


def load(tag: str, seed: int, out_dir: str) -> dict:
    with open(os.path.join(out_dir, f"metrics_{tag}_s{seed}.json")) as fh:
        return json.load(fh)


def collect(tag: str, seeds, out_dir: str):
    """-> {metric_key: {seed: value}} plus the raw dicts, so callers can audit config drift."""
    vals: dict[str, dict[int, float]] = {}
    raw: dict[int, dict] = {}
    for s in seeds:
        d = raw[s] = load(tag, s, out_dir)
        for key, *_ in SETTLED:
            vals.setdefault(key, {})[s] = d["settled"][key]
        for key, *_ in MPC:
            vals.setdefault(key, {})[s] = d["mpc"][key]
        term = d["terminations"]
        for key in ("collided", "out_of_bounds", "tilted", "time_out"):
            vals.setdefault(key, {})[s] = term[key]
    return vals, raw


def config_diff(raw_a: dict, raw_b: dict) -> list[str]:
    """Every non-weight config field that differs — an A/B is only valid if this is empty."""
    out = []
    for key in ("task", "tam", "hydro", "state_source", "policy_ckpt", "thruster_tau", "num_envs",
                "num_steps", "scored_episode", "settle_s", "d_ref_m"):
        if raw_a.get(key) != raw_b.get(key):
            out.append(f"{key}: {raw_a.get(key)!r} vs {raw_b.get(key)!r}")
    for key in ("horizon", "dt_mpc", "rti_iters", "wu", "spin_search"):
        if raw_a["mpc"].get(key) != raw_b["mpc"].get(key):
            out.append(f"mpc.{key}: {raw_a['mpc'].get(key)!r} vs {raw_b['mpc'].get(key)!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args()

    a, raw_a = collect(args.tag_a, args.seeds, args.out_dir)
    b, raw_b = collect(args.tag_b, args.seeds, args.out_dir)

    s0 = args.seeds[0]
    drift = config_diff(raw_a[s0], raw_b[s0])
    print(f"A = {args.tag_a}   B = {args.tag_b}   seeds = {args.seeds}")
    print("werr A:", raw_a[s0]["mpc"]["werr"])
    print("werr B:", raw_b[s0]["mpc"]["werr"])
    if drift:
        print("!! CONFIG DRIFT — the arms differ in more than the weights:")
        for line in drift:
            print("   ", line)
    else:
        print("config: identical apart from the cost weights (checked "
              "task/tam/hydro/state/ckpt/horizon/...)")

    hdr = f"\n{'metric':14} {'unit':5} " + " ".join(f"{'s'+str(s)+' A':>9} {'s'+str(s)+' B':>9}"
                                                    for s in args.seeds)
    print(hdr + f" {'mean d':>9} {'verdict':>10}")
    for key, label, unit, lower_better in SETTLED + MPC:
        cells, deltas = [], []
        for s in args.seeds:
            va, vb = a[key][s], b[key][s]
            deltas.append(vb - va)
            cells.append(f"{va:9.3f} {vb:9.3f}")
        signs = {d > 0 for d in deltas if abs(d) > 1e-12}
        if not signs:
            verdict = "no change"
        elif len(signs) > 1:
            verdict = "undecided"
        else:
            better = (deltas[0] < 0) == lower_better
            verdict = "B better" if better else "B worse"
        print(f"{label:14} {unit:5} " + " ".join(cells)
              + f" {sum(deltas)/len(deltas):9.3f} {verdict:>10}")

    print(f"\n{'termination':14} {'':5} " + " ".join(f"{'s'+str(s)+' A':>9} {'s'+str(s)+' B':>9}"
                                                     for s in args.seeds))
    for key in ("time_out", "collided", "out_of_bounds", "tilted"):
        cells = " ".join(f"{a[key][s]:9d} {b[key][s]:9d}" for s in args.seeds)
        print(f"{key:14} {'':5} " + cells)


if __name__ == "__main__":
    main()
