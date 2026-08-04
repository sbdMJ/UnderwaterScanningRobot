# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""CLI for ``marinelab.experiments.aggregate``: results dir -> paper-table csv.

Runs natively (no Isaac): ``python aggregate.py ../results/e1 --out table_e1.csv``.
Default metric set covers the Table 1 columns (scan quality + tracking + cost); pass
``--metrics`` with dotted paths to override.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# marinelab/__init__ is heavy (isaaclab); register the bare-package shim first.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
import conftest  # noqa: F401,E402  (installs the isaaclab/marinelab shims)

from marinelab.experiments.aggregate import collect, summarize, write_table_csv  # noqa: E402

# Table 1 columns; top-level keys are eval_metrics.compute_metrics' scored-window values.
DEFAULT_METRICS = [
    "cycles_mean",
    "wall_dist_err_cm",
    "tilt_heave_deg",
    "tilt_sway_deg",
    "crab_deg",
    "s_hat_err_cm",
    "heave_speed_mps",
    "sway_speed_mps",
    "terminations.collided",
    "controller_cost.solve_ms_mean",
    "controller_cost.fail_frac",
    "controller_cost.saturated_frac",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into a table")
    parser.add_argument("results_dir", help="e.g. ../../results/e1")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        help="dotted metric paths inside metrics_*.json")
    parser.add_argument("--out", default=None, help="output csv (default: <results_dir>/table.csv)")
    args = parser.parse_args()

    rows = collect(args.results_dir)
    if not rows:
        raise SystemExit(f"no metrics_*.json found under {args.results_dir}")
    missing = [m for m in args.metrics if not any(m in r for r in rows)]
    if missing:
        print(f"[WARN] metrics never seen in any row (check key paths): {missing}")
    summary = summarize(rows, args.metrics)
    out = args.out or os.path.join(args.results_dir, "table.csv")
    write_table_csv(summary, args.metrics, out)
    print(f"[INFO] {len(rows)} trials -> {out}")
    for (method, cond), entry in summary.items():
        first = args.metrics[0]
        s = entry[first]
        print(f"  {method:6s} {cond:12s} n={entry['n_trials']:2d}  "
              f"{first}={s['mean']:.3g} +- {s['sd']:.3g}")


if __name__ == "__main__":
    main()
