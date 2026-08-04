# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""CLI for the general comparison figures F1–F6 (plan §10). Runs natively (no Isaac).

Examples::

    python plot_figures.py f1 ../../results/e1
    python plot_figures.py f2 ../../results/e1 --cond nominal --seed 0
    python plot_figures.py f3 ../../results/e2 --metric score.objective
    python plot_figures.py f4 ../../results/e3 --cond step --seed 0 --t-event 90
    python plot_figures.py f5 ../../results/e2 ../../results/e2b --names zero-shot fine-tuned
    python plot_figures.py f6 ../../results/e1 --tuning ../../results/tuning
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
import conftest  # noqa: F401,E402  (installs the isaaclab/marinelab shims)

from marinelab.experiments import figures as F  # noqa: E402
from marinelab.experiments.aggregate import (  # noqa: E402
    _method_key, collect, collect_budgets, summarize,
)

DEFAULT_F1_METRICS = ["score.objective", "cycles_mean", "wall_dist_err_cm", "crab_deg"]


def _summary(results_dir: str, metrics: list[str]):
    rows = collect(results_dir)
    if not rows:
        raise SystemExit(f"no metrics_*.json under {results_dir}")
    return summarize(rows, metrics)


def _cases(results_dir: str, cond: str, seed: int, with_metrics: bool):
    """One (label, npz[, metrics]) per method found for the given condition/seed."""
    pat = re.compile(rf"^trajectory_([^_]+)_{re.escape(cond)}_s{seed}\.npz$")
    found = []
    for name in sorted(os.listdir(results_dir)):
        m = pat.match(name)
        if not m:
            continue
        method = m.group(1)
        npz = os.path.join(results_dir, name)
        if with_metrics:
            found.append((method, npz, os.path.join(
                results_dir, f"metrics_{method}_{cond}_s{seed}.json")))
        else:
            found.append((method, npz))
    if not found:
        raise SystemExit(f"no trajectory_*_{cond}_s{seed}.npz under {results_dir}")
    return sorted(found, key=lambda c: _method_key(c[0]))


def main() -> None:
    p = argparse.ArgumentParser(description="Paper figures from results/ artifacts")
    p.add_argument("fig", choices=["f1", "f2", "f3", "f4", "f5", "f6"])
    p.add_argument("results", nargs="+", help="results dir(s); f5 takes two")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_F1_METRICS, help="f1 panels")
    p.add_argument("--metric", default="score.objective", help="f3/f5 y-axis")
    p.add_argument("--cond", default=None, help="condition (f1/f2/f4/f5)")
    p.add_argument("--seed", type=int, default=0, help="trajectory seed (f2/f4)")
    p.add_argument("--t-event", type=float, default=None, help="f4 event marker [s]")
    p.add_argument("--names", nargs="+", default=None, help="f5 group names per results dir")
    p.add_argument("--tuning", default=None, help="f6 tuning root (budget.json dirs)")
    p.add_argument("--out", default=None, help="output stem (default <results>/fig_<fig>)")
    args = p.parse_args()

    root = args.results[0]
    out = args.out or os.path.join(root, f"fig_{args.fig}")
    if args.fig == "f1":
        paths = F.fig_overlay(_summary(root, args.metrics), args.metrics, out, cond=args.cond)
    elif args.fig == "f2":
        cond = args.cond or "nominal"
        paths = F.fig_trajectory(_cases(root, cond, args.seed, False), out)
    elif args.fig == "f3":
        paths = F.fig_sweep(_summary(root, [args.metric]), args.metric, out)
    elif args.fig == "f4":
        cond = args.cond or "step"
        paths = F.fig_timeseries(_cases(root, cond, args.seed, True), out, t_event=args.t_event)
    elif args.fig == "f5":
        if len(args.results) < 2:
            raise SystemExit("f5 needs two results dirs (zero-shot, fine-tuned)")
        names = args.names or [os.path.basename(os.path.normpath(r)) for r in args.results]
        named = {name: _summary(r, [args.metric]) for name, r in zip(names, args.results)}
        paths = F.fig_zeroshot_ft(named, args.metric, out, cond=args.cond)
    else:  # f6
        offline = collect_budgets(args.tuning) if args.tuning else {}
        # budget dirs are named by tuning method (bo_nmpc/ssi_mpc) -> map to method keys
        offline = {name.split("_")[0]: b for name, b in offline.items()}
        paths = F.fig_cost(offline, _summary(root, ["controller_cost.solve_ms_mean"]), out)
    print("[INFO] wrote " + " ".join(paths))


if __name__ == "__main__":
    main()
