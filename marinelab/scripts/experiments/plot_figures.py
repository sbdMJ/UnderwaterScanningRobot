# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""CLI for the general comparison figures F1–F6 (plan §10). Runs natively (no Isaac).

Examples::

    python plot_figures.py f1 ../../experimental_results/e1
    python plot_figures.py f2 ../../experimental_results/e1 --cond nominal --seed 0
    python plot_figures.py f3 ../../experimental_results/e2 --metric score.objective
    python plot_figures.py f4 ../../experimental_results/e3 --cond step --seed 0 --t-event 90
    python plot_figures.py f5 ../../experimental_results/e2 ../../experimental_results/e2b --names zero-shot fine-tuned
    python plot_figures.py f6 ../../experimental_results/e1 --tuning ../../experimental_results/tuning
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
    METHOD_LABELS, _method_key, collect, collect_budgets, summarize,
)

DEFAULT_F1_METRICS = ["score.objective", "cycles_mean", "wall_dist_err_cm", "crab_deg"]
DEFAULT_F1_LOGY = ("score.objective",)


def _summary(results_dir: str, metrics: list[str]):
    rows = collect(results_dir)
    if not rows:
        raise SystemExit(f"no metrics_*.json under {results_dir}")
    return summarize(rows, metrics)


def _scan_trajectories(results_dir: str, cond: str):
    """method -> {seed: npz_path} for a condition, plus the metrics dir.

    Looks in the structured layout (<exp>/raw + <exp>/metrics) first, falling back to a
    flat directory for legacy/synthetic data.
    """
    raw_dir = os.path.join(results_dir, "raw")
    raw_dir = raw_dir if os.path.isdir(raw_dir) else results_dir
    metrics_dir = os.path.join(results_dir, "metrics")
    metrics_dir = metrics_dir if os.path.isdir(metrics_dir) else results_dir
    pat = re.compile(rf"^trajectory_([^_]+)_{re.escape(cond)}_s(\d+)\.npz$")
    per: dict[str, dict[int, str]] = {}
    for name in sorted(os.listdir(raw_dir)):
        m = pat.match(name)
        if m:
            per.setdefault(m.group(1), {})[int(m.group(2))] = os.path.join(raw_dir, name)
    if not per:
        raise SystemExit(f"no trajectory_*_{cond}_s*.npz under {raw_dir}")
    return per, metrics_dir


def _cases_f2(results_dir: str, cond: str, seed: int):
    """[(method, label, npz)] — one representative seed per method."""
    per, _ = _scan_trajectories(results_dir, cond)
    cases = []
    for method in sorted(per, key=_method_key):
        seeds = per[method]
        npz = seeds.get(seed, seeds[min(seeds)])
        cases.append((method, METHOD_LABELS.get(method, method), npz))
    return cases


def _cases_f4(results_dir: str, cond: str):
    """[(method, label, [npz per seed], metrics_json)] — ALL seeds feed the mean±SD band."""
    per, metrics_dir = _scan_trajectories(results_dir, cond)
    cases = []
    for method in sorted(per, key=_method_key):
        seeds = per[method]
        first = min(seeds)
        cases.append((method, METHOD_LABELS.get(method, method),
                      [seeds[s] for s in sorted(seeds)],
                      os.path.join(metrics_dir, f"metrics_{method}_{cond}_s{first}.json")))
    return cases


def main() -> None:
    p = argparse.ArgumentParser(description="Paper figures from results/ artifacts")
    p.add_argument("fig", choices=["f1", "f2", "f3", "f4", "f5", "f6",
                                   "f7", "f8", "f9", "f10"])
    p.add_argument("results", nargs="+", help="results dir(s); f5 takes two")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_F1_METRICS, help="f1 panels")
    p.add_argument("--metric", default="score.objective", help="f3/f5 y-axis metric key")
    p.add_argument("--ylabel", default=None, help="f3/f5 y-axis label (default: metric key)")
    p.add_argument("--logy", action="store_true", default=False,
                   help="f3/f5: log y-scale (PPO's 8-env objective dwarfs the MPC methods)")
    p.add_argument("--logy-metrics", nargs="+", default=list(DEFAULT_F1_LOGY),
                   help="f1: panels drawn with a log y-axis (default: score.objective)")
    p.add_argument("--cond", default=None, help="condition (f1/f2/f4/f5)")
    p.add_argument("--seed", type=int, default=0, help="trajectory seed (f2/f4)")
    p.add_argument("--t-event", type=float, default=None, help="f4 event marker [s]")
    p.add_argument("--names", nargs="+", default=None, help="f5 group names per results dir")
    p.add_argument("--tuning", default=None, help="f6 tuning root (budget.json dirs)")
    p.add_argument("--out", default=None, help="output stem (default <results>/fig_<fig>)")
    args = p.parse_args()

    root = args.results[0]
    structured = os.path.isdir(os.path.join(root, "metrics"))
    out = args.out or os.path.join(root, "figures" if structured else "", f"fig_{args.fig}")
    if args.fig == "f1":
        paths = F.fig_overlay(_summary(root, args.metrics), args.metrics, out, cond=args.cond,
                              logy_metrics=tuple(args.logy_metrics))
    elif args.fig == "f2":
        cond = args.cond or "nominal"
        paths = F.fig_trajectory(_cases_f2(root, cond, args.seed), out)
    elif args.fig == "f3":
        paths = F.fig_sweep(_summary(root, [args.metric]), args.metric, out,
                            ylabel=args.ylabel, logy=args.logy)
    elif args.fig == "f4":
        cond = args.cond or "step"
        paths = F.fig_timeseries(_cases_f4(root, cond), out, t_event=args.t_event)
    elif args.fig == "f5":
        if len(args.results) < 2:
            raise SystemExit("f5 needs two results dirs (zero-shot, fine-tuned)")
        names = args.names or [os.path.basename(os.path.normpath(r)) for r in args.results]
        named = {name: _summary(r, [args.metric]) for name, r in zip(names, args.results)}
        paths = F.fig_zeroshot_ft(named, args.metric, out, cond=args.cond,
                                  ylabel=args.ylabel, logy=args.logy)
    elif args.fig == "f6":
        offline = collect_budgets(args.tuning) if args.tuning else {}
        # budget dirs are named by tuning method (bo_nmpc/ssi_mpc) -> map to method keys
        offline = {name.split("_")[0]: b for name, b in offline.items()}
        paths = F.fig_cost(offline, _summary(root, ["controller_cost.solve_ms_mean"]), out)
    elif args.fig == "f7":
        cond = args.cond or "nominal"
        per, metrics_dir = _scan_trajectories(root, cond)
        cases = []
        for method in sorted(per, key=_method_key):
            seeds = per[method]
            s = args.seed if args.seed in seeds else min(seeds)
            cases.append((method, METHOD_LABELS.get(method, method), seeds[s],
                          os.path.join(metrics_dir, f"metrics_{method}_{cond}_s{s}.json")))
        paths = F.fig_states(cases, out)
    elif args.fig == "f8":
        paths = F.fig_task(out)
    elif args.fig == "f9":
        cond = args.cond or "step"
        paths = F.fig_pred_error(_cases_f4(root, cond), out)
    else:  # f10
        paths = F.fig_sensitivity(collect(root), out, metric=args.metric,
                                  ylabel=args.ylabel)
    print("[INFO] wrote " + " ".join(paths))


if __name__ == "__main__":
    main()
