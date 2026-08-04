# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Single entry point for the competitor experiments (E1/E2/E3/E4a).

Runs every cell of a yaml experiment config — method x condition x seed — through the
``marinelab.control`` adapter layer, writing per-cell trajectory/metrics files in the
naming convention ``marinelab.experiments.protocol`` defines. One process, one Isaac app;
envs are rebuilt per cell (conditions may change the task). The closed loops themselves
live in ``_sim_loop.py``, shared with ``tune.py``.

The method is a REQUIRED positional argument — one method per invocation by default,
because a full config run (5 methods x seeds) is hours of wall-clock. Pass the literal
``all`` to deliberately run every method sequentially.

Examples::

    # one method (the default granularity):
    ./isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py fixed \\
        --config ../marinelab/scripts/experiments/configs/e1_nominal.yaml
    # one cell:
    ... diff --config .../e1_nominal.yaml --seed 0
    # everything (explicit opt-in):
    ... all --config .../e1_nominal.yaml
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Competitor experiment runner")
parser.add_argument("method", type=str,
                    help="method key from the config's methods: (fixed|bo|ppo|ssi|diff|...), "
                         "or 'all' to run every method sequentially")
parser.add_argument("--config", type=str, required=True, help="experiment yaml (see configs/)")
parser.add_argument("--cond", type=str, default=None, help="run only this condition")
parser.add_argument("--seed", type=int, default=None, help="run only this seed")
parser.add_argument("--results_root", type=str, default=None, help="default: <repo>/results")
parser.add_argument("--log_every", type=int, default=500)
parser.add_argument("--no_plot", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json

import numpy as np
import torch

import _sim_loop as sl

from marinelab.experiments.protocol import ExperimentCell, load_cells
from marinelab.experiments.scoring import ScoreAccumulator
from marinelab.tasks.pkrc_wallscan import eval_metrics as em


def run_cell(cell: ExperimentCell, results_root: str) -> None:
    opt = cell.options
    steps = int(opt.get("steps", 9000))
    print(f"\n[CELL] {cell.exp} / {cell.tag}  task={opt['task']}  steps={steps}")
    env, cfg = sl.build_env(cell)
    try:
        ctl, mpc_cfg = sl.build_controller(cell, env, cfg)
        if mpc_cfg is None:  # policy-only methods still need the error-vector cfg for scoring
            mpc_cfg = sl.make_mpc_cfg(cfg, env.step_dt, opt)
        score = ScoreAccumulator(env.num_envs)
        runner = sl.run_ppo_cell if cell.method == "ppo" else sl.run_mpc_cell
        with torch.inference_mode():
            result = runner(cell, env, cfg, ctl, steps, mpc_cfg, score,
                            sim_app=simulation_app, log_every=args_cli.log_every)

        dt = env.step_dt
        traj = result["log"].as_arrays(step_dt=dt)
        sway_step = cfg.scan.ref_step_s if cfg.scan.ref_step_s > 0.0 else cfg.scan.ref_step
        score_episode = opt.get("score_episode", 0)
        metrics = em.compute_metrics(
            traj, step_dt=dt, d_ref=cfg.d_ref,
            heave_target=cfg.scan.ref_step / dt if cfg.scan.ref_step > 0.0 else None,
            sway_target=sway_step / dt if sway_step > 0.0 else None,
            episode=None if score_episode < 0 else score_episode,
            episode_length_s=cfg.episode_length_s,
        )
        score.finalize()
        score_summary = score.summary(score_episode=max(0, score_episode))
        metrics.update({"exp": cell.exp, "method": cell.method, "cond": cell.cond,
                        "seed": cell.seed, "options": {k: v for k, v in opt.items()},
                        "controller_cost": ctl.stats.summary(),
                        "score": {k: score_summary[k] for k in
                                  ("objective", "collided", "scored_losses")}})
        metrics.update(result["extras"])

        out_dir = cell.out_dir(results_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cell.trajectory_path(results_root), **traj)
        with open(cell.metrics_path(results_root), "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(em.format_metrics(metrics))
        cost = metrics["controller_cost"]
        print(f"[SCORE] objective={score_summary['objective']:.4g} "
              f"collided={score_summary['collided']}")
        print(f"[COST] {cost['solve_ms_mean']:.2f} ms/step (p95 {cost['solve_ms_p95']:.2f})  "
              f"fail {100 * cost['fail_frac']:.2f}%  sat {100 * cost['saturated_frac']:.2f}%")
        if not args_cli.no_plot:
            em.plot_trajectory(traj, str(out_dir / f"trajectory_{cell.tag}.png"),
                               episode=None if score_episode < 0 else score_episode,
                               title=f"{cell.method} [{cell.tag}]")
    finally:
        env.close()


def main() -> None:
    only_method = None if args_cli.method == "all" else args_cli.method
    cells = load_cells(args_cli.config, only_method=only_method,
                       only_cond=args_cli.cond, only_seed=args_cli.seed)
    if not cells:
        import yaml

        with open(args_cli.config) as fh:
            available = list(yaml.safe_load(fh).get("methods", {}))
        raise SystemExit(f"no cells match method={args_cli.method!r} "
                         f"(config methods: {available}, or 'all')")
    results_root = os.path.abspath(args_cli.results_root or os.path.join(sl.REPO_ROOT, "results"))
    print(f"[INFO] {len(cells)} cell(s) -> {results_root}")
    for cell in cells:
        run_cell(cell, results_root)


if __name__ == "__main__":
    main()
    simulation_app.close()
