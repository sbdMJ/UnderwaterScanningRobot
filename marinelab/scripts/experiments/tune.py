# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Unified auto-tuning driver (plan §6): same pipeline for BO-static NMPC and SSI-MPC.

Optuna TPE over the yaml-declared search space; each trial runs the shared MPC closed
loop (``_sim_loop.run_mpc_cell``) under the shortened protocol and is scored with the
accumulated-task-loss objective (§10-2). One Isaac app and ONE acados build serve all
trials — weights are per-solve parameters. Top-k candidates are re-scored under the full
protocol before ``best_params.json`` is written; every trial lands in ``trials.csv`` and
the total effort in ``budget.json`` (the E4(b) tuning-cost row).

Requires ``optuna`` in the container:  /isaac-sim/python.sh -m pip install optuna

Example::

    ./isaaclab.sh -p ../marinelab/scripts/experiments/tune.py \\
        --config ../marinelab/scripts/experiments/configs/tune_bo_nmpc.yaml
"""
from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Unified baseline tuning (BO-static / SSI-MPC)")
parser.add_argument("--config", type=str, required=True, help="tuning yaml (see configs/)")
parser.add_argument("--out_root", type=str, default=None,
                    help="default: <repo>/experimental_results/tuning/<method>")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import optuna
import torch

import _sim_loop as sl

from marinelab.control.base import ControllerStats
from marinelab.experiments.protocol import ExperimentCell
from marinelab.experiments.scoring import ScoreAccumulator
from marinelab.experiments.tuning import (
    TuneRecorder, load_tune_config, params_from_trial, suggest_params,
)

FAIL_OBJECTIVE = 1e12  # finite stand-in for inf so TPE can still rank failed trials


def main() -> None:
    tcfg = load_tune_config(args_cli.config)
    method = tcfg["method"]
    if method not in ("bo_nmpc", "ssi_mpc"):
        raise SystemExit(f"unknown tuning method {method!r}")

    out_dir = os.path.abspath(args_cli.out_root or os.path.join(
        sl.REPO_ROOT, "experimental_results", "tuning", method))
    recorder = TuneRecorder(out_dir)

    # One env + one solver build for the whole study (weights are per-solve parameters;
    # SSI hyperparameters live outside the solver too).
    options = {
        "task": tcfg.get("task", "Isaac-PKRC-WallScan-Stage3-Direct-v0"),
        "tam": tcfg.get("tam", "fixed"),
        "state": tcfg.get("state", "gt"),
        "num_envs": 1,
        "horizon": int(tcfg.get("horizon", 30)),
        "dt_mpc": float(tcfg.get("dt_mpc", 0.05)),
        "rti_iters": int(tcfg.get("rti_iters", 8)),
    }
    if method == "ssi_mpc" and tcfg.get("inherit_weights"):
        options["params_json"] = tcfg["inherit_weights"]  # §6: SSI starts from BO weights
    cell = ExperimentCell(exp="tuning", method="ssi" if method == "ssi_mpc" else "nominal",
                          cond=method, seed=int(tcfg.get("seed", 0)), options=options)
    env, cfg = sl.build_env(cell)
    ctl, mpc_cfg = sl.build_controller(cell, env, cfg)

    def apply_params(params: dict) -> None:
        if method == "bo_nmpc":
            ctl.set_weights(params["werr"], params["wu"])
        else:  # ssi_mpc: fresh learner with the candidate hyperparameters, fixed seed
            ctl.reconfigure(lr=params["lr"][0], kernel_std=params["kernel_std"][0], seed=0)

    def episode_objective(steps: int) -> tuple[float, int]:
        ctl.stats = ControllerStats()
        score = ScoreAccumulator(1)
        with torch.inference_mode():
            sl.run_mpc_cell(cell, env, cfg, ctl, steps, mpc_cfg, score,
                            sim_app=simulation_app, log_every=0)
        score.finalize()
        summary = score.summary(0)
        obj = summary["objective"]
        return (FAIL_OBJECTIVE if not np.isfinite(obj) else float(obj)), len(score.episodes)

    search_steps = int(tcfg["steps"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, tcfg["space"])
        apply_params(params)
        t0 = time.perf_counter()
        obj, n_episodes = episode_objective(search_steps)
        wall = time.perf_counter() - t0
        recorder.record_trial(trial.number, params, obj, n_episodes, search_steps, wall)
        print(f"[TRIAL {trial.number:3d}] objective={obj:.4g}  ({wall:.0f} s)")
        return obj

    storage = f"sqlite:///{os.path.join(out_dir, 'study.db')}"
    study = optuna.create_study(
        study_name=method, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tcfg["sampler_seed"])),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, int(tcfg["trials"]) - done)
    print(f"[INFO] study {method}: {done} finished, running {remaining} more "
          f"(search {search_steps} steps/trial) -> {out_dir}")
    if remaining:
        study.optimize(objective, n_trials=remaining)

    # Re-score the top-k under the full protocol before committing to a winner.
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(finished, key=lambda t: t.value)[: int(tcfg["rescore_top_k"])]
    rescore_steps = int(tcfg["rescore_steps"])
    best_obj, best_trial, best_params = float("inf"), None, None
    for t in top:
        params = params_from_trial(t.params, tcfg["space"])
        apply_params(params)
        t0 = time.perf_counter()
        obj, n_episodes = episode_objective(rescore_steps)
        wall = time.perf_counter() - t0
        recorder.record_trial(t.number, params, obj, n_episodes, rescore_steps, wall)
        print(f"[RESCORE trial {t.number}] search={t.value:.4g} -> full={obj:.4g}")
        if obj < best_obj:
            best_obj, best_trial, best_params = obj, t.number, params

    if best_params is None:
        raise SystemExit("no completed trials to select from")
    recorder.write_best(best_params, objective=best_obj, rescored=True, trial_number=best_trial)
    recorder.write_budget({"method": method, "search_steps": search_steps,
                           "rescore_steps": rescore_steps,
                           "sampler": "TPE", "sampler_seed": int(tcfg["sampler_seed"])})
    print(f"[DONE] best trial {best_trial} objective {best_obj:.4g}")
    print(f"       -> {out_dir}\\best_params.json (use via e1 yaml: methods.bo.params_json)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
