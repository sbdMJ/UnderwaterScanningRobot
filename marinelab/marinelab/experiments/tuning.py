# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Pure core of the §6 unified tuning protocol (BO-static NMPC and SSI-MPC).

The sim-facing driver is ``scripts/experiments/tune.py``; everything here is
Optuna-agnostic enough to unit-test natively: config loading/validation, search-space
sampling against a trial-like object (anything with ``suggest_float``), and the mandatory
tuning logs — ``trials.csv`` (one row per trial), ``budget.json`` (total effort in the
parent paper's units: trials / episodes / env_steps / wall-clock) and
``best_params.json`` (directly loadable by ``FixedWeightNMPC(params_json=...)``).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REQUIRED_KEYS = ("method", "trials", "steps", "space")


def load_tune_config(path: str) -> dict:
    import yaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"tune config {path} is missing '{key}'")
    for name, spec in cfg["space"].items():
        if "low" not in spec or "high" not in spec:
            raise KeyError(f"space entry '{name}' needs 'low' and 'high'")
    cfg.setdefault("sampler_seed", 0)
    cfg.setdefault("rescore_top_k", 3)
    cfg.setdefault("rescore_steps", 9000)
    return cfg


def suggest_params(trial, space: dict) -> dict:
    """Sample every space entry: {name: {low, high, log?, size?}} -> {name: [floats]}.

    ``size > 1`` samples per-entry parameters ``name_0..name_{size-1}`` — how the 12-D
    ``werr`` vector is searched. ``log`` defaults to True (weights live on decades).
    """
    out = {}
    for name, spec in space.items():
        size = int(spec.get("size", 1))
        log = bool(spec.get("log", True))
        out[name] = [trial.suggest_float(f"{name}_{i}" if size > 1 else name,
                                         float(spec["low"]), float(spec["high"]), log=log)
                     for i in range(size)]
    return out


class TuneRecorder:
    """Writes the §6 tuning logs; every trial is recorded, effort totals are accumulated."""

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trials_csv = self.out_dir / "trials.csv"
        self._totals = {"trials": 0, "episodes": 0, "env_steps": 0, "wall_clock_s": 0.0}

    def record_trial(self, number: int, params: dict, objective: float,
                     episodes: int, env_steps: int, wall_s: float) -> None:
        new_file = not self.trials_csv.exists()
        with open(self.trials_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["trial", "objective", "episodes", "env_steps", "wall_s", "params"])
            w.writerow([number, f"{objective:.6g}", episodes, env_steps, f"{wall_s:.1f}",
                        json.dumps(params)])
        self._totals["trials"] += 1
        self._totals["episodes"] += int(episodes)
        self._totals["env_steps"] += int(env_steps)
        self._totals["wall_clock_s"] += float(wall_s)

    def write_best(self, params: dict, *, objective: float, rescored: bool,
                   trial_number: int) -> None:
        payload = dict(params)  # e.g. {"werr": [...], "wu": [...]} — FixedWeightNMPC-loadable
        payload.update({"objective": float(objective), "rescored": bool(rescored),
                        "trial": int(trial_number)})
        with open(self.out_dir / "best_params.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    def write_budget(self, extra: dict | None = None) -> None:
        payload = dict(self._totals)
        payload.update(extra or {})
        with open(self.out_dir / "budget.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    @property
    def totals(self) -> dict:
        return dict(self._totals)
