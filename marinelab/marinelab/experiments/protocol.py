# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Experiment matrix: yaml config -> list of runnable cells, plus the naming convention.

A *cell* is one (experiment, method, condition, seed) combination — the unit both the sim
runner executes and the aggregator groups by. Merge precedence for cell options is
``defaults < condition < method`` (a method override wins over a condition default), chosen
so a method can pin its own artifacts (checkpoint paths) regardless of condition.

Naming convention (compatible with ``rescore_trajectories.py``):
``results/<exp>/trajectory_<method>_<cond>_s<seed>.npz`` and ``metrics_<same>.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentCell:
    exp: str
    method: str
    cond: str
    seed: int
    options: dict = field(default_factory=dict)  # merged defaults/condition/method options

    @property
    def tag(self) -> str:
        return f"{self.method}_{self.cond}_s{self.seed}"

    def out_dir(self, results_root: str) -> Path:
        return Path(results_root) / self.exp

    def trajectory_path(self, results_root: str) -> Path:
        return self.out_dir(results_root) / f"trajectory_{self.tag}.npz"

    def metrics_path(self, results_root: str) -> Path:
        return self.out_dir(results_root) / f"metrics_{self.tag}.json"


def _merged(*layers: dict) -> dict:
    out: dict = {}
    for layer in layers:
        out.update(layer or {})
    return out


def load_cells(config_path: str, *, only_method: str | None = None,
               only_cond: str | None = None, only_seed: int | None = None) -> list[ExperimentCell]:
    """Expand a yaml experiment config into the full method x condition x seed matrix."""
    import yaml

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    for key in ("exp", "methods", "conditions", "seeds"):
        if key not in cfg:
            raise KeyError(f"experiment config {config_path} is missing '{key}'")

    cells = []
    for method, m_opts in cfg["methods"].items():
        if only_method is not None and method != only_method:
            continue
        for cond, c_opts in cfg["conditions"].items():
            if only_cond is not None and cond != only_cond:
                continue
            for seed in cfg["seeds"]:
                if only_seed is not None and int(seed) != only_seed:
                    continue
                cells.append(ExperimentCell(
                    exp=cfg["exp"], method=method, cond=cond, seed=int(seed),
                    options=_merged(cfg.get("defaults"), c_opts, m_opts),
                ))
    return cells
