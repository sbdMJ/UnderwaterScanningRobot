# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Experiment protocol and aggregation for the competitor comparison (pure, no isaaclab).

The sim-facing runner lives in ``marinelab/scripts/experiments/``; this package holds the
parts that must be unit-testable natively: the config -> cell-matrix expansion, the result
naming convention, and the statistics/aggregation used to build the paper tables.
"""
from __future__ import annotations

import importlib

_LAZY = {
    "ExperimentCell": ".protocol",
    "load_cells": ".protocol",
    "ScoreAccumulator": ".scoring",
    "step_losses": ".scoring",
    "TuneRecorder": ".tuning",
    "load_tune_config": ".tuning",
    "suggest_params": ".tuning",
    "collect": ".aggregate",
    "summarize": ".aggregate",
    "write_table_csv": ".aggregate",
    "collect_budgets": ".aggregate",
    "flatten": ".aggregate",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
