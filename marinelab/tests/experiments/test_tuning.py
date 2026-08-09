# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Tuning-protocol core: config validation, space sampling, logs, and weight swapping."""

import csv
import json

import numpy as np
import pytest

from marinelab.control.fixed_nmpc import FixedWeightNMPC, load_weight_params
from marinelab.experiments.tuning import TuneRecorder, load_tune_config, suggest_params

TUNE_YAML = """
method: bo_nmpc
trials: 10
steps: 100
space:
  werr: {size: 3, low: 0.1, high: 100.0}
  wu: {size: 1, low: 0.01, high: 1.0, log: false}
"""


class FakeTrial:
    """Duck-typed optuna trial: returns the geometric/arithmetic midpoint."""

    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, log=True):
        self.calls.append((name, low, high, log))
        return float(np.sqrt(low * high) if log else 0.5 * (low + high))


def test_load_tune_config(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(TUNE_YAML)
    cfg = load_tune_config(str(path))
    assert cfg["method"] == "bo_nmpc"
    assert cfg["rescore_top_k"] == 3 and cfg["rescore_steps"] == 9000  # defaults filled

    path.write_text("method: x\ntrials: 1\nsteps: 1\nspace:\n  a: {low: 1}\n")
    with pytest.raises(KeyError):
        load_tune_config(str(path))


def test_suggest_params_shapes_and_names(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(TUNE_YAML)
    cfg = load_tune_config(str(path))
    trial = FakeTrial()
    params = suggest_params(trial, cfg["space"])
    assert len(params["werr"]) == 3 and len(params["wu"]) == 1
    names = [c[0] for c in trial.calls]
    assert names == ["werr_0", "werr_1", "werr_2", "wu"]  # sized entries get suffixes
    assert trial.calls[0][3] is True and trial.calls[3][3] is False  # log default vs override


def test_recorder_logs_and_budget(tmp_path):
    rec = TuneRecorder(str(tmp_path / "bo_nmpc"))
    rec.record_trial(0, {"werr": [1, 2]}, 5.0, episodes=2, env_steps=100, wall_s=1.5)
    rec.record_trial(1, {"werr": [3, 4]}, 4.0, episodes=1, env_steps=100, wall_s=2.5)
    rec.write_best({"werr": [3, 4], "wu": [0.1]}, objective=4.0, rescored=True, trial_number=1)
    rec.write_budget({"method": "bo_nmpc"})

    with open(tmp_path / "bo_nmpc" / "trials.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and rows[1]["objective"] == "4"

    budget = json.loads((tmp_path / "bo_nmpc" / "budget.json").read_text())
    assert budget["trials"] == 2 and budget["env_steps"] == 200
    assert budget["wall_clock_s"] == pytest.approx(4.0)

    # best_params.json must be directly consumable by the BO controller
    werr, wu = load_weight_params(str(tmp_path / "bo_nmpc" / "best_params.json"))
    np.testing.assert_allclose(werr, [3, 4])
    np.testing.assert_allclose(wu, [0.1])


def test_set_weights_swaps_without_rebuild():
    from marinelab.tasks.pkrc_wallscan.mpc_reference import NE

    class FakeMPC:
        nu = 6
        n_pglobal = NE + 6

    ctl = FixedWeightNMPC(mpc=FakeMPC())
    new_werr = np.arange(1.0, NE + 1)
    ctl.set_weights(new_werr, [0.7])
    np.testing.assert_allclose(ctl.weights[:NE], new_werr)
    np.testing.assert_allclose(ctl.weights[NE:], 0.7)
    with pytest.raises(ValueError):
        ctl.set_weights([1.0], [0.7])
